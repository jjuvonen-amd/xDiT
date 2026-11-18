"""A symmetric workspace shared with hipIpc instead of torch symmetric-memory rendezvous (ROCm).

`symm_mem.get_symm_mem_workspace()` shares its buffer through `rendezvous()`, which imports the
peers' memory as dma-buf. On amdgpu 6.16.x (kernel 5.15) that import migrates the buffer from
VRAM to system memory (reads at ~55 GB/s instead of ~2.5 TB/s), and after the process exits the
node's RCCL stays degraded until a reboot. amdgpu 7.1.x keeps it in VRAM.

`IpcWorkspace` provides what the TP SDMA paths use of torch's workspace (`get_buffer` on any
rank's buffer, `barrier`) over one `hipMalloc` allocation per rank, shared with
`hipIpcGetMemHandle` / `hipIpcOpenMemHandle`: the KFD IPC path RCCL and the Ulysses SDMA
transport use, which maps the peers' VRAM directly on every driver.

- `get_workspace(group_name, min_size)`: the drop-in for `get_symm_mem_workspace`, used by
  layers/sdma_all_gather.py (XDIT_TP_COMM_OVERLAP=sdma).
- `install_torch_sdma_inbox()`: points the inbox of torch's patched ROCm SDMA ops
  (`torch.distributed._symmetric_memory._rocm_sdma`, patches/pytorch; XDIT_TP_COMM_OVERLAP=symm_mem)
  at this workspace. A patch with transport ABI 3 or later shares its inbox over hipIpc itself,
  and keeps it unless TORCH_SYMM_MEM_ROCM_SDMA_INBOX=symm_mem.

XDIT_TP_IPC_WORKSPACE=auto (default) uses them on amdgpu < 7 or an unknown amdgpu version, 1 on
any ROCm driver, 0 never (torch's workspace).

Growing a workspace is collective: every rank of the group makes the same calls in the same
order, as the ops using it already require.
"""

import ctypes
import math
from typing import Dict, Optional

import torch
import torch.distributed as dist

import xfuser.envs as envs
from xfuser.logger import init_logger

logger = init_logger(__name__)

_ALIGN = 2 << 20
_hip = None
_workspaces: Dict[str, "IpcWorkspace"] = {}


class _IpcHandle(ctypes.Structure):
    # hipIpcMemHandle_t: a 64-byte struct passed by value (a c_char array argument would be
    # passed as a pointer and rejected with hipErrorInvalidValue).
    _fields_ = [("reserved", ctypes.c_char * 64)]


def _lib():
    global _hip
    if _hip is None:
        try:
            lib = ctypes.CDLL("libamdhip64.so")
        except OSError:
            lib = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
        lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.hipFree.argtypes = [ctypes.c_void_p]
        lib.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
        lib.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
        lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        _hip = lib
    return _hip


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with hipError {rc}")


class _DeviceArray:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 2,
            "strides": None,
        }


_enabled: Optional[bool] = None


def amdgpu_version() -> Optional[tuple]:
    """The loaded amdgpu module's version, e.g. (6, 16, 6), or None (in-tree driver, or not readable)."""
    try:
        with open("/sys/module/amdgpu/version") as f:
            return tuple(int(p) for p in f.read().strip().split(".")[:3])
    except (OSError, ValueError):
        return None


def enabled() -> bool:
    """Whether the TP SDMA paths use this workspace. XDIT_TP_IPC_WORKSPACE: auto (default) uses it on
    ROCm with amdgpu < 7 or an unknown amdgpu version (the IPC path is safe on every driver, torch's
    rendezvous is not on 6.16.x); 1 always on ROCm; 0 never."""
    global _enabled
    if _enabled is None:
        mode = envs.environment_variables["XDIT_TP_IPC_WORKSPACE"]().strip().lower()
        if mode not in ("auto", "0", "1"):
            raise ValueError(f"XDIT_TP_IPC_WORKSPACE must be auto, 0 or 1; got {mode!r}")
        version = amdgpu_version()
        if torch.version.hip is None or mode == "0":
            _enabled = False
        elif mode == "1":
            _enabled = True
        else:
            _enabled = version is None or version < (7,)
        if torch.version.hip is not None:
            logger.info(
                "TP SDMA inboxes: %s (XDIT_TP_IPC_WORKSPACE=%s, amdgpu %s)",
                "hipIpc workspace" if _enabled else "torch symmetric-memory workspace",
                mode,
                ".".join(map(str, version)) if version else "unknown",
            )
    return _enabled


class IpcWorkspace:
    """`size` bytes per rank of a process group, every rank's buffer mapped on every rank."""

    def __init__(self, group_name: str, size: int):
        lib = _lib()
        self.group = dist.distributed_c10d._resolve_process_group(group_name)
        self.rank, self.world = self.group.rank(), self.group.size()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.size = -(-size // _ALIGN) * _ALIGN
        ptr = ctypes.c_void_p()
        _check(lib.hipMalloc(ctypes.byref(ptr), self.size), "hipMalloc")
        self.ptr = ptr.value
        _check(lib.hipMemset(self.ptr, 0, self.size), "hipMemset")
        handle = _IpcHandle()
        _check(lib.hipIpcGetMemHandle(ctypes.byref(handle), self.ptr), "hipIpcGetMemHandle")
        handles = [None] * self.world
        dist.all_gather_object(handles, bytes(handle), group=self.group)
        self.bases, self._opened = [], []
        for r in range(self.world):
            if r == self.rank:
                self.bases.append(self.ptr)
                continue
            peer = ctypes.c_void_p()
            _check(
                lib.hipIpcOpenMemHandle(ctypes.byref(peer), _IpcHandle.from_buffer_copy(handles[r]), 1),
                f"hipIpcOpenMemHandle (rank {r})",
            )
            self.bases.append(peer.value)
            self._opened.append(peer.value)
        self._one = torch.ones(1, dtype=torch.int32, device=self.device)
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def get_buffer(self, rank: int, shape, dtype: torch.dtype, storage_offset: int = 0) -> torch.Tensor:
        """`shape` of `dtype` at element `storage_offset` of `rank`'s buffer, as in torch's workspace."""
        n = math.prod(shape)
        flat = torch.as_tensor(_DeviceArray(self.bases[rank], self.size), device=self.device).view(dtype)
        if storage_offset + n > flat.numel():
            raise ValueError(f"{n} elements at {storage_offset} exceed the {self.size}-byte workspace")
        return flat[storage_offset : storage_offset + n].view(tuple(shape))

    def barrier(self, channel: int = 0, timeout_ms: Optional[int] = None) -> None:
        """Stream-ordered: work queued before it on every rank completes before work queued after it."""
        dist.all_reduce(self._one, group=self.group)

    def free(self) -> None:
        """Collective. Every rank first finishes all use of the group's buffers."""
        lib = _lib()
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        for peer in self._opened:
            lib.hipIpcCloseMemHandle(peer)
        self._opened = []
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        lib.hipFree(self.ptr)


def get_ipc_workspace(group_name: str, min_size: int) -> IpcWorkspace:
    """The group's workspace of at least `min_size` bytes per rank (collective when it grows)."""
    ws = _workspaces.get(group_name)
    if ws is None or ws.size < min_size:
        size = max(min_size, 2 * ws.size if ws is not None else 0)
        if ws is not None:
            ws.free()
        ws = _workspaces[group_name] = IpcWorkspace(group_name, size)
    return ws


def get_workspace(group_name: str, min_size: int):
    """The IPC workspace when enabled(), else torch's symmetric-memory workspace."""
    if enabled():
        return get_ipc_workspace(group_name, min_size)
    import torch.distributed._symmetric_memory as symm_mem

    return symm_mem.get_symm_mem_workspace(group_name, min_size=min_size)


def _torch_sdma_inbox(group_name: str, rank: int, world: int, regions, chunks: int):
    """`_rocm_sdma._inbox` on the IPC workspace: same layout and return values (regions 64-byte
    aligned, then (chunks, world) int32 flag words; every rank's base; the offsets)."""
    offsets, size = [], 0
    for shape, dtype in regions:
        offsets.append(size)
        size += -(-(math.prod(shape) * dtype.itemsize) // 64) * 64
    flag_offset = size
    size += chunks * world * 4
    ws = get_ipc_workspace(group_name, size)
    bufs = [
        ws.get_buffer(rank, tuple(shape), dtype, storage_offset=offset // dtype.itemsize)
        for (shape, dtype), offset in zip(regions, offsets)
    ]
    flags = ws.get_buffer(rank, (chunks, world), torch.int32, storage_offset=flag_offset // 4)
    return ws, bufs, offsets, flags, list(ws.bases), flag_offset


def install_torch_sdma_inbox() -> bool:
    """Point torch's patched ROCm SDMA ops at the IPC workspace, if they are installed and enabled()."""
    if not enabled():
        return False
    try:
        import torch.distributed._symmetric_memory._rocm_sdma as rocm_sdma
    except ImportError:
        return False
    if getattr(rocm_sdma, "ABI_VERSION", 0) >= 3 and rocm_sdma._inbox_mode() == "ipc":
        # The patch's own hipIpc inbox, with its GPU barrier (this workspace's barrier is an RCCL
        # all_reduce).
        return True
    if getattr(rocm_sdma, "_xfuser_orig_inbox", None) is None:
        rocm_sdma._xfuser_orig_inbox = rocm_sdma._inbox
        rocm_sdma._inbox = _torch_sdma_inbox
        logger.info("torch's ROCm SDMA symm_mem ops use the hipIpc workspace")
    return True
