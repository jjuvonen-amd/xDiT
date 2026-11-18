# torch symm_mem AsyncTP on ROCm SDMA engines (prototype patch)

`rocm_sdma_async_tp.patch` gives PyTorch's fused AsyncTP ops an opt-in ROCm path that moves the
data on SDMA copy engines:
- `torch.ops.symm_mem.fused_all_gather_matmul` and `fused_matmul_reduce_scatter`;
- their fp8 variants, `fused_all_gather_scaled_matmul` and `fused_scaled_matmul_reduce_scatter`.

Code that already calls these ops (directly, or through `torch.compile`'s micro-pipeline TP pass)
picks it up with one environment variable and no change of its own.

## Why

For large inputs, the default implementations copy shards out of peers' symmetric-memory buffers
with tensor copies on two alternating streams. On ROCm, HIP runs concurrent copies from those
buffers one at a time, so the all-gather moves at roughly one link's rate. The copies also compete
with the matmuls for compute units.

With the patch, every rank pushes its data into each peer's inbox with ROCr's
`hsa_amd_memory_async_copy_on_engine`:
- one SDMA engine per peer (the engine ROCr prefers for writing to that peer);
- all peers at once;
- in pieces, with each piece's matmul starting as soon as that piece has landed.

The copies use no compute units. Each rank's rows land in natural order, so the results, including
`return_A` and the output strides, are the same as the default path's.

The fp8 variants use the same pipeline. Row-wise scales that are sharded with A travel in the same
pushes as A's rows, under the same flag, so a piece's matmul never starts before its scales have
landed. `_scaled_mm` takes matrices only, so each piece runs one `_scaled_mm` per rank instead of
one batched matmul.

Results are deterministic: the copies only move data, and the reduce-scatter sums every piece's
partial sums in rank order, in fp32. Reruns with the same inputs are bitwise equal.

## Apply

The patch is Python plus one HIP source file. The transport is built with
`torch.utils.cpp_extension` on first use, so the installed torch does not need a rebuild.

```bash
bash patches/pytorch/apply_to_installed_torch.sh            # patch the active environment's torch
bash patches/pytorch/apply_to_installed_torch.sh --check    # applied or not
bash patches/pytorch/apply_to_installed_torch.sh --revert   # restore it
```

The script applies only the `torch/` part of the patch, and checks that it applies cleanly before
changing anything. In a PyTorch checkout, `git am rocm_sdma_async_tp.patch` applies the whole patch,
tests included, at the `base-commit` named at the end of the patch.

Requirements:
- a ROCm build of torch with symmetric memory;
- hipcc and ninja at first use (the build is cached under `~/.cache/torch_extensions`);
- the GPUs of a group in one node.

## Enable

| Variable | Default | Meaning |
|---|---|---|
| `TORCH_SYMM_MEM_ENABLE_ROCM_SDMA` | `0` | `1` turns the SDMA path on |
| `TORCH_SYMM_MEM_ROCM_SDMA_CHUNKS` | by size | pieces per rank (at most 16; see Pieces); fewer if they do not divide the rows |
| `TORCH_SYMM_MEM_ROCM_SDMA_TIMEOUT_S` | `30` | how long a call waits for its flags, or for the copies of the call that last used its slot, before it is reported (see Failures) |
| `TORCH_SYMM_MEM_ROCM_SDMA_ON_TIMEOUT` | `abort` | what happens once a failure is reported. `abort`: abort the process within about 0.1 s, logging what was missing. `raise`: raise on the next eager call. Either way the stream moves on, and that call's output is wrong |
| `TORCH_SYMM_MEM_ROCM_SDMA_ALLOW_UNTESTED` | `0` | `1` also runs on GPU architectures other than gfx950 |
| `TORCH_SYMM_MEM_ROCM_SDMA_INBOX` | `ipc` | where the copies land (see Inbox). `ipc`: device memory mapped with hipIpc handles, no symmetric-memory rendezvous. `symm_mem`: the group's symmetric-memory workspace |

The SDMA path is used for any dtype when the gather or scatter dimension is not the last one and
the group has 2 to 64 ranks. The fp8 variants take it with tensor-wise or row-wise scales. Everything
else runs the default path, with no error:
- the last-dimension variants;
- a scaled reduce-scatter whose A was reshaped with the scatter dimension other than 0, or whose
  `out_dtype` is not bf16, fp16 or fp32;
- other architectures (unless `ALLOW_UNTESTED` is set);
- non-ROCm builds.

## Pieces

By default each call picks its number of pieces per rank from the bytes each rank pushes per call
(its A rows for the all-gather, an owner's output rows for the reduce-scatter): the most pieces,
up to 8 and a power of two, that keep each piece at least 16 MiB. More pieces start the first
matmul sooner and leave less of the transfer exposed at the end. Pieces much below 16 MiB lose
more to per-piece costs (flag waits, engine launches, and for the fp8 variants one `_scaled_mm`
per rank's piece) than they gain. On gfx950, with Wan2.2 shapes at TP4 and TP8, this count was
the fastest or within a few percent of it in `benchmarks/tp_comm_gemm_overlap.py`, for bf16 and
fp8. The fp8 all-gather moves half the bytes of bf16, so it gets half as many pieces. Set
`TORCH_SYMM_MEM_ROCM_SDMA_CHUNKS` to fix the count; the benchmark's `--chunks` sweeps it.

## Inbox

Each rank's copies land in its peers' inboxes, and a call passes a barrier before its copies start
and another after its last wait.
- `ipc` (default): each rank's inbox is plain device memory (`hipMalloc`), which the other ranks'
  processes map with hipIpc handles, as RCCL does. The barriers are a one-thread kernel on arrival
  words mapped the same way. The path never rendezvouses symmetric memory, and its barriers are
  bounded like the flag waits.
- `symm_mem`: the inbox is the group's symmetric-memory workspace (`get_symm_mem_workspace`) and the
  barriers are its signal-pad barriers, as in earlier versions of this patch.

Either way the copies and the matmuls are the same, and so are the results.

Driver note: on some amdgpu 6.x kernel drivers, a symmetric-memory rendezvous (a dma-buf import of
the peers' memory) moves the memory to the host and degrades later inter-GPU traffic, RCCL included,
until a reboot. With the `ipc` inbox the SDMA path avoids it, but calls that run the default path
(see Enable) still rendezvous; the path logs a warning once when that happens. Prefer amdgpu 7.x or
later for workloads that use the default path.

## Memory

Each call takes its inbox from the group's workspace, which grows to the largest call so far and is
kept:
- all-gather: the gathered A (`world * rows * K` elements), plus the gathered row-wise scales for
  the fp8 variant with sharded scales. The fp8 variant always returns A (its schema does), as a copy
  of the inbox.
- reduce-scatter: this rank's share of every rank's partial sums, `world * rows * N` elements in the
  output dtype, plus as much again for this rank's own partial sums before they are pushed. That is
  twice the unreduced product, against once for the default path.

An `ipc` inbox grows to at least twice its size. The memory it outgrew stays mapped until the process
exits, because graphs captured earlier may still use it, so the total is at most about twice the
largest inbox.

## Failures

No wait on the GPU or the host is unbounded:
- A flag wait reports flags that did not land within the timeout. A call's later waits then
  return at once, so a call spends at most one timeout in its waits.
- Waiting for the copies of a slot's previous call is bounded by the timeout too.
- The call's barriers are bounded at three times the timeout. An `ipc` barrier that runs out of
  time is reported like a flag wait, naming the ranks that did not arrive; a `symm_mem` one faults
  the GPU.

A process does not leave copies waiting on its GPU's SDMA engines. The transport drains:
- when an exception, including `KeyboardInterrupt`, stops an eager call after its copies were
  issued;
- before the watchdog aborts the process;
- at exit;
- on `SIGTERM`. The transport installs a handler when it is set up. The handler drains, then
  runs the previous disposition.

A drain does three things:
1. It releases every copy that is still waiting for its turn.
2. It gives the copies 5 s to finish.
3. It makes the flag writes still to come carry a value that marks the call as abandoned. A peer
   waiting for one of those flags reports it at once instead of at the timeout.

After an exception, the rank still passes the call's last barrier, so its peers do not wait there
until the barrier times out. Then it raises. If the
copies did not finish within the drain, every later call on this path raises.

The transport is built under a name derived from its source, and it checks the version of the
library it loads. A build of another version of the patch is never picked up.

`SIGKILL` can still leave copies waiting. So can `SIGTERM` if the application replaces the
`SIGTERM` handler after the transport is set up, unless it exits through Python's normal exit
path.

## CUDA graphs

Captured calls are supported. A captured call issues its copies from a host-function node, so
every replay issues them when the stream reaches the call. As with the other symm_mem ops, the
inbox cannot grow during capture: run each op once, eagerly, with the shapes it will be captured
with.

## Tests

`test/distributed/test_symmetric_memory.py` gains `AsyncTPRocmSdmaTest`:
- the existing AsyncTP cases with the path on, each checking that the path ran;
- bf16 with a batch, several pieces and two Bs;
- the default piece count for a range of sizes;
- the fp8 variants with each scale mode, against the default path, with random (not constant)
  scales and pieces whose scales are not 16-byte aligned;
- a scaled reduce-scatter that is left to the default path;
- bitwise-equal reruns of all four ops;
- graph capture with replays on new inputs, bf16 and fp8;
- two process groups;
- a lost flag that is reported instead of hanging;
- an exception on one rank in the middle of a call, for both ops: the rank raises, its peers
  report the abandoned call without waiting for the timeout, and the next calls are right;
- the core cases above with each inbox, `ipc` and `symm_mem`;
- all four ops on a new group with the `ipc` inbox while symmetric-memory rendezvous raises;
- an `ipc` inbox that grows, with calls of both sizes right after;
- a rank that reaches a barrier late: its peers report it, and later calls are right;
- the version of the loaded transport.

`benchmarks/wan_tp/collect_torch_sdma.sh` in this repository runs those tests and the existing
`AsyncTPTest`, with the path off and on, on an installed torch. It also runs:
- Wan2.2 parity tests that use the stock ops on this path;
- a per-site benchmark;
- Wan2.2 T2V end-to-end runs.

## Demo with xDiT

xDiT's Wan2.2 tensor parallelism calls the stock ops with `XDIT_TP_COMM_OVERLAP=symm_mem`:

```bash
bash patches/pytorch/apply_to_installed_torch.sh
# per site, stock ops vs the same ops on SDMA, eager and replayed from a graph
torchrun --nproc_per_node 8 benchmarks/tp_comm_gemm_overlap.py --symm-mem --symm-mem-sdma --graph --chunks 2 4 8
# (symm_mem_sdma_auto is the default piece count) the same in fp8 with row-wise scales, next to bf16
torchrun --nproc_per_node 8 benchmarks/tp_comm_gemm_overlap.py --symm-mem --symm-mem-sdma --fp8 --chunks 2 4 8
# Wan2.2 T2V TP8, the same command with the path off and on
PROFILE=0 STEPS=40 bash benchmarks/wan_tp/run_profiles.sh tp8_xrepl_fused
TORCH_SYMM_MEM_ENABLE_ROCM_SDMA=1 PROFILE=0 STEPS=40 bash benchmarks/wan_tp/run_profiles.sh tp8_xrepl_fused
```

## Toward an upstream PR

This patch is a prototype for discussion with the PyTorch ROCm maintainers. It shows that on ROCm
the fused AsyncTP ops can overlap their communication with their matmuls, which the default path
does not do there (see Why), without changing their callers. An upstream version would need:
- the transport in C++ under `torch/csrc/distributed/c10d/symm_mem/`, as a ROCm-only source,
  instead of a JIT-built library called through ctypes. The Python side would stay: it only plans
  the pieces and runs the matmuls;
- ROCr version guards;
- reporting through the distributed watchdog instead of its own thread;
- the scaled reduce-scatter's remaining layouts (A reshaped with another scatter dimension);
- a decision on when the path is on by default, and a piece-count rule checked on more GPUs.
