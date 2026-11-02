# torch symm_mem AsyncTP on ROCm SDMA engines (prototype patch)

`rocm_sdma_async_tp.patch` gives PyTorch's fused AsyncTP ops,
`torch.ops.symm_mem.fused_all_gather_matmul` and `torch.ops.symm_mem.fused_matmul_reduce_scatter`,
an opt-in ROCm path that moves the data on SDMA copy engines. Code that already calls these ops
(directly, or through `torch.compile`'s micro-pipeline TP pass) picks it up with one environment
variable and no change of its own.

## Why

For large inputs, the default implementations copy shards out of peers' symmetric-memory buffers
with tensor copies on two alternating streams. On ROCm, HIP runs concurrent copies from those
buffers one at a time, so the all-gather moves at roughly one link's rate. The copies also compete
with the matmuls for compute units.

With the patch, every rank pushes its data into each peer's symmetric-memory inbox with ROCr's
`hsa_amd_memory_async_copy_on_engine`:
- one SDMA engine per peer (the engine ROCr prefers for writing to that peer);
- all peers at once;
- in pieces, with each piece's matmul starting as soon as that piece has landed.

The copies use no compute units. Each rank's rows land in natural order, so the results, including
`return_A` and the output strides, are the same as the default path's.

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
| `TORCH_SYMM_MEM_ROCM_SDMA_CHUNKS` | `4` | pieces per rank (at most 16); fewer if they do not divide the rows |
| `TORCH_SYMM_MEM_ROCM_SDMA_TIMEOUT_S` | `30` | how long a flag wait waits before it is reported |
| `TORCH_SYMM_MEM_ROCM_SDMA_ON_TIMEOUT` | `abort` | `abort`: hold the stream and abort the process, logging what was missing; `raise`: let the stream move on and raise on the next call |
| `TORCH_SYMM_MEM_ROCM_SDMA_ALLOW_UNTESTED` | `0` | `1` also runs on GPU architectures other than gfx950 |

The SDMA path is used for any dtype when the gather or scatter dimension is not the last one and
the group has 2 to 64 ranks. Everything else runs the default path, with no error:
- the last-dimension variants;
- the fp8 scaled ops;
- other architectures (unless `ALLOW_UNTESTED` is set);
- non-ROCm builds.

## CUDA graphs

Captured calls are supported. A captured call issues its copies from a host-function node, so
every replay issues them when the stream reaches the call. As with the other symm_mem ops, the
symmetric-memory workspace cannot grow during capture: run each op once, eagerly, with the shapes
it will be captured with.

## Tests

`test/distributed/test_symmetric_memory.py` gains `AsyncTPRocmSdmaTest`:
- the existing AsyncTP cases with the path on, each checking that the path ran;
- bf16 with a batch, several pieces and two Bs;
- graph capture with replays on new inputs;
- two process groups;
- a lost flag that is reported instead of hanging.

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
# Wan2.2 T2V TP8, the same command with the path off and on
PROFILE=0 STEPS=40 bash benchmarks/wan_tp/run_profiles.sh tp8_xrepl_fused
TORCH_SYMM_MEM_ENABLE_ROCM_SDMA=1 PROFILE=0 STEPS=40 bash benchmarks/wan_tp/run_profiles.sh tp8_xrepl_fused
```

## Toward an upstream PR

This patch is a prototype for discussion. An upstream version would need:
- the transport in C++ under `torch/csrc/distributed/c10d/symm_mem/`, as a ROCm-only source,
  instead of a JIT-built library called through ctypes;
- ROCr version guards;
- the fp8 scaled variants;
- reporting through the distributed watchdog instead of its own thread;
- a decision on when the path is on by default.
