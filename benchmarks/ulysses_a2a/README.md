# Ulysses all-to-all on SDMA copy engines

Ulysses attention exchanges Q/K/V before attention and the output after it. With
`XDIT_ULYSSES_A2A=sdma` both exchanges run on the GPU's SDMA copy engines instead of
RCCL (`xfuser/core/distributed/sdma`, `xfuser/model_executor/layers/usp_sdma.py`):

- each rank pushes to its peers on one engine per peer, all peers at once, and uses no
  compute units;
- the data lands in the layout the attention kernel reads (bshd over the whole
  sequence), so the receiver unpacks nothing before attention;
- each rank's heads are split into `XDIT_ULYSSES_HEAD_GROUPS` groups (default 2): a
  group attends as soon as its Q/K/V have landed while later groups are still moving,
  and its output goes back while later groups attend. `XDIT_ULYSSES_GROUP_STREAMS=1`
  runs each group's attention on a stream of its own, so a group whose data has landed
  fills the GPU while the previous one drains.

Peer buffers are mapped with HIP IPC handles, so no torch symmetric memory or recent ROCr
is needed. ROCm only, one node, gfx950 validated (`XDIT_SDMA_ALLOW_UNTESTED=1` for other
architectures). Calls the path does not cover (ring attention, joint tensors, FP8
communication, extra per-head inputs, grouped-query repetition, PipeFusion KV cache, head
balancing, sparse backends, CUDA graph capture) stay on RCCL; runners compile without CUDA
graphs when the SDMA path is selected.

Other settings: `XDIT_SDMA_LIB` (a prebuilt transport library, see
`python xfuser/core/distributed/sdma/build.py --help`), `XDIT_SDMA_WAIT_S` and
`XDIT_SDMA_ON_TIMEOUT` (how long a flag wait may take and what happens when it does not
end), `XDIT_SDMA_FLAGS` (where flag writes go).

## Scripts

All run on 8 GPUs of one node.

| Script | What it measures |
|---|---|
| `sdma_ipc_probe.py` | Raw all-to-all on the transport vs RCCL `all_to_all_single`, per buffer size; `--groups 2x4` for two concurrent groups of 4 (CFG x Ulysses 4). Checks every byte. |
| `block_bench.py` | A few real transformer blocks (Wan2.2-T2V, MiniMax-H3, FLUX.3 klein base) at their benchmark shapes, per transport: `rccl`, `nocomm` (exchange replaced by a local copy), `sdma` and `sdmac` (group streams) per head-group count. Reports the share of time spent waiting for the exchange and checks outputs against RCCL. |
| `sdma_stress.py` | New data every call, every call compared exactly with RCCL, over shapes and head-group counts. |

```bash
torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/sdma_ipc_probe.py --mib 16 64 256
torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/sdma_ipc_probe.py --groups 2x4
torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/block_bench.py --model wan --backend AITER_FP8 \
    --variants rccl nocomm sdma sdmac --head-groups 1 2 --compile
torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/block_bench.py --model flux3 --resolution 4096 \
    --backend AITER_FP8 --variants rccl nocomm sdma sdmac --head-groups 1 2 --compile
torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/sdma_stress.py --cfg 2
```

`block_bench.py --model flux3` needs a diffusers with FLUX.3. End to end, set the variables
on an `xdit` run, for example:

```bash
XDIT_ULYSSES_A2A=sdma XDIT_ULYSSES_HEAD_GROUPS=2 XDIT_ULYSSES_GROUP_STREAMS=1 \
    xdit --model Wan2.2-T2V --height 720 --width 1280 --num_frames 81 --ulysses_degree 8 \
    --use_torch_compile --attention_backend AITER_FP8 --prompt "..."
```
