# Ulysses all-to-all on SDMA copy engines

Ulysses attention exchanges Q/K/V before attention and the output after it. With
`--ulysses_a2a sdma` (or `XDIT_ULYSSES_A2A=sdma`) both exchanges run on the GPU's SDMA
copy engines instead of RCCL (`xfuser/core/distributed/sdma`,
`xfuser/model_executor/layers/usp_sdma.py`):

- each rank pushes to its peers on one engine per peer, all peers at once, and uses no
  compute units;
- the data lands in the layout the attention kernel reads (bshd over the whole
  sequence, joint text rows included), so the receiver unpacks nothing before attention;
- each rank's heads are split into `--ulysses_head_groups` groups (default 2): a group
  attends as soon as its Q/K/V have landed while later groups are still moving, and its
  output goes back while later groups attend. `--ulysses_group_streams` runs each group's
  attention on a stream of its own, so a group whose data has landed fills the GPU while
  the previous one drains;
- the sender packs each group's pieces in traced code, so under `torch.compile` Inductor
  fuses the pack into the kernels that produce Q/K/V;
- `--ulysses_overlap_qkv` projects Q/K/V one head group at a time (MiniMax-H3 and FLUX.3,
  whose QK norm and RoPE work per head), so each group's exchange overlaps the next
  group's projection. It keeps a second, row-reordered copy of each layer's Q/K/V weights.

The exchange takes any number of Q-shaped tensors (Q, extra per-head inputs) and K/V,
joint text in front of or behind the sequence, and grouped-query K/V (head groups keep
whole K/V heads). Peer buffers are mapped with HIP IPC handles, so no torch symmetric
memory or recent ROCr is needed. ROCm only, one node, gfx950 validated
(`XDIT_SDMA_ALLOW_UNTESTED=1` for other architectures).

Calls the path does not cover stay on RCCL with a warning: ring attention, FP8
communication, PipeFusion KV cache, head balancing, sparse backends, attention masks, and a
trailing sequence-parallel pad behind rear joint tensors. During CUDA graph capture the
ops run the same layout over RCCL; runners compile without CUDA graphs when the SDMA path
is selected. `--ulysses_a2a rccl_packed` runs the SDMA path's layout over RCCL everywhere,
to tell the transport's share of a difference from the layout's.

The gain is the part of the exchange RCCL leaves exposed. Where RCCL already hides nearly all
of it behind compute, as for Wan2.2, the SDMA path does not help: under `torch.compile`
Inductor moves independent work (such as cross-attention projections) next to RCCL's output
all-to-all, while the SDMA output wait sits inside a custom op it cannot reorder around.

Other settings: `XDIT_SDMA_LIB` (a prebuilt transport library, see
`python xfuser/core/distributed/sdma/build.py --help`), `XDIT_SDMA_WAIT_S` and
`XDIT_SDMA_ON_TIMEOUT` (how long a flag wait may take and what happens when it does not
end). The environment variables `XDIT_ULYSSES_HEAD_GROUPS`, `XDIT_ULYSSES_GROUP_STREAMS`
and `XDIT_ULYSSES_OVERLAP_QKV` give the defaults of the command-line options.

## Scripts

All run on 8 GPUs of one node.

| Script | What it measures |
|---|---|
| `sdma_ipc_probe.py` | Raw all-to-all on the transport vs RCCL `all_to_all_single`, per buffer size; `--groups 2x4` for two concurrent groups of 4 (CFG x Ulysses 4). Checks every byte. |
| `block_bench.py` | A few real transformer blocks (Wan2.2-T2V, MiniMax-H3, FLUX.3 klein base) at their benchmark shapes, per transport: `rccl`, `nocomm` (exchange replaced by a local copy), `sdma` and `sdmac` (group streams) per head-group count, `packed` (the layout over RCCL) and `sdmaq`/`sdmaqc` (grouped Q/K/V projection, without and with group streams). Reports the share of time spent waiting for the exchange and checks outputs against RCCL. |
| `sdma_stress.py` | New data every call, every call checked exactly: the exchange alone over plain, joint and grouped-query shapes, or USP against RCCL. |

```bash
torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/sdma_ipc_probe.py --mib 16 64 256
torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/sdma_ipc_probe.py --groups 2x4
torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/block_bench.py --model wan --backend AITER_FP8 \
    --variants rccl nocomm sdma sdmac packed --head-groups 1 2 --compile
torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/block_bench.py --model flux3 --resolution 4096 \
    --backend AITER_FP8 --variants rccl nocomm sdma sdmac sdmaq --head-groups 1 2 --qkv-groups 2 --compile
torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/sdma_stress.py --cfg 2 --switch-groups --head-groups 1 2 3
```

`block_bench.py --model flux3` needs a diffusers with FLUX.3. End to end, add the options
to an `xdit` run, for example:

```bash
xdit --model Wan2.2-T2V --height 720 --width 1280 --num_frames 81 --ulysses_degree 8 \
    --use_torch_compile --attention_backend AITER_FP8 --ulysses_a2a sdma --ulysses_head_groups 2 \
    --ulysses_group_streams --prompt "..."
```
