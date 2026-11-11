# MiniMax-H3-Image benchmark and parity tools

Scripts behind the numbers in the MiniMax-H3-Image PR. They run inside the xDiT ROCm container with this
repository at `/app/xDiT` and write to `/output`.

| Script | Purpose |
|---|---|
| `ref_run.py` | Stock diffusers run that dumps every transformer call (kwargs + output), the decoder input and the image |
| `xdit_run.py` | `xfuser.runner` with optional `DUMP_DIR` dumps, `TIMING` (CUDA-event breakdown: text encoder, each step, VAE), `TEACHER_DIR` (teacher forcing against `ref_run.py` dumps), `NO_FUSE` / `NO_FLYDSL` exactness switches, `COMPILE_MODE` / `AITER_RMSNORM` A/B switches |
| `parity_sp.py` | Teacher-forced Ulysses parity of the transformer wrapper on dumped inputs (`torchrun --nproc-per-node N`) |
| `bench.sh` | One benchmark config: `bench.sh <name> <nproc> <size> [runner args]` |
| `bench_matrix.sh`, `bench_phase2.sh`, `bench_final.sh`, `bench_hybrid.sh` | The benchmark sets, in the order they were run |
| `attn_probe.py`, `mha_v4_short.py` | Attention backend accuracy probes (found the AITER_BF16 short-key bug) |
| `te_bench.py` | Prompt-encoding time and profile on one GPU |
| `trace_summary.py` | GPU kernel time of a profiler trace by category |
| `psnr.py`, `run.sh` | Image PSNR; one plain runner invocation |
