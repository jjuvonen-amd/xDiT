# Sol-Attn on AITER mainline: Wan2.2-T2V and MiniMax-H3 scripts

Scripts behind the Sol-Attn PRs (backend-owned Sol setup, MiniMax-H3 Sol support, the pooled
value tilt). They are not part of those PRs. Results are reported in the PR descriptions.

Environment: `amdsiloai/pytorch-xdit:v26.10.1` on gfx950, with ROCm/aiter main built in place
(`GPU_ARCHS=gfx950 AITER_REBUILD=1 python3 setup.py develop`). The scripts assume a container
named `jjuvonen-sol-mainline` with this repository checked out under `/work` and model weights in
the Hugging Face cache; change `CONTAINER`, `SRC` and the paths for another setup. GPU jobs go
through the node's `brr` scheduler; replace `/opt/brr/brr run ... --` with a plain launch elsewhere.

| file | what it does |
|---|---|
| `probe_sol.py` | Lists which Sol recipes and tiles the installed AITER serves, through xDiT's own checks. |
| `run.sh` | End-to-end rows: Wan2.2-T2V (1280x720x81, 40 steps, Ulysses 8) and MiniMax-H3 (t2va, defaults, Ulysses 8), dense and Sol. `SEEDS`, `TAG`, `EXTRA`, `SRC`, `BRR_ARGS` select seeds, output suffix, extra flags, checkout and scheduling (`--node` for timing). |
| `compare.py` | Mean frame PSNR of each video against the first, byte-level identity, and a frame montage. |
| `score_video.py` | Per-latent-frame speckle and Laplacian scores of Wan videos; edge/mid ratios flag first- and last-frame artifacts. |
| `sol_dump.patch` | Experiment hook for the Sol launcher: `XFUSER_SOL_DUMP=<dir> XFUSER_SOL_DUMP_CALLS=75,510,...` saves rank 0's operands of those Sol calls in an eager run. |
| `repro_oob.py` | Places one pooled scale so its storage ends at an unmapped page (run with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`) and launches the Sol row: a read past the scale faults the GPU. Found the MXFP8 64x64 pooled K scale over-read. |
| `h3_exact_tokens.py` | On dumped MiniMax-H3 operands: relative error of Sol against fp32 dense attention for text/audio rows and video rows, with and without the text/audio blocks forced exact. |

Typical sequence:

```bash
SRC=/work/xDiT benchmarks/sol_mainline/run.sh wan_dense wan_fp8sol wan_fp8sol_tilt2
SEEDS="1 2 3" benchmarks/sol_mainline/run.sh wan_fp8sol wan_fp8sol_tilt2
EXTRA="--warmup_calls 1" BRR_ARGS="--node" TAG=_timed benchmarks/sol_mainline/run.sh h3_dense h3_fp8sol
python3 benchmarks/sol_mainline/compare.py --out montage.jpg --frames 0,40,-1 dense.mp4 sol.mp4
```
