#!/usr/bin/env bash
# Apply the torch/ part of rocm_sdma_async_tp.patch to the torch installed in the active Python
# environment (no rebuild: the patch is Python plus a HIP source built on first use), or revert
# it. The test/ part is for a PyTorch checkout and is skipped.
#
#   bash patches/pytorch/apply_to_installed_torch.sh            # apply
#   bash patches/pytorch/apply_to_installed_torch.sh --revert   # restore the installed torch
#   bash patches/pytorch/apply_to_installed_torch.sh --check    # report whether it is applied
#
# Environment: PYTHON (default python), PATCH (default: the patch next to this script).
set -euo pipefail

PYTHON=${PYTHON:-python}
PATCH_FILE=${PATCH:-$(dirname "$0")/rocm_sdma_async_tp.patch}
mode=${1:-apply}
case $mode in apply | --revert | --check) ;; *) echo "usage: $0 [--revert|--check]" >&2; exit 2 ;; esac

site=$("$PYTHON" -c "import os, torch; print(os.path.dirname(os.path.dirname(torch.__file__)))")
# Only the files under torch/: a diff section starts at each "diff --git" line.
torch_part=$(mktemp)
trap 'rm -f "$torch_part"' EXIT
"$PYTHON" - "$PATCH_FILE" > "$torch_part" <<'EOF'
import re, sys
keep = False
for line in open(sys.argv[1]):
    if line.startswith("diff --git "):
        keep = re.match(r"diff --git a/torch/", line) is not None
    if keep:
        sys.stdout.write(line)
EOF

applied() { patch -p1 -d "$site" --dry-run --reverse --force < "$torch_part" > /dev/null 2>&1; }

case $mode in
  --check)
    if applied; then echo "applied to $site"; else echo "not applied to $site"; fi
    ;;
  --revert)
    if ! applied; then echo "not applied to $site; nothing to revert"; exit 0; fi
    patch -p1 -d "$site" --reverse --force < "$torch_part"
    echo "reverted in $site"
    ;;
  apply)
    if applied; then echo "already applied to $site"; exit 0; fi
    if ! patch -p1 -d "$site" --dry-run --forward < "$torch_part"; then
      echo "the patch does not apply to the torch in $site (see above); nothing changed" >&2
      exit 1
    fi
    patch -p1 -d "$site" --forward < "$torch_part"
    echo "applied to $site; enable with TORCH_SYMM_MEM_ENABLE_ROCM_SDMA=1"
    ;;
esac
