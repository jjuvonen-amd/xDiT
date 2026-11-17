"""Which Sol rows and tiles does this aiter serve, through xDiT's own checks."""
import aiter.ops.mha_v4 as m
from aiter.ops.triton.attention import utils as u
print("mha_v4_sol", hasattr(m, "mha_v4_sol"), "packed", hasattr(m, "mha_v4_packed"),
      "ragged_kv", hasattr(m, "mha_v4_ragged_kv"), "sol_prepare", hasattr(u, "sol_prepare"))
try:
    import aiter.ops.triton.attention.sol_tilt as t
    print("sol_tilt:", sorted(n for n in dir(t) if n.startswith("sol_")))
except Exception as e:
    print("sol_tilt missing:", e)
from xfuser.core.attention.sparsity import sol
from xfuser.core.attention.backends.aiter_sol.spec import RECIPES
print("probe:", sol._AITER is not None)
for row in RECIPES:
    for tile in (None, (64, 64)):
        try:
            sol.check_sol_attn_recipe(row.recipe, block_tile=tile)
            r = "ok " + str(sol.sol_attn_block_tile(sol._resolve_recipe(row.recipe), tile))
        except Exception as e:
            r = f"NO {type(e).__name__}: {e}"
        print(f"{row.name:8s} {str(tile):9s} {r}")
