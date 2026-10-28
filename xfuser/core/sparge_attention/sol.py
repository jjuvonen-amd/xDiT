"""Sol-Attn: block-sparse attention with a pooled-KV correction for the blocks it skips.

Plain block-sparse attention routes each query tile to a subset of KV blocks and DROPS the rest.
Sol-Attn (arXiv 2607.24027) computes the same selected blocks exactly and additionally recovers the
contribution of the skipped blocks from pooled (mean) K/V, so the mass outside the selection is
approximated instead of discarded.

This targets aiter's mha_v4 mode-2 rows as the mha_v4-sol-attn branch ships them: mha_v4_sol for
the raw path, sol_prepare + mha_v4_packed for everything that needs a say in routing (forced
blocks, head cost, the error router, the Jensen correction). Older builds with mha_v4_sol_attn are
refused at setup rather than half supported.
"""
import functools
import os
from types import SimpleNamespace
from typing import NamedTuple

import torch

# Which recipes a device serves is the manifest's to say -- gfx950 has a mode-2 row for every
# recipe below, gfx942 currently has none -- so this only names the archs worth asking.
_SUPPORTED_ARCHS = ("gfx942", "gfx950")

# The routers sol_prepare knows. "mean" is the paper's q_mean . k_mean; "error" scores a block by
# its second-order mass times the spread of the weights inside it, so the exact budget goes where
# the pooled estimate is least reliable.
SOL_ATTN_ROUTERS = ("mean", "error")


class SolAttnUnsupported(RuntimeError):
    """Raised when the inputs or the environment cannot be served correctly by the Sol-Attn kernel."""


def _keep_bf16(tensor):
    """Pass a BF16 operand through unquantized, in the (tensor, descale) shape the others return.

    The BF16 rows carry a NONE scale mode, so the kernel never reads the descale. aiter's own
    mha_v4 hands the tensor itself back as the placeholder rather than a unit scalar, and this
    matches that so the packed path here cannot disagree with the raw one about what it sent.
    """
    return tensor, tensor


def _probe_aiter():
    """aiter's Sol-Attn entry points, or None when this build does not ship them.

    Resolved at import. This module is imported lazily, only once the backend is selected, so builds
    that never touch Sol-Attn pay nothing; binding the entry points once also keeps repeated imports
    out of any traced graph. native_fp8_format is held as a function rather than called here because
    it queries the device, which import time is too early for.

    _k_mean and _resolve_raw_recipe are private to aiter. They are what mha_v4_sol uses to decide
    K smoothing and a recipe's scale modes and V packing, and the packed path here has to make the
    same decisions or forcing a block would silently change the output; importing them keeps the
    two from drifting until aiter exports them.
    """
    try:
        from aiter.ops.mha_v4 import (
            AttentionFormat,
            AttentionScaleMode,
            MHA_V4_SOL_MODE,
            _k_mean,
            _resolve_raw_recipe,
            mha_v4_block_tile,
            mha_v4_block_tiles,
            mha_v4_block_tiles_in_any_precision,
            mha_v4_kv_tile_for_q_tile,
            mha_v4_operands,
            mha_v4_packed,
            mha_v4_q_multiplier,
            mha_v4_ragged_kv,
            mha_v4_sol,
            native_fp8_format,
        )
        from aiter.ops.mha_v4_quant import (
            mxfp4_k_view,
            mxfp4_v_view,
            mxfp6_k_view,
            quantize_fp8,
            quantize_fp8_rotated,
            quantize_int8,
            quantize_mxfp4_k,
            quantize_mxfp4_q,
            quantize_mxfp6_k,
            quantize_mxfp6_q,
            quantize_mxfp8_k,
            quantize_mxfp8_q,
            quantize_v_fp8,
            quantize_v_mxfp4_fp6_p,
            quantize_v_mxfp6_fp6_p,
        )
        from aiter.ops.triton.attention.utils import sol_prepare
    except ImportError:
        return None
    return SimpleNamespace(
        sol=mha_v4_sol,
        packed=mha_v4_packed,
        prepare=sol_prepare,
        native_fp8_format=native_fp8_format,
        block_tile=mha_v4_block_tile,
        kv_tile_for_q_tile=mha_v4_kv_tile_for_q_tile,
        block_tiles=mha_v4_block_tiles,
        block_tiles_in_any_precision=mha_v4_block_tiles_in_any_precision,
        operands=mha_v4_operands,
        ragged_kv=mha_v4_ragged_kv,
        sol_mode=MHA_V4_SOL_MODE,
        q_multiplier=mha_v4_q_multiplier,
        k_mean=_k_mean,
        resolve_raw_recipe=_resolve_raw_recipe,
        quantize_fp8=quantize_fp8,
        quantize_fp8_rotated=quantize_fp8_rotated,
        quantize_int8=quantize_int8,
        quantize_mxfp8_q=quantize_mxfp8_q,
        quantize_mxfp8_k=quantize_mxfp8_k,
        quantize_mxfp4_q=quantize_mxfp4_q,
        quantize_mxfp4_k=quantize_mxfp4_k,
        quantize_mxfp6_q=quantize_mxfp6_q,
        quantize_mxfp6_k=quantize_mxfp6_k,
        quantize_v_fp8=quantize_v_fp8,
        quantize_v_mxfp4_fp6_p=quantize_v_mxfp4_fp6_p,
        quantize_v_mxfp6_fp6_p=quantize_v_mxfp6_fp6_p,
        mxfp4_k_view=mxfp4_k_view,
        mxfp4_v_view=mxfp4_v_view,
        mxfp6_k_view=mxfp6_k_view,
        fmt=AttentionFormat,
        per_tensor_scale=AttentionScaleMode.F32_PER_TENSOR,
        block_scale=AttentionScaleMode.E8M0_PER_1X32,
    )


_AITER = _probe_aiter()
SOL_ATTN_AVAILABLE = _AITER is not None

# Held as a module global rather than an _AITER field, which is not a style choice. aiter decorates
# mha_v4_kv_tile with functools.cache, so the name is an _lru_cache_wrapper -- a C object that
# implements the descriptor protocol. Reached through an attribute Dynamo binds the owner as self
# and the call dies with "Too many positional arguments: got 1, expected 0", which under
# fullgraph=True is a hard error and otherwise a graph break in the middle of every attention layer.
# Eager does not bind (SimpleNamespace holds it in an instance dict) so this shows up only compiled.
_default_block_tile = _AITER.block_tile if _AITER is not None else None
_kv_tile_for_q_tile = _AITER.kv_tile_for_q_tile if _AITER is not None else None
_block_tiles_in_any_precision = (
    _AITER.block_tiles_in_any_precision if _AITER is not None else None
)
# mha_v4_ragged_kv is functools.cache'd too, so the same applies.
_mha_v4_ragged_kv = _AITER.ragged_kv if _AITER is not None else None


def _probe_second_order():
    """aiter's Jensen cap and pooled value tilt builders (mha_v4-sol-attn 0c56c8d+), or Nones.

    Held as module globals like the tile queries above, so a compiled call reaches them without
    Dynamo binding an owner. Older builds lack them; asking for the cap or the tilt then fails at
    setup with the reason.
    """
    try:
        from aiter.ops.triton.attention.sol_tilt import (
            sol_jensen_cap,
            sol_q_abs_max,
            sol_tilt_factors,
            sol_tilt_planes,
            sol_tilt_scaled_params,
            sol_tilt_valu_params,
        )
        from aiter.ops.triton.attention.utils import _e8m0_dequantize
    except ImportError:
        return (None,) * 7
    return (sol_q_abs_max, sol_jensen_cap, sol_tilt_factors, sol_tilt_planes,
            sol_tilt_valu_params, sol_tilt_scaled_params, _e8m0_dequantize)


(_q_abs_max, _jensen_cap, _tilt_factors, _tilt_planes, _tilt_valu_params, _tilt_scaled_params,
 _e8m0_dequantize) = _probe_second_order() if _AITER is not None else (None,) * 7


def _read_block_tile_override():
    """Parse XFUSER_SOL_ATTN_BLOCK_TILE into (q_tile, kv_tile), or None. See xfuser/envs.py.

    Read at import, which is the only place it can be: the value has to be a plain Python tuple by
    the time a traced call reaches it, and os.environ is not traceable. Whether the tile actually
    has a kernel is NOT decided here -- that needs the device -- so this only parses, and
    check_sol_attn_supported validates.
    """
    from xfuser.envs import environment_variables

    spec = environment_variables["SOL_ATTN_BLOCK_TILE"]()
    if not spec:
        return None
    try:
        q_tile, kv_tile = (int(part) for part in spec.lower().split("x"))
    except ValueError:
        raise ValueError(
            f"XFUSER_SOL_ATTN_BLOCK_TILE must be QxKV, e.g. 64x64; got {spec!r}"
        ) from None
    return q_tile, kv_tile


_BLOCK_TILE_OVERRIDE = _read_block_tile_override()


def sol_attn_block_tile(recipe):
    """The (q_tile, kv_tile) `recipe` routes, pools and dispatches Sol-Attn at.

    One source for all three. They are not independently choosable: the LUT and the selection
    bitmap are in units of kv_tile, the pooled K/V are one row per kv_tile tokens, and the kernel
    folds log2(kv_tile) into its softmax bias as a build-time constant. A mismatch between any two
    is silently wrong rather than an error, which is why every site below reads it from here.

    The recipe is required rather than defaulted because there is no longer an arch-wide answer to
    default to: gfx950's rows disagree at a 256-row query tile, where BF16 and BF16/FP8 route on 64
    keys and the rest on 128. aiter refuses an operand-blind query rather than pick one, since the
    caller is about to shape a mask to the answer and a mask cut for the wrong block addresses the
    wrong keys.

    Deliberately not cached: the override is already a constant and the aiter fallback is cached and
    torch_compile_guard'd on its side, so this body is a tuple return that Dynamo folds away.
    """
    if _BLOCK_TILE_OVERRIDE is not None:
        return _BLOCK_TILE_OVERRIDE
    return _default_block_tile(_recipe_operands(recipe), _AITER.sol_mode)


class _Recipe(NamedTuple):
    """One mode-2 manifest row, named by its formats.

    Scale modes, V packing and the K-smoothing kind are not repeated here: they are aiter's
    decision for a format triple (_resolve_raw_recipe), and restating them is how this file and
    mha_v4_sol would come to quantize the same recipe two ways. The one thing the formats cannot
    say is MXFP8, which is FP8's formats with E8M0 Q/K scales, so it names its scale modes.

    k_packed / v_packed name an operand whose stored codes are sub-byte and permuted into the ASM's
    tile order. Such an operand cannot be pooled or routed from its codes, so pooling works from the
    BF16 source, routing scores the BF16 Q, and the operand's own scale is not handed to
    sol_prepare.
    """

    id: str
    qk_format: str            # attribute name on AttentionFormat; "native" is this GPU's FP8
    v_format: str
    block_scaled_fp8: bool = False
    k_packed: str | None = None
    v_packed: str | None = None

    @property
    def routes_through_raw(self) -> bool:
        """Whether mha_v4_sol can serve this row, or it has to go packed.

        mha_v4_sol takes formats only and resolves them to canonical scale modes, so MXFP8 -- the
        same formats as FP8 -- is out of its reach and always runs through mha_v4_packed.
        """
        return not self.block_scaled_fp8

    @property
    def supports_error_router(self) -> bool:
        """The error router needs K's per-block variance from addressable codes, so not a packed K."""
        return self.k_packed is None


_RECIPES = {
    r.id: r
    for r in (
        _Recipe("bf16", "BF16", "BF16"),
        # Q/K stay BF16 and only V drops to FP8, which halves the traffic on the operand the
        # correction pass reads once per KV block while leaving the scores exact.
        _Recipe("bf16fp8", "BF16", "native"),
        _Recipe("fp8", "native", "native"),
        _Recipe("i8fp8", "INT8", "native"),
        _Recipe("mxfp8", "native", "native", block_scaled_fp8=True),
        _Recipe("mxfp4", "MXFP4", "MXFP4", k_packed="mxfp4", v_packed="mxfp4_fp6_p"),
        _Recipe("mxfp6", "MXFP6", "MXFP6", k_packed="mxfp6", v_packed="mxfp6_fp6_p"),
        # FP8 Q/K are addressable, so f8f6 routes and pools on its codes like fp8 does.
        _Recipe("f8f6", "native", "MXFP6", v_packed="mxfp6_fp6_p"),
        # f6f8's FP8 V carries a per-channel descale, which survives pooling over the sequence.
        _Recipe("f6f8", "MXFP6", "native", k_packed="mxfp6"),
        _Recipe("f6f4", "MXFP6", "MXFP4", k_packed="mxfp6", v_packed="mxfp4_fp6_p"),
    )
}

SOL_ATTN_RECIPES = tuple(_RECIPES)


def check_sol_attn_device(device=None):
    """Raise unless this build and device can run Sol-Attn. Defaults to the current CUDA device.

    Both facts are fixed for the process, so runtime_state calls this once during backend setup and
    the per-call path does not repeat it. Callers reaching sol_attn_bhsd directly, outside xDiT's
    setup, should call it themselves.
    """
    if not SOL_ATTN_AVAILABLE:
        raise SolAttnUnsupported(
            "Sol-Attn requires aiter with mha_v4_sol and sol_prepare (the mha_v4-sol-attn "
            "branch); please update AITER")
    if device is None:
        if not torch.cuda.is_available():
            raise SolAttnUnsupported("Sol-Attn is a GPU kernel and no CUDA device is available")
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        raise SolAttnUnsupported(f"Sol-Attn is a GPU kernel, got device {device}")
    arch = _device_arch(device)
    if arch is None:
        reported = torch.cuda.get_device_properties(device).gcnArchName or ""
        raise SolAttnUnsupported(
            f"Sol-Attn ships kernels for {', '.join(_SUPPORTED_ARCHS)}, "
            f"this device reports '{reported}'")


def _device_arch(device=None):
    """The supported arch this device is, or None. gcnArchName carries a target-feature suffix."""
    if device is None:
        if not torch.cuda.is_available():
            return None
        device = torch.device("cuda", torch.cuda.current_device())
    name = torch.cuda.get_device_properties(device).gcnArchName or ""
    return next((arch for arch in _SUPPORTED_ARCHS if name.startswith(arch)), None)


def check_sol_attn_recipe(recipe_id, device=None, router="mean"):
    """Raise unless this device has a Sol-Attn manifest row for recipe_id, and can route it as asked.

    Separate from check_sol_attn_device because the answer is per recipe, and it is the manifest's
    to give: a build may ship a mode-2 row for one precision and not another, and gfx942 currently
    has none at all. Selecting a recipe without a row has to fail at setup with the reason rather
    than at the first launch with a missing-kernel error.
    """
    check_sol_attn_device(device)
    recipe = _resolve_recipe(recipe_id)
    if not _AITER.block_tiles(_recipe_operands(recipe), _AITER.sol_mode):
        raise SolAttnUnsupported(
            f"Sol-Attn '{recipe_id}' has no mode-2 manifest row on {_device_arch(device)} in this "
            f"aiter build")
    _check_block_tile_override(recipe)
    _check_router(recipe, router)


def _check_router(recipe, router):
    if router not in SOL_ATTN_ROUTERS:
        raise SolAttnUnsupported(
            f"unknown Sol-Attn router {router!r}; choose from {', '.join(SOL_ATTN_ROUTERS)}")
    if router == "error" and not recipe.supports_error_router:
        raise SolAttnUnsupported(
            f"the 'error' router scores each block by K's per-block variance, which a packed "
            f"{recipe.k_packed} K does not expose; Sol-Attn '{recipe.id}' supports the 'mean' "
            f"router only")


# The rows aiter builds with the pooled value tilt (0c56c8d), at the block tiles each is built for.
# f6f8 has no tilt object.
_TILT_BLOCK_TILES = {
    "fp8": {(256, 128), (64, 64)}, "i8fp8": {(256, 128), (64, 64)}, "mxfp8": {(256, 128), (64, 64)},
    "bf16fp8": {(256, 64), (64, 64)}, "bf16": {(256, 64), (64, 64)}, "f8f6": {(256, 128), (64, 64)},
    "mxfp6": {(256, 128), (64, 64)}, "f6f4": {(256, 128), (64, 64)}, "mxfp4": {(256, 128), (64, 64)}}


def check_sol_second_order(recipe, jensen=False, jensen_cap=True, tilt_rank=0):
    """Raise unless this aiter can serve the Jensen cap and the tilt as asked, on `recipe`'s tile.

    The tilt adds each pooled block's leading Cov(v, k) q terms to its mean value, at the block's
    capped Jensen weight, so it implies the capped term. Called at setup by runtime_state and per
    call by sol_attn_bhsd.
    """
    if tilt_rank < 0:
        raise SolAttnUnsupported(f"--solattn_tilt_rank must be >= 0, got {tilt_rank}")
    if tilt_rank and not jensen_cap:
        raise SolAttnUnsupported(
            "the pooled value tilt is scaled by the Jensen cap's per-block gain, so it cannot run "
            "with --solattn_jensen_uncapped")
    if (jensen or tilt_rank) and jensen_cap and _jensen_cap is None:
        raise SolAttnUnsupported(
            "the capped Jensen term needs aiter's sol_jensen_cap (mha_v4-sol-attn 0c56c8d or "
            "later); this aiter does not have it")
    if not tilt_rank:
        return
    if _tilt_factors is None:
        raise SolAttnUnsupported(
            "--solattn_tilt_rank needs aiter's sol_tilt builders and tilt-enabled Sol kernels "
            "(mha_v4-sol-attn 0c56c8d or later); this aiter has neither")
    tile = sol_attn_block_tile(recipe)
    if tile not in _TILT_BLOCK_TILES.get(recipe.id, ()):
        served = ", ".join(
            f"'{r}' at {' or '.join('x'.join(map(str, t)) for t in sorted(ts))}"
            for r, ts in _TILT_BLOCK_TILES.items())
        raise SolAttnUnsupported(
            f"the pooled value tilt is served by {served} only, not '{recipe.id}' at "
            f"{'x'.join(map(str, tile))}")


def _second_order(recipe, quantized, routing, softmax_scale, jensen_cap, tilt_rank):
    """The capped Jensen term and the pooled value tilt, as mha_v4_packed's keyword arguments.

    Returns the kwargs that replace routing's mean_k / mean_v / their scales and mean_k_var /
    mean_k_var_scale, plus sol_tilt. Follows nsakkine/xDiT feature/sol-attn ef1bd08, mapped onto
    _Quantized: his (codes, descale, source) triples are our q/k/v pairs plus routing_q /
    k_source / v_source, the BF16 sources of a packed operand.

    Uncapped (jensen_cap=False) is the old behaviour, kept to measure what the cap does: routing's
    variance goes in as is. Capped, every recipe damps each block's term to at most ln(kv_tile);
    without it, late Wan layers hand pooled blocks the whole softmax.
    """
    out = dict(mean_k=routing["mean_k"], mean_v=routing["mean_v"],
               mean_k_scale=routing["mean_k_scale"], mean_v_scale=routing["mean_v_scale"],
               mean_k_var=routing.get("mean_k_var"),
               mean_k_var_scale=routing.get("mean_k_var_scale"))
    if not jensen_cap:
        return out
    q, k, v = quantized.q, quantized.k, quantized.v
    q_tile, kv_tile = sol_attn_block_tile(recipe)
    packed = recipe.k_packed is not None
    q_cap, qk_descales, block_scaled = q[0], (None, None), {}
    if recipe.qk_format == "BF16":
        q_abs_max = _q_abs_max(q[0], q_tile)
    elif not recipe.block_scaled_fp8 and not packed:
        # Per-tensor FP8 / INT8 Q and K.
        q_abs_max = _q_abs_max(q[0], q_tile)
        qk_descales = (q[1], k[1])
    elif not packed:
        # MXFP8: Q codes carry softmax_scale * log2(e), folded in by the quantizer.
        q_abs_max = _q_abs_max(q[0], q_tile, q_scale=q[1])
        block_scaled = dict(mean_k_var_scale=routing["mean_k_var_scale"],
                            q_multiplier=_AITER.q_multiplier(softmax_scale))
    else:
        # A packed Q cannot be read back, so its BF16 source goes through the packers' rotation.
        q_cap = quantized.routing_q
        q_abs_max = _q_abs_max(q_cap, q_tile, rotate=True)
        block_scaled = dict(mean_k_var_scale=routing["mean_k_var_scale"],
                            mean_k_var_pooled=routing["mean_k_var_pooled"])
    out["mean_k_var"], out["mean_k_var_scale"], gain = _jensen_cap(
        q_cap, *qk_descales, routing["mean_k_var"], softmax_scale, kv_tile,
        block_attn_mask=routing["block_attn_mask"], q_tile=q_tile, q_abs_max=q_abs_max,
        **block_scaled)
    if not tilt_rank:
        return out

    gain = gain.sqrt()
    mean_k, mean_v = out["mean_k"], out["mean_v"]
    mean_k_scale, mean_v_scale = out["mean_k_scale"], out["mean_v_scale"]
    if recipe.id == "fp8":
        out["mean_k"], out["mean_v"], out["sol_tilt"] = _tilt_factors(
            q[0], k[0], v[0], q[1], k[1], v[1], mean_k, mean_v, softmax_scale, tilt_rank,
            kv_tile, block_gain=gain, q_abs_max=q_abs_max)
        return out
    if recipe.id == "mxfp8":
        # A is E8M0-quantized on top of the pooled K. The score MFMA decodes Q and A, so the
        # parameter block's multiplier carries no descale, and the K-scale plane stride rides in
        # its third dword. Plane 0 of the stacked scales replaces the pooled scale.
        mult = _AITER.q_multiplier(softmax_scale)
        planes = _tilt_planes(
            _e8m0_dequantize(k[0], k[1]).to(torch.bfloat16), v[0], q_abs_max, mean_k, mean_v,
            softmax_scale, tilt_rank, kv_tile, k_format="mxfp8", v_format="e4m3",
            q_descale=1.0 / mult, v_descale=v[1], q_multiplier=mult, mean_k_scale=mean_k_scale,
            block_gain=gain)
        params = _tilt_scaled_params(planes, planes.k_scale_plane_stride)
    elif recipe.id in ("bf16", "bf16fp8"):
        # BF16 A planes against the raw BF16 Q. BF16/FP8's B planes are FP8 at half the K planes'
        # width, so the V plane stride rides in the parameter block's third dword; the BF16 row's
        # descale slot holds V itself, not a scale.
        bf16_v = recipe.id == "bf16"
        planes = _tilt_planes(
            k[0], v[0], q_abs_max, mean_k, mean_v, softmax_scale, tilt_rank, kv_tile,
            k_format="bf16", v_format="bf16" if bf16_v else "e4m3",
            v_descale=1.0 if bf16_v else v[1], block_gain=gain)
        params = _tilt_scaled_params(planes, planes.v_plane_stride)
    elif recipe.id == "i8fp8":
        planes = _tilt_planes(
            k[0], v[0], q_abs_max, mean_k, mean_v, softmax_scale, tilt_rank, kv_tile,
            k_format="int8", v_format="e4m3", q_descale=q[1], k_descale=k[1], v_descale=v[1],
            block_gain=gain)
        params = _tilt_valu_params(planes, q[1], k[1])
    elif recipe.id == "f8f6":
        # e4m3 A planes against the rotated FP8 Q; MXFP6 B planes packed with the pooled BF16 V by
        # V's own packer, their data and scale strides riding after the score multiplier.
        planes = _tilt_planes(
            k[0], quantized.v_source, q_abs_max, mean_k, mean_v, softmax_scale, tilt_rank,
            kv_tile, k_format="e4m3", v_format="mxfp6_fp6_p", q_descale=q[1], k_descale=k[1],
            mean_v_scale=mean_v_scale, mean_v_pooled=routing["mean_v_pooled"], block_gain=gain)
        params = _tilt_valu_params(planes, q[1], k[1],
                                   strides=(planes.v_plane_stride, planes.v_scale_plane_stride))
    elif recipe.id in ("mxfp6", "f6f4", "mxfp4"):
        # Packed K: A planes go through K's own packer, which rotates them as it rotates K, so the
        # factors take the unrotated sources and Q maxima. The score MFMA decodes Q and A, so the
        # multiplier carries no descale. MXFP4's A-plane scales sit in an image of their own,
        # whose stride rides after V's two.
        k_format = "mxfp4" if recipe.id == "mxfp4" else "mxfp6"
        v_format = "mxfp6_fp6_p" if recipe.id == "mxfp6" else "mxfp4_fp6_p"
        planes = _tilt_planes(
            quantized.k_source, quantized.v_source, _q_abs_max(quantized.routing_q, q_tile),
            mean_k, mean_v, softmax_scale, tilt_rank, kv_tile, k_format=k_format,
            v_format=v_format, q_multiplier=_AITER.q_multiplier(softmax_scale),
            mean_k_scale=mean_k_scale, mean_v_scale=mean_v_scale,
            mean_k_pooled=routing["mean_k_pooled"], mean_v_pooled=routing["mean_v_pooled"],
            block_gain=gain)
        strides = (planes.v_plane_stride, planes.v_scale_plane_stride)
        if recipe.id == "mxfp4":
            strides += (planes.k_scale_plane_stride,)
        params = _tilt_valu_params(planes, strides=strides)
    else:
        raise SolAttnUnsupported(f"the pooled value tilt has no '{recipe.id}' factor path")
    out["mean_k"], out["mean_v"] = planes.mean_k, planes.mean_v
    if planes.mean_k_scale is not None:
        out["mean_k_scale"] = planes.mean_k_scale
    if planes.mean_v_scale is not None:
        out["mean_v_scale"] = planes.mean_v_scale
    out["sol_tilt"] = (planes.rank, planes.k_plane_stride, params)
    return out


def _check_block_tile_override(recipe=None):
    """Raise unless XFUSER_SOL_ATTN_BLOCK_TILE names a geometry this device has a kernel for.

    Checked against `recipe`'s operands when one is given, because a geometry need not exist in
    every precision. The membership is aiter's to state -- it is read from the manifest per call,
    not pinned here -- so a build that adds rows widens this on its own.

    Without a recipe it asks whether ANY precision serves the tile, which is all a caller reaching
    sol_attn_bhsd() directly has settled by then. That question goes to
    mha_v4_block_tiles_in_any_precision rather than to the per-Q-tile query, which raises where
    the rows disagree.
    """
    if _BLOCK_TILE_OVERRIDE is None:
        return
    q_tile, kv_tile = _BLOCK_TILE_OVERRIDE
    if recipe is None:
        served = _block_tiles_in_any_precision(_AITER.sol_mode)
        if _BLOCK_TILE_OVERRIDE in served:
            return
        hint = "Unset it for this recipe's default geometry."
    else:
        operands = _recipe_operands(recipe)
        if _kv_tile_for_q_tile(q_tile, operands, _AITER.sol_mode) == kv_tile:
            return
        served = _AITER.block_tiles(operands, _AITER.sol_mode)
        default = _default_block_tile(operands, _AITER.sol_mode)
        hint = (f"Unset it for the default geometry "
                f"({'x'.join(str(part) for part in default)}).")
    detail = "" if recipe is None else f" with the '{recipe.id}' recipe"
    raise SolAttnUnsupported(
        f"XFUSER_SOL_ATTN_BLOCK_TILE asks for {q_tile}x{kv_tile}, which this GPU has no Sol-Attn "
        f"kernel for{detail}; it serves "
        f"{', '.join(f'{qo}x{kv}' for qo, kv in served) or 'none'}. {hint}")


def _recipe_plan(recipe):
    """aiter's plan for `recipe`: its K-smoothing kind, scale modes and V packing.

    Pure enum logic on aiter's side, so a traced call folds it to constants.
    """
    qk, v = _format(recipe.qk_format), _format(recipe.v_format)
    modes = ((_AITER.block_scale, _AITER.block_scale, _AITER.per_tensor_scale)
             if recipe.block_scaled_fp8 else (None, None, None))
    return _AITER.resolve_raw_recipe(qk, qk, v, *modes, sparse=True)


def _recipe_operands(recipe):
    """The seven manifest values identifying `recipe`'s row, for aiter's geometry queries.

    Resolved per device, since a recipe's format may be "native".
    """
    plan = _recipe_plan(recipe)
    qk, v = _format(recipe.qk_format), _format(recipe.v_format)
    return _AITER.operands(qk, qk, v, *plan.scale_modes, plan.v_pack)


def _ragged_kv(recipe_id, block_tile):
    """Whether `recipe_id`'s row at `block_tile` takes a key length that is not a whole tile.

    aiter caches the answer and guards the manifest lookup behind it, as it does the tile query,
    so a traced call folds this to a constant.
    """
    recipe = _RECIPES[recipe_id]
    return bool(_mha_v4_ragged_kv(_recipe_operands(recipe), _AITER.sol_mode, block_tile))


def check_sol_attn_supported(query, key, value, is_causal, ring_world_size=1, recipe=None,
                             router="mean"):
    """Validate the per-call constraints aiter's Sol-Attn contract does not already cover.

    `recipe` narrows the block-tile check to the row this call will actually dispatch on, which is
    what sol_attn_bhsd passes. Left None it falls back to asking whether any precision serves the
    tile, for a caller that has not resolved a recipe yet.
    """
    if not query.dtype == key.dtype == value.dtype == torch.bfloat16:
        raise SolAttnUnsupported(
            f"Sol-Attn's mha_v4 row takes bf16 Q/K/V and returns bf16, got q={query.dtype} "
            f"k={key.dtype} v={value.dtype}. Select another attention backend for other dtypes.")
    if is_causal:
        raise SolAttnUnsupported(
            "Sol-Attn has no causal variant: its pooled correction assumes every skipped block is "
            "fully attendable, which a causal mask breaks. Use AITER_FP8 for causal attention.")
    if ring_world_size > 1:
        raise SolAttnUnsupported(
            "Sol-Attn does not support ring parallelism: its routing threshold is taken over the "
            "blocks one call is shown, so a rank holding part of the KV routes against its own "
            "shard. Use ulysses_degree for sequence parallelism instead.")
    # Checked per call rather than at import because it needs the device. A run that selected the
    # backend through runtime_state has already had this checked against its recipe at setup; this
    # is for a caller reaching sol_attn_bhsd() directly, and it names the env var rather than
    # leaving a wrong tile to surface as a missing manifest row several frames below it.
    _check_block_tile_override(recipe)
    if recipe is not None:
        _check_router(recipe, router)


def _resolve_recipe(recipe):
    """The named recipe, or a clear error listing what this build actually ships."""
    try:
        return _RECIPES[recipe]
    except KeyError:
        raise SolAttnUnsupported(
            f"unknown Sol-Attn recipe {recipe!r}; this build has "
            f"{', '.join(SOL_ATTN_RECIPES)}") from None


def _format(name):
    """An AttentionFormat by name, with "native" resolved against the active GPU."""
    return _AITER.native_fp8_format() if name == "native" else getattr(_AITER.fmt, name)


class _Quantized(NamedTuple):
    """One call's operands as the kernel and the routing take them.

    q/k/v are (tensor, descale) pairs. k_source / v_source are the BF16 tensors a packed operand
    was quantized from, which is all pooling can read for it, and None for an addressable one.
    routing_q is what routing scores: the stored Q where its
    codes are addressable, the BF16 Q where they are packed. k_shift is the K-smoothing vector a
    packed K's quantizer subtracted, which sol_prepare needs to pool from the unshifted source; an
    addressable K's codes carry the shift already. q_scale is an MXFP8 Q's E8M0 scale, without
    which routing would score codes that are not proportional to Q. routing_scale turns routing's
    q . k into the softmax logit in nats, which only the error router needs.
    """

    q: tuple
    k: tuple
    v: tuple
    k_source: torch.Tensor | None
    v_source: torch.Tensor | None
    routing_q: torch.Tensor
    k_shift: torch.Tensor | None
    q_scale: torch.Tensor | None
    routing_scale: object


def _quantize(recipe, query, key, value, softmax_scale):
    """Quantize BSHD bf16 Q/K/V for one recipe exactly as mha_v4_sol would.

    Mirrors mha_v4_sol and its MX helper branch for branch, K smoothing included, because the
    packed path here and the raw one there must hand the kernel the same operands: asking for the
    head cost, forcing a block or switching on the error router must change nothing else.
    """
    plan = _recipe_plan(recipe)
    k_mean = _AITER.k_mean(key, plan.kind)
    multiplier = _AITER.q_multiplier(softmax_scale)
    qk, v_format = recipe.qk_format, recipe.v_format
    q_scale = None

    if qk == "BF16":
        q, k = _keep_bf16(query), _keep_bf16(key)
        routing_scale = softmax_scale
    elif qk == "INT8":
        # INT8's quantizer takes no mean, so the subtraction is materialised, as in mha_v4_sol.
        q = _AITER.quantize_int8(query)
        k = _AITER.quantize_int8(
            key if k_mean is None else (key.float() - k_mean.unsqueeze(1)).to(key.dtype))
        routing_scale = softmax_scale * (q[1] * k[1]).reshape(())
    elif recipe.block_scaled_fp8:
        q = _AITER.quantize_mxfp8_q(query, multiplier)
        k = _AITER.quantize_mxfp8_k(key, k_mean)
        q_scale = q[1]
        routing_scale = softmax_scale / multiplier
    elif qk == "MXFP4":
        q = _AITER.quantize_mxfp4_q(query, multiplier)
        k_raw, k_descale = _AITER.quantize_mxfp4_k(key, k_mean)
        k = (_AITER.mxfp4_k_view(k_raw, k_descale), k_descale)
        routing_scale = None
    elif qk == "MXFP6":
        q = _AITER.quantize_mxfp6_q(query, multiplier)
        k_raw, k_descale_raw = _AITER.quantize_mxfp6_k(key, k_mean)
        k = _AITER.mxfp6_k_view(k_raw, k_descale_raw, *key.shape[:3])
        routing_scale = None
    else:
        q = _AITER.quantize_fp8_rotated(query)
        k = _AITER.quantize_fp8_rotated(key, k_mean)
        routing_scale = softmax_scale * (q[1] * k[1]).reshape(())

    if v_format == "BF16":
        v = _keep_bf16(value)
    elif v_format == "MXFP4":
        v_raw, v_descale = _AITER.quantize_v_mxfp4_fp6_p(value)
        v = (_AITER.mxfp4_v_view(v_raw, v_descale, key.shape[1]), v_descale)
    elif v_format == "MXFP6":
        v = _AITER.quantize_v_mxfp6_fp6_p(value)
    elif recipe.k_packed is not None:
        # f6f8: an MX Q/K pairs with the per-channel FP8 V.
        v = _AITER.quantize_v_fp8(value)
    else:
        v = _AITER.quantize_fp8(value)

    return _Quantized(
        q=q, k=k, v=v,
        k_source=key if recipe.k_packed is not None else None,
        v_source=value if recipe.v_packed is not None else None,
        routing_q=query if recipe.k_packed is not None else q[0],
        k_shift=k_mean if recipe.k_packed is not None else None,
        q_scale=q_scale,
        routing_scale=routing_scale,
    )


def _force_blocks_from_tokens(exact_tokens, key_seqlen, blocked_seqlen, kv_tile):
    """Per-KV-block "compute this exactly" flags from a per-token mask.

    A block is forced on when it holds any flagged token, because the block is the finest thing
    the selection can express. Follows K/V through the same trim so the two cannot fall out of
    step, and is pure tensor work -- no host-side read of device data -- so it stays traceable.

    blocked_seqlen is the key length rounded up to whole blocks, whether or not K/V themselves were
    padded to it: a ragged row still counts its short last block as a block. kv_tile is passed
    rather than read, because it is a property of the recipe this call dispatches on and the
    caller has already resolved it.
    """
    if key_seqlen is not None and key_seqlen < exact_tokens.shape[0]:
        exact_tokens = exact_tokens[:key_seqlen]
    pad = blocked_seqlen - exact_tokens.shape[0]
    if pad > 0:
        exact_tokens = torch.nn.functional.pad(exact_tokens, (0, pad))
    return exact_tokens.reshape(-1, kv_tile).any(dim=1)


def _force_rows_from_tokens(exact_query_tokens, query_seqlen, q_tile):
    """Per-query-tile "compute every KV block exactly" flags from a per-token mask, (num_q_tiles, 1).

    The query-side mirror of _force_blocks_from_tokens. Routing picks one selection per query
    tile, so a tile is the finest unit that can be made exact: a tile holding any flagged token
    is routed to every KV block, and its neighbours in that tile come along. The trailing axis is
    left at 1 so the result broadcasts against a per-KV-block force into the
    (num_q_tiles, num_kv_blocks) mask aiter ORs into its selection.
    """
    pad = -query_seqlen % q_tile
    if pad:
        exact_query_tokens = torch.nn.functional.pad(exact_query_tokens, (0, pad))
    return exact_query_tokens.reshape(-1, q_tile).any(dim=1, keepdim=True)


def sol_attn_routing_for(quantized, beta, recipe=_RECIPES["fp8"], force_blocks=None,
                         block_attn_mask=None, router="mean", jensen=False):
    """Pooled K/V, ragged LUT and selection bitmap for one call, as the kernel consumes them.

    quantized is what _quantize returns. num_heads is passed explicitly because the routing is per
    query head under GQA, while the pooled K/V it returns carry the KV head count instead.

    block_attn_mask supplies an already chosen selection instead of routing from beta, which is
    what lets a caller measure some other rule's selection through this same kernel and pooled
    correction. Pass exactly one of it and beta.

    jensen asks for each block's K variance (mean_k_var), which the kernel uses to add the
    second-order term to a pooled block's logit instead of stopping at its Jensen lower bound
    q . mean_k. router="error" spends the exact budget on the blocks whose pooled estimate is least
    reliable; it is meant for a kernel that takes mean_k_var, so it pairs naturally with jensen.
    """
    plan = _recipe_plan(recipe)
    k_scale_mode, v_scale_mode = plan.scale_modes[1], plan.scale_modes[2]
    return _AITER.prepare(
        quantized.routing_q,
        quantized.k[0],
        quantized.v[0],
        beta,
        # Not aiter's SOL_ATTN_TS_QO/TS_KV defaults: an opted-in run may be on the 64x64 row, and
        # pooling at the wrong tile hands the kernel pooled tensors of the wrong height. The
        # dispatch reads the same source, which is what keeps the two from drifting apart.
        *sol_attn_block_tile(recipe),
        num_heads=quantized.routing_q.shape[2],
        # An addressable E8M0-scaled operand pools in dequantized space and needs its scale; a
        # packed one pools from its source and is quantized again, so offering a scale is an error.
        k_scale=(quantized.k[1]
                 if k_scale_mode == _AITER.block_scale and recipe.k_packed is None else None),
        v_scale=(quantized.v[1]
                 if v_scale_mode == _AITER.block_scale and recipe.v_packed is None else None),
        k_source=quantized.k_source,
        v_source=quantized.v_source,
        k_packed_format=recipe.k_packed,
        v_packed_format=recipe.v_packed,
        force_block_mask=force_blocks,
        block_attn_mask=block_attn_mask,
        k_variance=jensen,
        router=router,
        routing_scale=quantized.routing_scale if router == "error" else None,
        q_scale=quantized.q_scale,
        k_shift=quantized.k_shift,
    )


def _head_cost_from_routing(routing):
    """Per-head count of exactly-computed KV blocks, float32 (nheads_q,).
    """
    mask = routing["block_attn_mask"]  # (batch, nheads_q, num_q_tiles, num_kv_blocks) bool
    return mask.sum(dim=(0, 2, 3), dtype=torch.float32)


# Below this many KV blocks, routing has too few samples to say anything. tau is a mean plus a
# multiple of a standard deviation taken across the blocks, so with a handful of them the
# threshold is noise, and whatever it does not select is replaced by block means that are then a
# large fraction of the whole sequence. 16 is a judgement call, not a measured cliff.
_MIN_USEFUL_KV_BLOCKS = 16
_short_kv_warned = set()


def _warn_if_kv_too_short(seqlen_k, kv_tile):
    """Say something when a call is too short for Sol-Attn to be worth routing.

    This is the failure mode that does not announce itself: nothing raises, the answer is just
    quietly worse than the dense backend would have given, which is easy to ship by accident when
    a model turns out to have a second attention site on a short sequence.

    Skipped under torch.compile. The branch folds away at trace time, which keeps the warning from
    being a side effect in a traced region -- so a fully compiled run will not see it, and this is
    a backstop for eager runs rather than a guarantee.
    """
    if torch.compiler.is_compiling():
        return
    blocks = seqlen_k // kv_tile
    if blocks >= _MIN_USEFUL_KV_BLOCKS or blocks in _short_kv_warned:
        return
    _short_kv_warned.add(blocks)
    import warnings

    warnings.warn(
        f"Sol-Attn was handed a {seqlen_k}-token KV, i.e. {blocks} blocks of {kv_tile}. Its "
        f"routing threshold is a mean plus a multiple of a standard deviation over the blocks, "
        f"which says little at that count, and the pooled correction is then standing in for much "
        f"of the sequence. Expect a worse result than a dense backend, not a faster one. If this "
        f"is a short text-only attention inside a model whose main attention is long, that site "
        f"should be on a dense backend.",
        RuntimeWarning,
        stacklevel=3,
    )


def _pad_kv_to_tile(key, value, kv_tile):
    """Right-pad BSHD K/V with zero tokens so seqlen_k is a whole number of KV blocks.

    Only for a row that does not declare ragged_kv. Every gfx950 Sol-Attn row does: it masks the
    keys past the end of a short last block itself, so a ragged call hands it the true length and
    aiter's routing keeps that short block exact. This stays for a build or geometry without that,
    because the alternative there is a refused launch -- Wan is ragged at every standard size
    (720p is 75600 tokens, 590.6 blocks of 128).

    Zero tokens are not free: a zero key scores q . 0 = 0 rather than -inf, so the pad draws
    softmax weight exp(-max) per token instead of none. Q is deliberately left alone: nothing
    constrains seqlen_q, and padding it would only add rows to slice back off the output.
    """
    pad = -key.shape[1] % kv_tile
    if pad == 0:
        return key, value
    # BSHD, so the seqlen axis is the second of four and F.pad counts from the last.
    widths = (0, 0, 0, 0, 0, pad)
    return (torch.nn.functional.pad(key, widths),
            torch.nn.functional.pad(value, widths))


@functools.cache
def _dump_call_indices():
    """Which call indices SOL_ATTN_DUMP saves. See XFUSER_SOL_ATTN_DUMP_CALLS in xfuser/envs.py."""
    from xfuser.envs import environment_variables

    spec = environment_variables["SOL_ATTN_DUMP_CALLS"]()
    return frozenset(int(part) for part in spec.split(",") if part.strip())


def _maybe_dump(path, query, key, value):
    """Save selected calls' BSHD q/k/v so kernel benchmarks can be replayed on real tensors.

    A single requested call index writes `path` itself; several write `path` with the index inserted
    before the suffix, so a set of samples lands beside each other and stays attributable to the
    call it came from.
    """
    if not path:
        return
    call_indices = _dump_call_indices()
    index = _maybe_dump._calls
    _maybe_dump._calls = index + 1
    if index not in call_indices:
        return
    if len(call_indices) > 1:
        root, ext = os.path.splitext(path)
        path = f"{root}.call{index}{ext}"
    torch.save({"q": query.detach().cpu(), "k": key.detach().cpu(), "v": value.detach().cpu(),
                "call": index}, path)


_maybe_dump._calls = 0


def sol_attn_bhsd(query, key, value, is_causal=False, beta=1.0, softmax_scale=None,
                  routing=None, ring_world_size=1, dump_path=None,
                  return_head_cost=False, recipe="fp8", key_seqlen=None,
                  exact_tokens=None, exact_query_tokens=None, sequence_permutation=None,
                  sequence_inverse_permutation=None, router="mean", jensen=False,
                  jensen_cap=True, tilt_rank=0):
    """Sol-Attn over BHSD tensors, returning (BHSD bf16 output, per-head cost or None).

    query/key/value are (batch, nheads, seqlen, head_dim) bf16 tensors. They are permuted into the
    BSHD layout the kernel takes and handed to aiter's mha_v4 Sol-Attn row for `recipe`, which owns
    the Q/K Hadamard rotation, format validation and the ASM launch. See SOL_ATTN_RECIPES for the
    rows this build ships.

    return_head_cost asks for the per-head count of exactly-computed KV blocks, float32 (nheads_q,),
    which the Ulysses head balancer consumes. It is opt-in rather than free: the routing dict it
    reduces is internal to the raw entrypoint, so asking for it moves the call onto aiter's packed
    API and makes quantization and routing explicit here. MXFP8 has no raw entrypoint and takes
    that path always, and so does a caller-supplied `routing`, a forced block, router="error" or
    jensen=True. Every path is traceable; none graph-break.

    router picks how routing scores blocks against beta: "mean" (the paper's q_mean . k_mean) or
    "error" (second-order mass times in-block spread, which spends the exact budget where pooling
    is least reliable; not available with a packed K). jensen passes each block's K variance to
    the kernel so a pooled block's logit carries its second-order term rather than stopping at
    q . mean_k. Both change the result, both default off, and see SOL_ATTN_ROUTERS.

    jensen_cap damps each block's Jensen term to at most ln(kv_tile) (aiter's sol_jensen_cap): the
    term is a block's log-mean-exp only while its scores are near Gaussian, and uncapped it hands
    late Wan layers' pooled blocks the whole softmax. False keeps the old uncapped term, for
    comparison. tilt_rank > 0 adds that many leading Cov(v, k) q terms to each pooled block's mean
    value (aiter's pooled value tilt, a PoC), at the block's capped weight; it implies jensen.

    key_seqlen is the number of real KV tokens when the caller has already padded the sequence for
    its own alignment, as MiniMax-H3 does to pack one request into one audiovisual sequence. Sol-Attn
    takes no attention mask, so the padded tail is dropped here rather than attended: left in, it
    would draw softmax weight and also enter the routing threshold's per-tile mean and standard
    deviation, which is a quieter error than the weight itself. Defaults to the whole tensor.

    exact_tokens is a per-KV-token bool mask, (seqlen_k,), naming tokens that must be computed
    exactly rather than left to the pooled approximation. Routing still runs; these are added to
    what it picked. A caller whose sequence is one modality does not need this. A caller packing
    several into one sequence does: the threshold is built from statistics over every block, so a
    modality holding a small share of the sequence is judged against a distribution the majority
    wrote, and its own queries lose the blocks they needed. Flagging the minority costs about one
    exact block per block it occupies, which is few by the same token.

    exact_query_tokens is the query-side counterpart, (seqlen_q,) bool: every query tile holding a
    flagged token computes ALL its KV blocks exactly, so those rows come out as dense attention
    would give them. exact_tokens makes some keys exact for every query; this makes some queries
    exact against every key. It costs a full row of blocks per flagged tile.

    sequence_permutation optionally reorders the shared Q/K/V sequence before routing, and
    sequence_inverse_permutation restores the query rows on return. This is used by MiniMax-H3 to
    make spatially adjacent video tokens share blocks while leaving its external packed-row
    contract unchanged.
    """
    if _AITER is None:
        raise SolAttnUnsupported(
            "Sol-Attn requires aiter with mha_v4_sol and sol_prepare (the mha_v4-sol-attn "
            "branch); please update AITER")
    recipe = _resolve_recipe(recipe)
    # Resolved once and threaded, not re-read at each site. The answer is per recipe now, and the
    # LUT, the bitmap, the pooled K/V and the tile pad all have to be cut to the same one.
    block_tile = sol_attn_block_tile(recipe)
    kv_tile = block_tile[1]

    if sequence_permutation is not None:
        if query.shape[2] != key.shape[2] or key.shape[2] != value.shape[2]:
            raise SolAttnUnsupported(
                "Sol-Attn sequence reordering requires equal Q/K/V sequence lengths."
            )
        if sequence_permutation.ndim != 1 or sequence_permutation.numel() != query.shape[2]:
            raise ValueError(
                "Sol-Attn sequence permutation must be one-dimensional and match "
                f"the sequence length ({query.shape[2]}), got "
                f"{list(sequence_permutation.shape)}."
            )
        if sequence_inverse_permutation is None:
            sequence_inverse_permutation = torch.argsort(sequence_permutation)
        query = query.index_select(2, sequence_permutation)
        key = key.index_select(2, sequence_permutation)
        value = value.index_select(2, sequence_permutation)
        if exact_tokens is not None:
            exact_tokens = exact_tokens.index_select(0, sequence_permutation)
        if exact_query_tokens is not None:
            exact_query_tokens = exact_query_tokens.index_select(0, sequence_permutation)

    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    def restore_output(out):
        out = out.permute(0, 2, 1, 3)
        if sequence_inverse_permutation is not None:
            out = out.index_select(2, sequence_inverse_permutation)
        return out

    check_sol_attn_supported(
        query, key, value, is_causal, ring_world_size=ring_world_size, recipe=recipe,
        router=router,
    )
    jensen = jensen or tilt_rank > 0
    check_sol_second_order(recipe, jensen, jensen_cap, tilt_rank)
    _maybe_dump(dump_path, query, key, value)
    # Before any tile pad, so the two do not stack: the caller's alignment is dropped and the row
    # is handed the real length.
    if key_seqlen is not None and key_seqlen < key.shape[1]:
        key, value = key[:, :key_seqlen], value[:, :key_seqlen]
    real_kv_len = key.shape[1]
    ragged = _ragged_kv(recipe.id, block_tile)
    if not ragged:
        key, value = _pad_kv_to_tile(key, value, kv_tile)
    num_kv_blocks = -(-key.shape[1] // kv_tile)
    _warn_if_kv_too_short(key.shape[1], kv_tile)

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    force_blocks = (
        None if exact_tokens is None
        else _force_blocks_from_tokens(exact_tokens, key_seqlen, num_kv_blocks * kv_tile, kv_tile)
    )
    # A ragged row is handed the true length, so aiter sees the short last block and keeps it
    # exact itself (partial_tail in its routing). A padded one hides it: the approximate branch
    # weights a pooled block by the CONSTANT block size, right only for a full block, so the real
    # last block is named here instead. The raw entrypoint cannot take a forced block, so a padded
    # ragged call goes packed.
    if not ragged and real_kv_len % kv_tile:
        tail = torch.zeros(num_kv_blocks, dtype=torch.bool, device=key.device)
        tail[(real_kv_len - 1) // kv_tile] = True
        force_blocks = tail if force_blocks is None else (force_blocks | tail)
    if exact_query_tokens is not None:
        rows = _force_rows_from_tokens(exact_query_tokens, query.shape[1], block_tile[0])
        force_blocks = rows if force_blocks is None else (rows | force_blocks)
        if force_blocks.shape[-1] == 1:
            # Rows alone: widen to every KV block, since aiter checks the trailing axis.
            force_blocks = force_blocks.expand(-1, num_kv_blocks)

    # The raw entrypoint routes internally from beta with the mean router and no K variance, so it
    # cannot be told about forced blocks, the error router or the Jensen term any more than it can
    # be asked for the head cost.
    if (routing is None and not return_head_cost and force_blocks is None
            and router == "mean" and not jensen and recipe.routes_through_raw):
        qk, v_fmt = _format(recipe.qk_format), _format(recipe.v_format)
        out = _AITER.sol(query, key, value, qk, qk, v_fmt, beta=beta,
                         softmax_scale=softmax_scale,
                         block_tile=block_tile)
        return restore_output(out), None

    # Quantize exactly as the raw entrypoint would, K smoothing included, so asking for the head
    # cost or forcing a block cannot change the output. On the FP8 rows that means
    # quantize_fp8_rotated for Q/K: rotating both by the same orthonormal matrix leaves Q @ K.T
    # alone while spreading the outliers that dominate fp8 error.
    quantized = _quantize(recipe, query, key, value, softmax_scale)
    if routing is None:
        routing = sol_attn_routing_for(quantized, beta, recipe, force_blocks=force_blocks,
                                       router=router, jensen=jensen)
    q, k, v = quantized.q, quantized.k, quantized.v
    # Pooled K/V and their scales as routing built them, or as the capped Jensen term and the tilt
    # rewrite them: the tilt's factor planes sit behind plane 0 of the same allocations.
    pooled = (_second_order(recipe, quantized, routing, softmax_scale, jensen_cap, tilt_rank)
              if jensen else
              dict(mean_k=routing["mean_k"], mean_v=routing["mean_v"],
                   mean_k_scale=routing["mean_k_scale"], mean_v_scale=routing["mean_v_scale"]))
    plan = _recipe_plan(recipe)
    qk_fmt, v_fmt = _format(recipe.qk_format), _format(recipe.v_format)
    out = _AITER.packed(
        q[0], k[0], v[0],
        q[1], k[1], v[1],
        qk_fmt, qk_fmt, v_fmt,
        *plan.scale_modes,
        v_pack=plan.v_pack,
        softmax_scale=softmax_scale,
        kv_block_indices=routing["kv_block_indices"],
        lut_start=routing["lut_start"],
        lut_count=routing["lut_count"],
        block_bitmap=routing["block_bitmap"],
        # mean_k / mean_v and their scales (None on the per-tensor rows, where the pooled operand
        # reuses the source descale; the MX rows pool in dequantized space and requantize, so the
        # pooled tensor has a scale of its own). With jensen also mean_k_var / mean_k_var_scale, for
        # the second-order term on each pooled block's logit, and with a tilt sol_tilt.
        **pooled,
        # The row to dispatch, not a preference: routing above already pooled and packed for this
        # geometry, so the kernel has to be the one that reads it that way.
        block_tile=block_tile,
    )
    return restore_output(out), _head_cost_from_routing(routing)


def sol_attn_dump_path():
    """Operand dump path from the environment, or None. See XFUSER_SOL_ATTN_DUMP in xfuser/envs.py.

    beta and hadamard are not read here: beta arrives per call through attention_kwargs from
    --solattn_beta, and the rotation is a property of the selected backend rather than a user knob.
    """
    from xfuser.envs import environment_variables

    return environment_variables["SOL_ATTN_DUMP"]() or None
