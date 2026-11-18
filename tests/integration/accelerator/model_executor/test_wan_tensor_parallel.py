"""Wan under tensor parallelism, alone and next to Ulysses or ring attention, matches one GPU.

TP shards the heads and the FFN, splits the residual rows across the TP group
and normalizes Q/K over heads held by different ranks. Cross-attention is
either sharded the same way or kept whole on every rank's own rows. The
collectives run next to their GEMMs, pipelined against them in chunks (which
holds the rows block-cyclically), inside torch's fused symm_mem ops (on ROCm, their fallback or,
on a torch patched with patches/pytorch, their SDMA path), or (ROCm) with the all-gathers on SDMA
copy engines in pieces. The model pads the sequence to a multiple of SP x TP x chunks, so
a short sequence also checks the pad.
Tiny random weights in fp32 on SDPA_EFFICIENT, so only reduction order
differs from one GPU; nothing is downloaded.
"""

import pytest
import torch

from _spawn import spawn_accelerator_ranks

pytestmark = [pytest.mark.multi_gpu, pytest.mark.slow]

_TINY_CONFIG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=4,
    attention_head_dim=128,
    in_channels=4,
    out_channels=4,
    text_dim=32,
    freq_dim=32,
    ffn_dim=256,
    num_layers=2,
    rope_max_seq_len=64,
)
TEXT_LEN = 16
# Latent (frames, height, width). 9 tokens after the 1x2x2 patching, padded to 10 or 12 rows.
PADDED = (3, 2, 6)
# 256 tokens, which split evenly on up to four ranks.
UNPADDED = (4, 16, 16)
# (tensor parallel, ulysses, ring)
MODES = {"tp2": (2, 1, 1), "tp4": (4, 1, 1), "tp2-ulysses2": (2, 2, 1), "tp2-ring2": (2, 1, 2)}
# Eight GPUs, so only for the cases that ask for it: two TP4 groups next to ring attention (the tiny
# model has too few heads for TP4 with Ulysses).
WIDE_MODES = {"tp4-ring2": (4, 1, 2)}
# Largest relative L2 error against one GPU, measured <= 3.2e-7 on MI350X. Q/K normalized over one
# rank's heads errs by 2.5e-3 at TP 2, a dropped row-parallel FFN bias by 4.3e-3.
# Row-wise FP8 E4M3 at every TP site: measured 3.9e-3..4.4e-3 against the one-GPU bf16 output.
FP8_TOLERANCE = 2e-2
TOLERANCE = 1e-5


def _inputs(name, latent_shape, batch):
    generator = torch.Generator().manual_seed(1)
    inputs = dict(
        hidden_states=torch.randn(batch, _TINY_CONFIG["in_channels"], *latent_shape, generator=generator),
        encoder_hidden_states=torch.randn(batch, TEXT_LEN, _TINY_CONFIG["text_dim"], generator=generator),
        timestep=torch.full((batch,), 500.0),
    )
    if name == "wan-ti2v":
        # Wan2.2 TI2V: a timestep per token, so the per-token modulation splits with the residual rows.
        seq_len = int(torch.Size(latent_shape).numel()) // 4
        inputs["timestep"] = torch.full((batch, seq_len), 500.0)
        inputs["timestep"][:, : seq_len // latent_shape[0]] = 0
    return inputs


def _worker(rank, world_size, init_method, tp, ulysses, ring, name, latent_shape, options, output_path):
    cross_attention, chunks, batch, fused = (
        options["cross_attention"],
        options["chunks"],
        options["batch"],
        options["fused"],
    )
    import contextlib
    import os

    import torch.distributed._symmetric_memory as symm_mem

    if options.get("sdma_reduce_scatter"):
        os.environ["XDIT_TP_SDMA_REDUCE_SCATTER"] = "on"
    if options.get("fp8"):
        os.environ["XDIT_TP_FP8"] = "1"
    if options.get("persistent"):
        os.environ["XDIT_TP_PERSISTENT"] = "1"
    if options.get("row_align"):
        os.environ["XDIT_TP_ROW_ALIGN"] = str(options["row_align"])
    if options.get("relay"):
        # The relay takes the FP8 sites it fits (the persistent kernels on its transport); the rest fall back.
        os.environ.update(XDIT_TP_RELAY="1", XDIT_TP_PERSISTENT="1", XDIT_TP_IPC_WORKSPACE="1")
    torch_sdma = options.get("torch_sdma", False)
    if torch_sdma:
        os.environ.update(
            TORCH_SYMM_MEM_ENABLE_ROCM_SDMA="1",
            TORCH_SYMM_MEM_ROCM_SDMA_ALLOW_UNTESTED="1",
            TORCH_SYMM_MEM_ROCM_SDMA_CHUNKS="2",
        )

    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.models.transformers.transformer_wan import (
        xFuserWanTPTransformerBlock,
        xFuserWanTransformer3DWrapper,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(tensor_parallel_degree=tp, ulysses_degree=ulysses, ring_degree=ring)
    try:
        args = xFuserArgs(
            attention_backend="SDPA_EFFICIENT",
            tensor_parallel_degree=tp,
            ulysses_degree=ulysses,
            ring_degree=ring,
        )
        engine_config, _ = args.create_config()
        initialize_runtime_state(engine_config=engine_config)
        runtime_state = get_runtime_state()
        runtime_state.set_attention_backend("SDPA_EFFICIENT")
        # Without a pipeline there is no denoising schedule to advance, but the wrapper's forward asks to.
        runtime_state.increment_step_counter = lambda: None

        device = torch.device("cuda", rank)
        # Seeded on the CPU, so every process builds the same weights.
        torch.manual_seed(0)
        config = {**_TINY_CONFIG, **options.get("config", {})}
        model = xFuserWanTransformer3DWrapper(**config).eval().to(device=device, dtype=torch.float32)
        model.shard_for_tensor_parallel(
            shard_cross_attention=cross_attention == "sharded", overlap_chunks=chunks, fused_comm=fused
        )
        if tp > 1:
            heads = config["num_attention_heads"]
            assert all(isinstance(block, xFuserWanTPTransformerBlock) for block in model.blocks)
            assert model.blocks[0].attn1.heads == heads // tp
            assert model.blocks[0].attn2.heads == (heads // tp if cross_attention == "sharded" else heads)

        inputs = {
            key: tensor.to(device=device, dtype=tensor.dtype if key == "timestep" else torch.float32)
            for key, tensor in _inputs(name, latent_shape, batch).items()
        }
        # torch's symm_mem kernels segfault on ROCm; its test mode runs their collective + GEMM
        # fallback through the same op calls, so the wiring is still checked there.
        fallback = fused == "symm_mem" and torch.version.hip and not torch_sdma
        fused_kernels = symm_mem._test_mode() if fallback else contextlib.nullcontext()
        if options.get("qk_first_groups"):
            output = _run_qk_first(model, inputs, options["qk_first_groups"], fused_kernels)
        else:
            with torch.no_grad(), fused_kernels:
                output = model(**inputs, return_dict=False)[0]
        # the persistent path and the relay bypass torch's ops
        if torch_sdma and not options.get("persistent") and not options.get("relay"):
            from torch.distributed._symmetric_memory import _rocm_sdma

            ops = ("fused_all_gather_matmul", "fused_matmul_reduce_scatter")
            if options.get("fp8"):
                ops = ("fused_all_gather_scaled_matmul", "fused_scaled_matmul_reduce_scatter")
            calls = _rocm_sdma.num_calls
            assert min(calls[op] for op in ops) > 0, f"the SDMA path did not run: {calls}"
            if rank == 0:
                print(f"SDMA op calls: {dict(calls)}", flush=True)
        if options.get("persistent"):
            from xfuser.model_executor.layers import tp_persistent

            ran = [tp_persistent.calls[k] for k in ("all_gather_gemm", "gemm_reduce_scatter")]
            assert min(ran) > 0, f"the persistent FP8 kernels did not run: {tp_persistent.calls}"
        if rank == 0:
            torch.save(output.cpu(), output_path)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _run_qk_first(model, inputs, head_groups, fused_kernels):
    """--ulysses_a2a sdma with and without --ulysses_sdma_overlap_qkv (TP: Q/K's GEMM and send before V's
    GEMM, from the rows the all-gather kept): the outputs match and the Q/K-first path ran."""
    from types import SimpleNamespace

    from xfuser.config.ulysses_a2a import UlyssesA2AConfig
    from xfuser.core.distributed import get_runtime_state
    from xfuser.core.distributed.sdma import ulysses
    from xfuser.model_executor.layers import usp_sdma
    from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanSdmaAttnProcessor

    runtime = get_runtime_state().runtime_config
    sends = []
    original_send = ulysses.send

    def counted_send(process_group, gi, packed, tensors=None):
        sends.append(tuple(tensors) if tensors is not None else None)
        return original_send(process_group, gi, packed, tensors)

    ulysses.send = counted_send
    try:
        runtime.ulysses_a2a = UlyssesA2AConfig("sdma", head_groups)
        with torch.no_grad(), fused_kernels:
            expected = model(**inputs, return_dict=False)[0]
        runtime.ulysses_a2a = UlyssesA2AConfig("sdma", head_groups, overlap_qkv=True)
        usp_sdma.prepare(SimpleNamespace(transformer=model), ["transformer"], cpu_offload=False)
        assert type(model.blocks[0].attn1.processor) is xFuserWanSdmaAttnProcessor
        sends.clear()
        with torch.no_grad(), fused_kernels:
            output = model(**inputs, return_dict=False)[0]
        layers = len(model.blocks)
        qk_sends = sends.count((0, 1))
        assert qk_sends >= layers, f"the TP Q/K-first path did not run: sends {sends}"
        assert qk_sends == layers * head_groups, f"{qk_sends} Q/K sends for {layers} layers x {head_groups} groups"
        torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
    finally:
        ulysses.send = original_send
        runtime.ulysses_a2a = UlyssesA2AConfig()
    return output


def _run(directory, mode, name, latent_shape, cross_attention="sharded", chunks=1, batch=1, fused=None, **extra):
    tp, ulysses, ring = {**MODES, **WIDE_MODES}[mode] if mode else (1, 1, 1)
    output_path = directory / "output.pt"
    options = dict(cross_attention=cross_attention, chunks=chunks, batch=batch, fused=fused, **extra)
    spawn_accelerator_ranks(
        _worker,
        directory,
        world_size=tp * ulysses * ring,
        timeout=600,
        args=(tp, ulysses, ring, name, latent_shape, options, str(output_path)),
    )
    return torch.load(output_path)


@pytest.fixture(scope="module")
def one_gpu_output(tmp_path_factory):
    outputs = {}

    def get(name, latent_shape, batch=1):
        if (name, latent_shape, batch) not in outputs:
            outputs[name, latent_shape, batch] = _run(
                tmp_path_factory.mktemp("one-gpu"), None, name, latent_shape, batch=batch
            )
        return outputs[name, latent_shape, batch]

    return get


def _relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.parametrize("latent_shape", [UNPADDED, PADDED], ids=["unpadded", "padded"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ["wan", "wan-ti2v"])
def test_tensor_parallel_matches_one_gpu(tmp_path, one_gpu_output, name, mode, latent_shape):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    expected = one_gpu_output(name, latent_shape)

    actual = _run(tmp_path, mode, name, latent_shape)

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE, f"relative error {error:.3e}"


@pytest.mark.parametrize("latent_shape", [UNPADDED, PADDED], ids=["unpadded", "padded"])
@pytest.mark.parametrize("mode", ["tp2", "tp4", "tp2-ulysses2"])
def test_tensor_parallel_with_replicated_cross_attention_matches_one_gpu(tmp_path, one_gpu_output, mode, latent_shape):
    """Cross-attention unsharded on each rank's own rows, as XDIT_WAN_TP_CROSS_ATTENTION=replicated runs it."""
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    expected = one_gpu_output("wan", latent_shape)

    actual = _run(tmp_path, mode, "wan", latent_shape, cross_attention="replicated")

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE, f"relative error {error:.3e}"


@pytest.mark.parametrize("latent_shape", [UNPADDED, PADDED], ids=["unpadded", "padded"])
@pytest.mark.parametrize("cross_attention", ["sharded", "replicated"])
@pytest.mark.parametrize("mode, chunks", [("tp2", 2), ("tp2", 4), ("tp4", 2), ("tp2-ulysses2", 2), ("tp2-ring2", 2)])
@pytest.mark.parametrize("name", ["wan", "wan-ti2v"])
def test_tensor_parallel_with_chunked_overlap_matches_one_gpu(
    tmp_path, one_gpu_output, name, mode, chunks, cross_attention, latent_shape
):
    """Collectives pipelined against their GEMMs, as XDIT_TP_COMM_OVERLAP=chunked runs them."""
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    expected = one_gpu_output(name, latent_shape)

    actual = _run(tmp_path, mode, name, latent_shape, cross_attention=cross_attention, chunks=chunks)

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE, f"relative error {error:.3e}"


@pytest.mark.parametrize("chunks", [1, 2])
def test_tensor_parallel_batch_of_two_matches_one_gpu(tmp_path, one_gpu_output, chunks):
    """Two samples per forward: the chunked collectives' batched path, and the plain one."""
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    expected = one_gpu_output("wan", PADDED, batch=2)

    actual = _run(tmp_path, "tp2", "wan", PADDED, chunks=chunks, batch=2)

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE, f"relative error {error:.3e}"


@pytest.mark.parametrize("latent_shape", [UNPADDED, PADDED], ids=["unpadded", "padded"])
@pytest.mark.parametrize("cross_attention", ["sharded", "replicated"])
@pytest.mark.parametrize("mode", ["tp2", "tp4", "tp2-ulysses2"])
def test_tensor_parallel_with_fused_comm_matches_one_gpu(tmp_path, one_gpu_output, mode, cross_attention, latent_shape):
    """torch's symm_mem fused ops for every pair, as XDIT_TP_COMM_OVERLAP=symm_mem runs them.

    On CUDA this runs the real kernels. On ROCm, where they segfault, it runs
    torch's fallback for them, which checks the call wiring but not the kernels.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    expected = one_gpu_output("wan", latent_shape)

    actual = _run(tmp_path, mode, "wan", latent_shape, cross_attention=cross_attention, fused="symm_mem")

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE, f"relative error {error:.3e}"


@pytest.mark.rocm
@pytest.mark.parametrize("latent_shape", [UNPADDED, PADDED], ids=["unpadded", "padded"])
@pytest.mark.parametrize("cross_attention", ["sharded", "replicated"])
@pytest.mark.parametrize("mode, chunks", [("tp2", 1), ("tp2", 2), ("tp4", 2), ("tp2-ulysses2", 2)])
@pytest.mark.parametrize("reduce_scatter", ["chunked", "sdma"])
def test_tensor_parallel_with_sdma_all_gather_matches_one_gpu(
    tmp_path, one_gpu_output, reduce_scatter, mode, chunks, cross_attention, latent_shape
):
    """All-gathers on SDMA copy engines in pieces, as XDIT_TP_COMM_OVERLAP=sdma runs them; reduce-scatters
    as chunked overlap or, with XDIT_TP_SDMA_REDUCE_SCATTER=on, with their partial sums pushed on SDMA engines."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("SDMA copy engines need ROCm")
    expected = one_gpu_output("wan", latent_shape)

    actual = _run(
        tmp_path,
        mode,
        "wan",
        latent_shape,
        cross_attention=cross_attention,
        chunks=chunks,
        fused="sdma",
        sdma_reduce_scatter=reduce_scatter == "sdma",
    )

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE, f"relative error {error:.3e}"


@pytest.mark.rocm
@pytest.mark.parametrize("reduce_scatter", ["chunked", "sdma"])
@pytest.mark.parametrize("mode, batch", [("tp4-ring2", 1), ("tp2", 2)])
def test_tensor_parallel_with_sdma_on_several_groups_and_batches(tmp_path, one_gpu_output, mode, batch, reduce_scatter):
    """SDMA all-gathers (and reduce-scatters) with two TP4 groups next to ring attention on 8 GPUs, and with two samples."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("SDMA copy engines need ROCm")
    expected = one_gpu_output("wan", PADDED, batch=batch)

    actual = _run(
        tmp_path,
        mode,
        "wan",
        PADDED,
        chunks=2,
        batch=batch,
        fused="sdma",
        sdma_reduce_scatter=reduce_scatter == "sdma",
    )

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE, f"relative error {error:.3e}"


def _torch_has_sdma_patch():
    import importlib.util

    return importlib.util.find_spec("torch.distributed._symmetric_memory._rocm_sdma") is not None


@pytest.mark.rocm
@pytest.mark.parametrize("latent_shape", [UNPADDED, PADDED], ids=["unpadded", "padded"])
@pytest.mark.parametrize("mode, batch", [("tp2", 1), ("tp4", 1), ("tp2-ulysses2", 1), ("tp4-ring2", 1), ("tp2", 2)])
def test_tensor_parallel_with_patched_torch_symm_mem_matches_one_gpu(
    tmp_path, one_gpu_output, mode, batch, latent_shape
):
    """torch's symm_mem fused ops on their SDMA path (a torch patched with patches/pytorch, run with
    TORCH_SYMM_MEM_ENABLE_ROCM_SDMA=1), as XDIT_TP_COMM_OVERLAP=symm_mem runs them."""
    if not torch.cuda.is_available() or torch.version.hip is None or not _torch_has_sdma_patch():
        pytest.skip("needs ROCm and a torch patched with patches/pytorch")
    expected = one_gpu_output("wan", latent_shape, batch=batch)

    actual = _run(tmp_path, mode, "wan", latent_shape, batch=batch, fused="symm_mem", torch_sdma=True)

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE, f"relative error {error:.3e}"


@pytest.mark.rocm
@pytest.mark.parametrize("latent_shape", [UNPADDED, PADDED], ids=["unpadded", "padded"])
@pytest.mark.parametrize("mode, batch", [("tp2", 1), ("tp4", 1), ("tp2-ulysses2", 1), ("tp2", 2)])
def test_tensor_parallel_fp8_with_patched_torch_symm_mem_is_close_to_one_gpu(
    tmp_path, one_gpu_output, mode, batch, latent_shape
):
    """XDIT_TP_FP8=1: every TP site as torch's row-wise FP8 fused ops on their SDMA path; close to the
    one-GPU bf16 output within FP8 rounding."""
    if not torch.cuda.is_available() or torch.version.hip is None or not _torch_has_sdma_patch():
        pytest.skip("needs ROCm and a torch patched with patches/pytorch")
    expected = one_gpu_output("wan", latent_shape, batch=batch)

    actual = _run(tmp_path, mode, "wan", latent_shape, batch=batch, fused="symm_mem", torch_sdma=True, fp8=True)

    error = _relative_error(actual, expected)
    print(f"FP8 TP relative error {error:.3e}")
    assert error <= FP8_TOLERANCE, f"relative error {error:.3e}"


@pytest.mark.parametrize("latent_shape", [UNPADDED, PADDED], ids=["unpadded", "padded"])
@pytest.mark.parametrize("mode", ["tp2", "tp4", "tp2-ulysses2"])
def test_tensor_parallel_with_row_alignment_matches_one_gpu(tmp_path, one_gpu_output, mode, latent_shape):
    """XDIT_TP_ROW_ALIGN=256 pads each TP rank's rows to whole 256-row tiles; on the test model that is mostly
    pad, which must stay masked."""
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    expected = one_gpu_output("wan", latent_shape)

    actual = _run(tmp_path, mode, "wan", latent_shape, row_align=256)

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE, f"relative error {error:.3e}"


@pytest.mark.rocm
@pytest.mark.parametrize("latent_shape", [UNPADDED, PADDED], ids=["unpadded", "padded"])
# TP 2: every site's K (hidden 512, FFN 256, split two ways) has the two 128-wide K blocks the kernel needs.
@pytest.mark.parametrize("mode, batch", [("tp2", 1), ("tp2-ulysses2", 1), ("tp2", 2)])
def test_tensor_parallel_fp8_persistent_is_close_to_one_gpu(tmp_path, one_gpu_output, mode, batch, latent_shape):
    """XDIT_TP_PERSISTENT=1: the FP8 TP sites as persistent FlyDSL GEMMs on the SDMA transport (rows padded to whole
    256-row tiles); close to the one-GPU bf16 output within FP8 rounding."""
    if not torch.cuda.is_available() or torch.version.hip is None or not _torch_has_sdma_patch():
        pytest.skip("needs ROCm and a torch patched with patches/pytorch")
    pytest.importorskip("flydsl")
    expected = one_gpu_output("wan", latent_shape, batch=batch)

    actual = _run(
        tmp_path,
        mode,
        "wan",
        latent_shape,
        batch=batch,
        fused="symm_mem",
        torch_sdma=True,
        fp8=True,
        persistent=True,
        row_align=256,
    )

    error = _relative_error(actual, expected)
    assert error <= FP8_TOLERANCE, f"relative error {error:.3e}"


@pytest.mark.rocm
@pytest.mark.parametrize("latent_shape", [UNPADDED, PADDED], ids=["unpadded", "padded"])
@pytest.mark.parametrize("heads, head_groups", [(4, 1), (8, 2)])
@pytest.mark.parametrize("relay", [False, True], ids=["direct", "relay"])
def test_tensor_parallel_fp8_ulysses_qk_first_matches_and_is_close_to_one_gpu(
    tmp_path, one_gpu_output, heads, head_groups, relay, latent_shape
):
    """TP2 x Ulysses2, FP8 TP sites, SDMA Ulysses with --ulysses_sdma_overlap_qkv: self-attention's Q/K GEMM and
    exchange run before V's GEMM, which reads the rows the all-gather kept (the relay's receive buffer with
    XDIT_TP_RELAY=1); the output matches the same run without it and, for the stock head count, one GPU."""
    if not torch.cuda.is_available() or torch.version.hip is None or not _torch_has_sdma_patch():
        pytest.skip("needs ROCm and a torch patched with patches/pytorch")
    if relay:
        pytest.importorskip("flydsl")
    extra = dict(fused="symm_mem", torch_sdma=True, fp8=True, qk_first_groups=head_groups)
    if heads != _TINY_CONFIG["num_attention_heads"]:
        extra["config"] = dict(num_attention_heads=heads, attention_head_dim=64)
    if relay:
        extra.update(relay=True, row_align=256)

    actual = _run(tmp_path, "tp2-ulysses2", "wan", latent_shape, **extra)

    if "config" not in extra:
        error = _relative_error(actual, one_gpu_output("wan", latent_shape))
        assert error <= FP8_TOLERANCE, f"relative error {error:.3e}"
