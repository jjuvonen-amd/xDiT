"""Wan under tensor parallelism, alone and next to Ulysses or ring attention, matches one GPU.

TP shards the heads and the FFN, splits the residual rows across the TP group
and normalizes Q/K over heads held by different ranks. Cross-attention is
either sharded the same way or kept whole on every rank's own rows. The
collectives run either next to their GEMMs or pipelined against them in
chunks, which holds the rows block-cyclically. The model pads the sequence to
a multiple of SP x TP x chunks, so a short sequence also checks the pad.
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
# Largest relative L2 error against one GPU, measured <= 3.2e-7 on MI350X. Q/K normalized over one
# rank's heads errs by 2.5e-3 at TP 2, a dropped row-parallel FFN bias by 4.3e-3.
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
    cross_attention, chunks, batch = options["cross_attention"], options["chunks"], options["batch"]
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
        model = xFuserWanTransformer3DWrapper(**_TINY_CONFIG).eval().to(device=device, dtype=torch.float32)
        model.shard_for_tensor_parallel(shard_cross_attention=cross_attention == "sharded", overlap_chunks=chunks)
        if tp > 1:
            heads = _TINY_CONFIG["num_attention_heads"]
            assert all(isinstance(block, xFuserWanTPTransformerBlock) for block in model.blocks)
            assert model.blocks[0].attn1.heads == heads // tp
            assert model.blocks[0].attn2.heads == (heads // tp if cross_attention == "sharded" else heads)

        inputs = {
            key: tensor.to(device=device, dtype=tensor.dtype if key == "timestep" else torch.float32)
            for key, tensor in _inputs(name, latent_shape, batch).items()
        }
        with torch.no_grad():
            output = model(**inputs, return_dict=False)[0]
        if rank == 0:
            torch.save(output.cpu(), output_path)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _run(directory, mode, name, latent_shape, cross_attention="sharded", chunks=1, batch=1):
    tp, ulysses, ring = MODES[mode] if mode else (1, 1, 1)
    output_path = directory / "output.pt"
    options = dict(cross_attention=cross_attention, chunks=chunks, batch=batch)
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
