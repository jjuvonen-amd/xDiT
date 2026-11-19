"""One Wan TP chain, both TP groups at once (U2 x TP4): ffn_down GEMM + reduce-scatter -> gated residual add,
LayerNorm and modulation of this rank's rows -> next block's QKV all-gather + GEMM. Sequential (the two relayed ops
with the row-wise work in between) vs pipelined (TPRelay.gemm_rs_then_ag_fp8: each reduced piece goes on as it
lands). torchrun --nproc-per-node=8 bench_wan_chain.py"""

import os

import torch
import torch.distributed as dist

ROWS = int(os.environ.get("ROWS", "9472"))  # Wan 720p x 81 frames, rows per TP rank (aligned)
K1, N1, N2 = 3456, 5120, 3840  # ffn_down (K = 13824 / 4) -> hidden 5120 -> qkv (3 x 5120 / 4)


def main():
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel, ipc_workspace
    from xfuser.core.distributed.parallel_state import get_tp_group
    from xfuser.model_executor.layers import tensor_parallel as tpm
    from xfuser.model_executor.layers import tp_relay

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed_environment(rank=rank, world_size=8, local_rank=rank)
    initialize_model_parallel(tensor_parallel_degree=4, ulysses_degree=2)
    ipc_workspace.install_torch_sdma_inbox()
    relay = tp_relay.relay_for(get_tp_group().device_group.group_name)
    g = torch.Generator(device="cuda").manual_seed(rank)
    dev = "cuda"
    w1q, w1s = tpm._rowwise_fp8(torch.randn(N1, K1, device=dev, dtype=torch.bfloat16, generator=g) * K1**-0.5)
    w2q, w2s = tpm._rowwise_fp8(torch.randn(N2, N1, device=dev, dtype=torch.bfloat16, generator=g) * N1**-0.5)
    w1_t, w1_s, w2_t, w2_s = w1q.t(), w1s.view(1, -1), w2q.t(), w2s.view(1, -1)
    xq, xs = tpm._rowwise_fp8(torch.randn(4 * ROWS, K1, device=dev, dtype=torch.bfloat16, generator=g))
    residual = torch.randn(ROWS, N1, device=dev, dtype=torch.bfloat16, generator=g)
    gate, scale, shift = (torch.randn(N1, device=dev, generator=g) * 0.1 for _ in range(3))
    norm = torch.nn.LayerNorm(N1, elementwise_affine=False, eps=1e-6)
    C = relay.chunks(ROWS)
    new_res = torch.empty_like(residual)

    def then(r0, y):  # gated residual add (kept), then the next block's norm1 + modulation
        h = residual[r0 : r0 + y.shape[0]].float() + y.float() * gate
        new_res[r0 : r0 + y.shape[0]] = h.to(torch.bfloat16)
        return (norm(h) * (1 + scale) + shift).to(torch.bfloat16)

    def sequential():
        red = relay.gemm_reduce_scatter_fp8(xq, xs, w1_t, w1_s, C)
        q, s = tpm._rowwise_fp8(then(0, red))
        return relay.all_gather_gemm_fp8(q, s, w2_t, w2_s, C)

    def pipelined():
        return relay.gemm_rs_then_ag_fp8(xq, xs, w1_t, w1_s, then, w2_t, w2_s, C)[1]

    res = {}
    with torch.inference_mode():
        a, b = sequential().clone(), pipelined().clone()
        torch.cuda.synchronize()
        same = torch.tensor([float(torch.equal(a, b))], device=dev)
        dist.all_reduce(same, op=dist.ReduceOp.MIN)
        for name, fn in (
            ("sequential", sequential),
            ("pipelined", pipelined),
            ("sequential2", sequential),
            ("pipelined2", pipelined),
        ):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s0.record()
            for _ in range(30):
                fn()
            s1.record()
            torch.cuda.synchronize()
            t = torch.tensor([s0.elapsed_time(s1) / 30], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            res[name] = round(t.item(), 3)
    if rank == 0:
        print(
            f"CHAIN {os.environ.get('TAG', '')}: pieces {C} bitwise_equal {bool(same.item())} "
            + " ".join(f"{k} {v} ms" for k, v in res.items())
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
