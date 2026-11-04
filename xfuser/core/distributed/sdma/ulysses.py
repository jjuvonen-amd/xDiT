"""Ulysses attention with its all-to-alls on the SDMA engines, pipelined over head groups.

Ulysses sends each rank's Q/K/V for head block p to rank p, which then attends over
the whole sequence for its heads and sends each rank's rows back. This does both
exchanges with the SDMA transport (transport.py) and splits each rank's heads into
`head_groups` groups: group g's attention starts once group g's Q/K/V have landed,
while later groups are still moving, and its output is pushed back as soon as it is
done, while later groups attend.

Layouts, for P ranks, local sequence s, H heads (H / P per rank, split into groups of
h_g), n = 3 tensors (Q, K, V), batch b, head dim d:

- send (this rank's, packed on the stream): [G][P][n][b][s][h_g][d], so every
  (group, peer) is one contiguous run of n * b blocks of [s, h_g, d].
- inbox (shared): [G][n][b][P * s][h_g][d]. Rank r's blocks land at rows r * s, so
  group g's Q, K and V are bshd tensors over the whole sequence, which is the layout
  the attention kernels read: no unpacking.
- attention output of group g: bshd [b][P * s][h_g][d], so the rows rank p gets
  back are one contiguous [s, h_g, d] block per batch item.
- out inbox (shared): [G][P][b][s][h_g][d], unpacked with one copy per group into
  the [b, s, H, d] output, which the output projection reads as [b, s, H * d].

Ordering needs no barrier after the first call: each rank pushes a call's Q/K/V
only once its stream has received every peer's output of its previous call, and a
peer sends that output only after its attention, which reads its inbox, is done,
and after this rank's earlier pushes into it have landed. So the inboxes, the send
buffer and the flags (reset by the waits) are free whenever they are written. That
holds as long as every Ulysses attention of the group goes through this function;
an attention output is held until the next call for the same reason.
"""

from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist

from xfuser.core.distributed.sdma import transport

# What `attend` gets: Q, K, V of one head group as [b, h_g, P * s, d] views of bshd
# tensors; it returns that group's attention output as [b, h_g, P * s, d].
Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

_N = 3  # tensors exchanged: Q, K, V
# Head groups at most: the flag buffer is sized for this many, so changing the group count
# between calls needs no new buffer.
MAX_HEAD_GROUPS = 16


def split_heads(heads: int, groups: int) -> List[int]:
    """Sizes of `groups` head groups covering `heads` heads, as even as possible, larger first."""
    groups = max(1, min(groups, heads, MAX_HEAD_GROUPS))
    return [heads // groups + (1 if i < heads % groups else 0) for i in range(groups)]


class _Layout:
    """Byte offsets of one call shape in the send buffer, inbox and out inbox."""

    def __init__(self, world: int, b: int, s: int, heads: int, d: int, dtype: torch.dtype, groups: int):
        elt = torch.empty((), dtype=dtype).element_size()
        self.world, self.b, self.s, self.d, self.elt, self.dtype = world, b, s, d, elt, dtype
        self.per_rank = heads // world
        self.sizes = split_heads(self.per_rank, groups)
        self.groups = len(self.sizes)
        self.head_offsets = [sum(self.sizes[:g]) for g in range(self.groups)]
        # One [s, h_g, d] block per group.
        self.block = [s * h * d * elt for h in self.sizes]
        # Every group region holds P * n * b blocks (send, inbox) or P * b blocks (out inbox).
        self.send_offsets = self._prefix([world * _N * b * blk for blk in self.block])
        self.inbox_offsets = self.send_offsets
        self.out_offsets = self._prefix([world * b * blk for blk in self.block])
        self.send_bytes = self.send_offsets[-1]
        self.out_bytes = self.out_offsets[-1]

    @staticmethod
    def _prefix(sizes):
        out = [0]
        for size in sizes:
            out.append(out[-1] + size)
        return out


class _State:
    """Per Ulysses group: the send buffer and the attention outputs held for one call."""

    def __init__(self):
        self.send: Optional[torch.Tensor] = None
        self.held: List[torch.Tensor] = []
        self.layouts: Dict[tuple, _Layout] = {}


_states: Dict[transport.PeerGroup, _State] = {}


def _view(buf: torch.Tensor, offset: int, shape, dtype) -> torch.Tensor:
    numel = 1
    for n in shape:
        numel *= n
    elt = torch.empty((), dtype=dtype).element_size()
    return buf[offset : offset + numel * elt].view(dtype).view(shape)


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attend: Attend,
    process_group: dist.ProcessGroup,
    head_groups: int = 2,
    concurrent: bool = False,
) -> torch.Tensor:
    """Ulysses attention over `process_group`, both all-to-alls on SDMA engines.

    `query`, `key`, `value` are this rank's [b, H, s, d] (any strides), all of one
    shape and dtype; H must divide by the group size. Returns the attention output
    for this rank's rows and every head as a contiguous [b, s, H, d] tensor.

    With `concurrent`, each head group attends on a stream of its own (the first at
    high priority), so a group whose data has landed fills the GPU while the one
    before it drains instead of waiting for it to finish: fewer heads per kernel
    leave a larger part of its last wave idle.
    """
    if not (query.shape == key.shape == value.shape) or not (query.dtype == key.dtype == value.dtype):
        raise ValueError("SDMA Ulysses attention needs Q, K and V of one shape and dtype")
    g = transport.group_for(process_group)
    P, r = g.world, g.rank
    b, heads, s, d = query.shape
    if heads % P:
        raise ValueError(f"{heads} heads do not divide over {P} Ulysses ranks")
    dtype, device = query.dtype, query.device
    state = _states.get(g)
    if state is None:
        state = _states[g] = _State()
    shape_key = (b, s, heads, d, dtype, head_groups)
    lay = state.layouts.get(shape_key)
    if lay is None:
        lay = state.layouts[shape_key] = _Layout(P, b, s, heads, d, dtype, head_groups)
    G, n_peers = lay.groups, P - 1
    stream = torch.cuda.current_stream()

    inbox = g.buffer("ulysses_inbox", lay.send_bytes)
    outbox = g.buffer("ulysses_outbox", lay.out_bytes)
    # Words [0, G * P): input flags, [G * P, 2 * G * P): output flags; word p is from rank p.
    flags = g.buffer("ulysses_flags", 2 * MAX_HEAD_GROUPS * P * 4, memory="uncached")
    if state.send is None or state.send.numel() < lay.send_bytes:
        state.send = torch.empty(lay.send_bytes, dtype=torch.uint8, device=device)
    send = state.send

    n_data = G * n_peers * (_N * b + b)
    slot, call = transport.start(n_data, 2 * G * n_peers)
    # The Q/K/V pushes are queued first, so the host issues them while the GPU packs: they wait
    # for trigger 0, which the stream releases once the packing is done. Trigger 1 + g lets
    # group g's output go once its attention is done.
    for gi in range(G):
        blk = lay.block[gi]
        transport.push(
            g,
            slot,
            0,
            gi * n_peers * _N * b,
            gi * n_peers,
            send.data_ptr() + lay.send_offsets[gi],
            _N * b * blk,
            blk,
            inbox,
            lay.inbox_offsets[gi] + r * blk,
            P * blk,
            blk,
            _N * b,
            flags,
            (gi * P + r) * 4,
        )

    # Pack: send[g][p][t] = tensor t's heads p * H/P + offset_g ... for group g, as [b, s, h_g, d].
    for gi, (h, off) in enumerate(zip(lay.sizes, lay.head_offsets)):
        packed = _view(send, lay.send_offsets[gi], (P, _N, b, s, h, d), dtype)
        for t, x in enumerate((query, key, value)):
            src = x.view(b, P, lay.per_rank, s, d)[:, :, off : off + h].permute(1, 0, 3, 2, 4)
            packed[:, t].copy_(src)
    flags.local[: 2 * G * P * 4].view(torch.int32).view(2, G, P)[:, :, r].fill_(1)
    transport.release(slot, 0, stream)
    # This rank's own blocks, on the stream.
    for gi, h in enumerate(lay.sizes):
        mine = _view(send, lay.send_offsets[gi], (P, _N, b, s, h, d), dtype)[r]
        _view(inbox.local, lay.inbox_offsets[gi], (_N, b, P, s, h, d), dtype)[:, :, r].copy_(mine)

    held = []
    out_first_data = G * n_peers * _N * b
    if concurrent:
        ready = torch.cuda.Event()
        ready.record(stream)
        side = _streams(device, G)
    for gi, h in enumerate(lay.sizes):
        blk = lay.block[gi]
        group_stream = stream
        if concurrent:
            group_stream = side[gi]
            group_stream.wait_event(ready)
        with torch.cuda.stream(group_stream):
            held.append(
                _attend_group(
                    lay, gi, h, blk, g, slot, call, inbox, outbox, flags, attend, group_stream, out_first_data
                )
            )
    if concurrent:
        for gi in range(G):
            stream.wait_stream(side[gi])
    # The previous call's outputs have been read by every peer's engines: each peer sent this
    # call's Q/K/V only after receiving them, and the waits above are behind those Q/K/V.
    state.held = held

    result = torch.empty(b, s, P, lay.per_rank, d, dtype=dtype, device=device)
    for gi, (h, off) in enumerate(zip(lay.sizes, lay.head_offsets)):
        transport.wait(stream, flags.ptr + (G + gi) * P * 4, P, G + gi, call)
        got = _view(outbox.local, lay.out_offsets[gi], (P, b, s, h, d), dtype)
        result[:, :, :, off : off + h].copy_(got.permute(1, 2, 0, 3, 4))
    return result.view(b, s, heads, d)


_side_streams: Dict[torch.device, List[torch.cuda.Stream]] = {}


def _streams(device: torch.device, count: int) -> List[torch.cuda.Stream]:
    """`count` side streams on `device`, the first at high priority."""
    streams = _side_streams.setdefault(device, [])
    while len(streams) < count:
        streams.append(torch.cuda.Stream(device=device, priority=-1 if not streams else 0))
    return streams


def _attend_group(lay, gi, h, blk, g, slot, call, inbox, outbox, flags, attend, stream, out_first_data):
    """Group gi on `stream`: wait for its Q/K/V, attend, and push its output back to every rank."""
    P, b, s, d, r, G, n_peers = lay.world, lay.b, lay.s, lay.d, g.rank, lay.groups, g.world - 1
    transport.wait(stream, flags.ptr + gi * P * 4, P, gi, call)
    qkv = _view(inbox.local, lay.inbox_offsets[gi], (_N, b, P * s, h, d), lay.dtype)
    out = attend(qkv[0].transpose(1, 2), qkv[1].transpose(1, 2), qkv[2].transpose(1, 2))
    out = out.transpose(1, 2)
    if out.dtype != lay.dtype:
        out = out.to(lay.dtype)
    if not out.is_contiguous():
        out = out.contiguous()
    transport.release(slot, 1 + gi, stream)
    transport.push(
        g,
        slot,
        1 + gi,
        out_first_data + gi * n_peers * b,
        G * n_peers + gi * n_peers,
        out.data_ptr(),
        blk,
        P * blk,
        outbox,
        lay.out_offsets[gi] + r * b * blk,
        blk,
        blk,
        b,
        flags,
        ((G + gi) * P + r) * 4,
    )
    _view(outbox.local, lay.out_offsets[gi], (P, b, s, h, d), lay.dtype)[r].copy_(out.view(b, P, s, h, d)[:, r])
    return out
