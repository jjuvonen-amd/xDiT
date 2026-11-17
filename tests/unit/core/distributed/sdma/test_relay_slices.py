"""How the relayed pair transport cuts a copy into one slice per other GPU."""

import pytest

from xfuser.core.distributed.sdma.relay import slices


@pytest.mark.parametrize("nbytes", [1, 255, 256, 7 * 256, 7 * 256 + 3, 1 << 20, 123_456_789])
@pytest.mark.parametrize("world, sender", [(4, 0), (4, 3), (8, 5)])
def test_slices_cover_the_copy_once(nbytes, world, sender):
    parts = slices(nbytes, sender, world)
    assert all(size > 0 for _, _, size in parts)
    assert all(receiver != sender for receiver, _, _ in parts)
    assert [r for r, _, _ in parts] == sorted({r for r, _, _ in parts})
    covered = 0
    for _, offset, size in parts:
        assert offset == covered
        covered += size
    assert covered == nbytes
    # every slice but the last starts and ends on a 256-byte boundary
    assert all(offset % 256 == 0 and size % 256 == 0 for _, offset, size in parts[:-1])


def test_large_copies_use_every_other_gpu():
    parts = slices(8 << 20, 2, 8)
    assert [r for r, _, _ in parts] == [0, 1, 3, 4, 5, 6, 7]
