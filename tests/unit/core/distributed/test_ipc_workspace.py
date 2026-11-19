"""When the TP SDMA paths take the hipIpc workspace instead of torch's symmetric-memory one."""

import pytest
import torch

from xfuser.core.distributed import ipc_workspace


@pytest.fixture
def decide(monkeypatch):
    def run(mode, version, hip="7.2"):
        monkeypatch.setattr(ipc_workspace, "_enabled", None)
        monkeypatch.setenv("XDIT_TP_IPC_WORKSPACE", mode)
        monkeypatch.setattr(ipc_workspace, "amdgpu_version", lambda: version)
        monkeypatch.setattr(torch.version, "hip", hip)
        return ipc_workspace.enabled()

    return run


@pytest.mark.parametrize(
    "version, expected",
    [((6, 16, 6), True), ((6, 16, 13), True), ((7, 1, 3), False), ((7, 2, 0), False), (None, True)],
)
def test_auto_follows_the_amdgpu_version(decide, version, expected):
    assert decide("auto", version) is expected


@pytest.mark.parametrize("version", [(6, 16, 6), (7, 1, 3), None])
def test_explicit_modes_override_the_version(decide, version):
    assert decide("1", version) is True
    assert decide("0", version) is False


def test_never_without_rocm(decide):
    assert decide("1", (6, 16, 6), hip=None) is False


def test_rejects_unknown_modes(decide):
    with pytest.raises(ValueError, match="XDIT_TP_IPC_WORKSPACE"):
        decide("yes", (6, 16, 6))


def test_amdgpu_version_parses_the_module_version(monkeypatch, tmp_path):
    path = tmp_path / "version"
    path.write_text("7.1.3.31500000\n")
    real_open = open
    monkeypatch.setattr(
        "builtins.open", lambda p, *a, **k: real_open(path if p == "/sys/module/amdgpu/version" else p, *a, **k)
    )
    assert ipc_workspace.amdgpu_version() == (7, 1, 3)
    path.write_text("")
    assert ipc_workspace.amdgpu_version() is None


@pytest.mark.parametrize(
    "abi, inbox_mode, replaced",
    [(2, None, True), (3, "ipc", False), (3, "symm_mem", True)],
)
def test_install_torch_sdma_inbox_defers_to_the_patch_inbox(monkeypatch, abi, inbox_mode, replaced):
    import sys
    import types

    import torch.distributed._symmetric_memory as symm_mem

    def own_inbox(*args):
        raise AssertionError("not called")

    fake = types.ModuleType("torch.distributed._symmetric_memory._rocm_sdma")
    fake.ABI_VERSION = abi
    fake._inbox = own_inbox
    fake._inbox_mode = lambda: inbox_mode
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(symm_mem, "_rocm_sdma", fake, raising=False)
    monkeypatch.setattr(ipc_workspace, "enabled", lambda: True)

    assert ipc_workspace.install_torch_sdma_inbox() is True
    assert (fake._inbox is ipc_workspace._torch_sdma_inbox) is replaced
