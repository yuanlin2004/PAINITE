"""Device choice (CUDA, Apple MPS, CPU) with a stand-in torch module."""

import sys
import types

import pytest

from painite_server.models import devices


def _fake_torch(cuda: bool, mps: bool):
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda i: "NVIDIA GB10")
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return t


@pytest.mark.parametrize(("cuda", "mps", "want", "got"), [
    (True, False, "auto", "cuda"),
    (False, True, "auto", "mps"),
    (True, True, "auto", "cuda"),
    (False, False, "auto", "cpu"),
    (True, True, "mps", "mps"),
    (False, False, "mps", "cpu"),
    (False, True, "cuda", "cpu"),
    (True, True, "cpu", "cpu"),
])
def test_resolve(monkeypatch, cuda, mps, want, got):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda, mps))
    assert devices.resolve(want) == got


def test_env_and_bad_value(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, True))
    monkeypatch.setenv("PAINITE_DEVICE", "MPS")
    assert devices.resolve() == "mps"
    monkeypatch.setenv("PAINITE_DEVICE", "tpu")
    assert devices.resolve() == "cuda"


def test_status(monkeypatch):
    monkeypatch.delenv("PAINITE_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, False))
    assert devices.status() == {"available": True, "name": "NVIDIA GB10"}
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, True))
    monkeypatch.setattr(devices, "_mac_chip", lambda: "Apple M4 Max")
    assert devices.status() == {"available": True, "name": "Apple M4 Max (MPS)"}
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, False))
    assert devices.status() == {"available": False, "name": None}
