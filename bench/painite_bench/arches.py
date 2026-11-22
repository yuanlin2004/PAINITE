"""Deblur architectures spandrel does not know, imported from pinned upstream clones under
$PAINITE_BENCH_DIR/src/unblur/<repo> (fetch.py --src clones them). Upstream code is not
vendored into the repo: several of these projects publish no license. Only the model that
wins the bake-off gets vendored into the server (see UNBLUR.md).

build(name, weights) -> (nn.Module mapping (B,3,h,w)->(B,3,h,w) in [0,1], pad multiple)
"""
import importlib
import sys
import types
from pathlib import Path

import torch

from .config import DATA_DIR

SRC = DATA_DIR / "src/unblur"

# repo dir -> upstream commit (fetch.py checks these out)
REPOS = {
    "Stripformer-ECCV-2022-": ("https://github.com/pp00704831/Stripformer-ECCV-2022-", "9efba889fe2da349714fece642541129b9c313a4"),
    "UFPDeblur": ("https://github.com/Fangzhenxuan/UFPDeblur", "b43d867b31bc79be08364d90bd7e43454603fa89"),
    "MLWNet": ("https://github.com/thqiu0419/MLWNet", "523d164b21b88004962a770c11da9b59da996638"),
    "EVSSM": ("https://github.com/kkkls/EVSSM", "5098a5276640694a39a941119f9e8bcc3ece9fb0"),
    "LoFormer": ("https://github.com/INVOKERer/LoFormer", "5c2041591ececb5fcc7df51b2d6ab1ae4bf3f50c"),
    "LaKDNet": ("https://github.com/lingyanruan/LaKDNet", "0c6b262cd84bfaa407d79e085010fa90ea7446e6"),
    "DRBNet": ("https://github.com/lingyanruan/DRBNet", "0fa89a56381130637ca44bf10b023a2d66a71d87"),
}


def _fresh(repo: str):
    """Make `repo` importable as top-level packages without running basicsr/__init__.py
    (which pulls in training code). Clears any other repo's modules of the same names."""
    for m in [m for m in sys.modules if m.split(".")[0] in ("basicsr", "models", "mamba_ssm")]:
        del sys.modules[m]
    root = SRC / repo
    for p in [p for p in sys.path if p.startswith(str(SRC))]:
        sys.path.remove(p)
    sys.path.insert(0, str(root))
    for pkg in ("basicsr", "basicsr.models", "basicsr.models.archs", "basicsr.utils", "models"):
        d = root / pkg.replace(".", "/")
        if d.is_dir():
            m = types.ModuleType(pkg)
            m.__path__ = [str(d)]
            sys.modules[pkg] = m
    # arch_util imports logging helpers from basicsr.utils; give it plain ones
    import logging
    if "basicsr.utils" in sys.modules:
        sys.modules["basicsr.utils"].get_root_logger = lambda *a, **k: logging.getLogger("basicsr")


def _state(path: Path, *keys):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for k in keys:
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
    return {k.removeprefix("module."): v for k, v in sd.items()}


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """Pure-PyTorch selective scan (port of mamba_ssm.ops.selective_scan_interface.
    selective_scan_ref, Apache-2.0), for when the CUDA extension is not built.
    u, delta: (b,d,l); A: (d,n); B, C: (b,g,n,l) or (b,n,l); D: (d,)."""
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
    b, d, l = u.shape
    n = A.shape[1]
    B, C = B.float(), C.float()
    if B.dim() == 4:  # grouped: (b,g,n,l) -> (b,d,n,l)
        B = B.repeat_interleave(d // B.shape[1], dim=1)
        C = C.repeat_interleave(d // C.shape[1], dim=1)
    else:
        B, C = B[:, None], C[:, None]
    dA = torch.exp(delta[:, :, None, :] * A.float()[None, :, :, None])  # b d n l
    dBu = delta[:, :, None, :] * B * u[:, :, None, :]
    x = u.new_zeros(b, d, n)
    ys = []
    # chunked scan: unroll in blocks to limit Python overhead
    for i in range(l):
        x = dA[..., i] * x + dBu[..., i]
        ys.append((x * C[..., i]).sum(-1))
    y = torch.stack(ys, 2)
    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * torch.nn.functional.silu(z.float())
    y = y.to(dtype_in)
    return (y, x) if return_last_state else y


def _stub_mamba():
    import importlib.util
    spec = importlib.util.find_spec("mamba_ssm")
    if spec and spec.submodule_search_locations:  # the real CUDA kernel, if installed
        # skip mamba_ssm/__init__.py: it imports the LM code (transformers)
        root = spec.submodule_search_locations[0]
        for name, sub in (("mamba_ssm", ""), ("mamba_ssm.ops", "/ops")):
            m = types.ModuleType(name)
            m.__path__ = [root + sub]
            sys.modules[name] = m
        try:
            importlib.import_module("mamba_ssm.ops.selective_scan_interface")
            return
        except ImportError:
            pass
    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.selective_scan_interface"):
        sys.modules[name] = types.ModuleType(name)
    m = sys.modules["mamba_ssm.ops.selective_scan_interface"]
    m.selective_scan_fn = m.selective_scan_ref = selective_scan_ref


def tlc(model: torch.nn.Module, train_size: int = 256) -> torch.nn.Module:
    """Test-time Local Converter (Chu et al., ECCV 2022; NAFNet's `Local_Base`): replace
    global average pools (NAFNet's simplified channel attention) with local pools of
    1.5x the training crop, so statistics at inference match training. Uses the upstream
    local_arch.py (from the MLWNet clone, same code as NAFNet's)."""
    _fresh("MLWNet")
    la = importlib.import_module("basicsr.models.archs.local_arch")
    b = int(train_size * 1.5)
    la.replace_layers(model, base_size=(b, b), train_size=(1, 3, train_size, train_size), fast_imp=False)
    with torch.no_grad():  # pool kernel sizes are fixed on the first forward, as in Local_Base.convert
        model.eval()(torch.rand(1, 3, train_size, train_size))
    return model


def build(name: str, weights: Path):
    if name == "stripformer":
        _fresh("Stripformer-ECCV-2022-")
        net = importlib.import_module("models.Stripformer").Stripformer()
        net.load_state_dict(_state(weights, "params", "state_dict"))
        return _Centered(net), 8
    if name == "ufpnet":
        _fresh("UFPDeblur")
        mod = importlib.import_module("basicsr.models.archs.UFPNet_code_uncertainty_arch")
        net = mod.UFPNet_code_uncertainty_Local(width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1,
                                               dec_blk_nums=[1, 1, 1, 1])
        net.load_state_dict(_state(weights, "params"))
        return net, 16
    if name.startswith("mlwnet"):
        _fresh("MLWNet")
        mod = importlib.import_module("basicsr.models.archs.MLWNet_arch")
        net = mod.MLWNet_Local(dim=64)
        net.load_state_dict(_state(weights, "params"))
        return net, 16
    if name == "evssm":
        _fresh("EVSSM")
        _stub_mamba()
        net = importlib.import_module("models.EVSSM").EVSSM()
        net.load_state_dict(_state(weights, "params"))
        return net, 8
    if name == "loformer":
        _fresh("LoFormer")
        for stub in ("seaborn", "matplotlib", "matplotlib.pyplot"):  # imported, unused at inference
            sys.modules.setdefault(stub, types.ModuleType(stub))
        mod = importlib.import_module("basicsr.models.archs.LoFormer_arch")
        import yaml
        cfg = yaml.safe_load(open(SRC / "LoFormer/Motion_Deblurring/Options/train_RealBlurJ_LoformerLarge_600k_8gpu.yml"))
        cfg = {k: v for k, v in cfg["network_g"].items() if k != "type"}
        net = mod.LoFormer(**cfg)
        net.load_state_dict(_state(weights, "params"))
        return net, 32
    if name.startswith("lakdnet"):
        _fresh("LaKDNet")
        mod = importlib.import_module("models.LaKDNet")
        dim = 48 if "-l" in name else 32
        k = 7 if "motion" in name else 9  # upstream Test_configs.yml
        net = mod.LaKDNet(dim=dim, num_blocks=[8, 12, 12, 16], mix_kernel_size=[k] * 4)
        net.load_state_dict(_state(weights))
        return net, 8
    if name == "drbnet":
        _fresh("DRBNet")
        net = importlib.import_module("models.DRBNet").DRBNet_single()
        net.load_state_dict(_state(weights))
        return _First(net), 16
    raise ValueError(name)


class _Centered(torch.nn.Module):
    """Stripformer works on [-0.5, 0.5] images."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x - 0.5) + 0.5


class _First(torch.nn.Module):
    """Models that return a tuple/list (multi-scale outputs): keep the full-resolution one."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)
        return y[-1] if isinstance(y, (list, tuple)) else y
