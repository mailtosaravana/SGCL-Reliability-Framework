"""Fast unit checks of the fault model and statistics (no training). Run: python tests_smoke.py"""
import math, torch, numpy as np
from sgcl.common import flip_bits_, sample_bit_positions, FaultInjector, build_model, DEVICE, BIT_DISTS
from sgcl.stats import paired_z

def test_flip_is_single_bit():
    g = torch.Generator(device=DEVICE).manual_seed(0)
    for dtype, width in ((torch.float32, 32), (torch.float16, 16), (torch.float8_e4m3fn, 8)):
        t = torch.ones(100_000, dtype=dtype, device=DEVICE); ref = t.clone(); k = flip_bits_(t, 1e-3, g)
        itype = {32: torch.int32, 16: torch.int16, 8: torch.int8}[width]
        x = (t.view(itype) ^ ref.view(itype))
        nz = x[x != 0]; assert nz.numel() == k, (nz.numel(), k)
        assert all(bin(int(v) & ((1 << width) - 1)).count("1") == 1 for v in nz.tolist()), "each corrupted element differs in exactly one bit"
    print("flip_is_single_bit ok")

def test_bit_distribution():
    g = torch.Generator(device=DEVICE).manual_seed(1); n = 200_000
    for dist, (ps, pe, pm) in BIT_DISTS.items():
        b = sample_bit_positions(n, g, DEVICE, dist, 32).cpu().numpy()
        fs, fe = (b == 31).mean(), ((b >= 23) & (b <= 30)).mean()
        assert abs(fs - ps) < 0.01 and abs(fe - pe) < 0.01, (dist, fs, fe)
    print("bit_distribution ok")

def test_paired_design_is_deterministic():
    m = build_model("resnet18", 10).to(DEVICE).eval(); x = torch.randn(4, 3, 32, 32, device=DEVICE)
    with FaultInjector(m, 1e-4, 7) as fi: o1 = m(x).clone(); k1 = fi.n_flips
    with FaultInjector(m, 1e-4, 7) as fi: o2 = m(x).clone(); k2 = fi.n_flips
    assert k1 == k2 and torch.allclose(o1, o2, equal_nan=True), "same seed must inject identical faults"
    with FaultInjector(m, 1e-4, 8) as fi: o3 = m(x).clone(); k3 = fi.n_flips
    assert k3 != k1 or not torch.allclose(o1, o3, equal_nan=True); print("paired_design ok, flips per trial:", k1)

def test_weights_restored():
    m = build_model("resnet18", 10).to(DEVICE).eval(); w = m.layer4[1].conv2.weight.data.clone()
    with FaultInjector(m, 1e-3, 3): pass
    assert torch.equal(w, m.layer4[1].conv2.weight.data); print("weights_restored ok")

def test_paired_z():
    rng = np.random.default_rng(0); a = rng.normal(0.03, 0.004, 500); b = a - rng.normal(0.01, 0.003, 500)
    r = paired_z(a, b); assert r["p_adj"] < 1e-10 and 2 < r["delta_cer_pp"] / r["se_pp"]; print("paired_z ok", {k: r[k] for k in ("delta_cer_pp", "z", "p_adj", "rif")})

if __name__ == "__main__":
    test_flip_is_single_bit(); test_bit_distribution(); test_paired_design_is_deterministic(); test_weights_restored(); test_paired_z(); print("all tests passed")
