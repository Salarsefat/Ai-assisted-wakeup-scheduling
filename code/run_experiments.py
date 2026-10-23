"""Run all experiments of the paper and store raw results in ../results/*.json.

Usage: python3 run_experiments.py <experiment> [...]
Experiments: val, val_myopic (hyper-parameter selection), main, budget, alpha, headroom, eps,
runtime, diag, sched, robust (paper results).
"""
import json
import math
import os
import platform
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import gen_synth, run_pull, run_dps_best, DEFAULT_FG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
SPD = 1440
EPS = 0.5
RHO0 = 1 / 30
HEAD0 = 3.0                       # channel capacity K = ceil(HEAD * N * rho)
SYN = dict(N=100, days=11, noise=0.02, trend=False)
TRACES = {"trace": ("intel_temp_1min.npy", "intel_mask_1min.npy"),
          "heldout": ("intel_temp_1min_heldout.npy", "intel_mask_1min_heldout.npy")}
SCHED_EXPOS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5]

FG = dict(DEFAULT_FG)
FG_MYO = None
if os.path.exists(os.path.join(RES, "val_best.json")):
    FG.update(json.load(open(os.path.join(RES, "val_best.json")))["fg"])
if os.path.exists(os.path.join(RES, "val_best_myopic.json")):
    FG_MYO = dict(FG, **json.load(open(os.path.join(RES, "val_best_myopic.json")))["fg"])

# label -> (policy, fg overrides); ablations differ from FG-Wake in one component
VARIANTS = {
    "fgw": ("fgw", {}),
    "fgw_shape": ("fgw", {"B": 1 if FG.get("B", 48) > 1 else 48}),   # toggles the time-of-day shape
    "fgw_gauss": ("fgw_gauss", {}),
    "fgw_myopic": ("fgw_myopic", {}),
    "fgw_noproj": ("fgw_noproj", {}),
    "fgw_static": ("fgw_static", {}),
    "fgw_genie": ("fgw_genie", {}),
    "fgw_post": ("fgw", {"shape_u": "post"}),
    "fgw_nocap": ("fgw", {"Qcap": 1e9}),
}

_cache = {}


def get_data(src, seed):
    if src == "synth":
        rng = np.random.default_rng(10_000 + seed)
        x, sig = gen_synth(SYN["N"], SYN["days"] * SPD, rng)
        return x, sig, None, SYN["noise"], SYN["trend"]
    if src not in _cache:
        fx, fm = TRACES[src]
        _cache[src] = (np.load(os.path.join(DATA, fx)), np.load(os.path.join(DATA, fm)))
    x, m = _cache[src]
    return x, None, m, 0.0, True                # trend estimator for the drift-dominated trace


def K_of(N, rho, head=HEAD0):
    return max(1, math.ceil(head * N * rho - 1e-9))


def job(spec):
    src, seed, label, rho, alpha, head, fg, trace, eps = spec
    x, sig, mask, noise, trend = get_data(src, seed)
    N = x.shape[1]
    K = K_of(N, rho, head)
    t0 = time.time()
    if label == "dps":
        r = run_dps_best(x, rho, K, eps, seed, alpha=alpha, noise=noise, warm=SPD,
                         trend=trend, return_trace=trace, mask=mask)
    elif label in ("schedrr", "sched_causal", "sched_best"):
        # hour-of-day schedules; exponent (and, for sched_best, the profile statistic)
        # chosen in hindsight on the evaluation data
        variants = []
        if label in ("schedrr", "sched_best"):
            variants += [("incr", None)]
        if label == "sched_causal":
            variants += [("causal", None)]
        if label == "sched_best":                    # RR's own hourly violation profile
            rr0 = run_pull(x, "rr", rho, K, eps, np.random.default_rng(seed), alpha=alpha, noise=noise,
                           warm=SPD, fg=fg, trend=trend, mask=mask, return_trace=True)
            vs, os_ = rr0["viol_slot"][SPD:], rr0["obs_slot"][SPD:]
            hr = (np.arange(len(vs)) % SPD) // 60
            prof = np.array([vs[hr == h].sum() / max(os_[hr == h].sum(), 1) for h in range(24)]) + 0.01
            variants += [("rrviol", prof)]
        r = None
        for stat, prof in variants:
            for e in SCHED_EXPOS:
                rr_ = run_pull(x, "schedrr", rho, K, eps, np.random.default_rng(seed), alpha=alpha,
                               noise=noise, warm=SPD, fg=dict(fg, sched_expo=e, sched_stat=stat,
                                                              sched_prof=prof),
                               trend=trend, mask=mask, return_trace=trace)
                rr_["sched_expo"] = e
                rr_["sched_stat"] = stat
                if r is None or rr_["viol"] < r["viol"]:
                    r = rr_
    else:
        pol, over = VARIANTS.get(label, (label, {}))
        r = run_pull(x, pol, rho, K, eps, np.random.default_rng(seed), alpha=alpha,
                     noise=noise, warm=SPD, sig=sig, fg=dict(fg, **over), trend=trend,
                     mask=mask, return_trace=trace)
    if trace:                                        # per-day violation ratio (observed minutes)
        vs, os_ = r.pop("viol_slot"), r.pop("obs_slot")
        r.pop("poll_slot", None)
        days = (len(vs) - SPD) // SPD
        r["viol_day"] = [float(vs[SPD + d * SPD: SPD + (d + 1) * SPD].sum()
                               / max(os_[SPD + d * SPD: SPD + (d + 1) * SPD].sum(), 1))
                         for d in range(days)]
    r.update(src=src, seed=seed, pol=label, rho=rho, alpha=alpha, head=head, K=K, N=N,
             eps=eps, secs=time.time() - t0)
    return r


def run_all(specs, name):
    t0 = time.time()
    with Pool(2) as p:
        out = p.map(job, specs, chunksize=1)
    with open(os.path.join(RES, f"{name}.json"), "w") as f:
        json.dump(out, f)
    print(f"[{name}] {len(specs)} runs in {time.time() - t0:.0f}s", flush=True)
    return out


def S(src, seed, label, rho=RHO0, alpha=1.0, head=HEAD0, fg=None, trace=False, eps=EPS):
    """Job spec. Without an explicit fg, the myopic ablation uses its own validated
    parameters and every other policy uses FG-Wake's."""
    if fg is None:
        fg = FG_MYO if (label == "fgw_myopic" and FG_MYO is not None) else FG
    return (src, seed, label, rho, alpha, head, fg, trace, eps)


S_FIXED = 10      # credit/debt window fixed by design (allowed budget deviation, Prop. 2)
GRID_FG = ([dict(S=s_, V=v_, lr=l_, B=b_) for s_ in (5, 10, 20) for v_ in (50, 100, 200, 400)
            for l_ in (0.01, 0.02) for b_ in (1, 48)]
           + [dict(S=s_, V=v_, lr=0.02, B=48) for s_ in (10, 20, 40) for v_ in (800, 1600)]
           + [dict(S=40, V=400, lr=0.02, B=48)])
GRID_MYO = ([dict(S=s_, V=v_, lr=l_) for s_ in (5, 10, 20) for v_ in (10, 25, 50, 100, 200, 400)
             for l_ in (0.01, 0.02)]
            + [dict(S=10, V=v_, lr=0.02) for v_ in (800, 1600)])


def _grid_select(label, grid, name, base):
    """Run the grid on the synthetic validation seeds 100-104 and select, among the
    configurations with S == S_FIXED, the one with the lowest mean violation ratio."""
    specs = [S("synth", seed, label, fg=dict(base, **g)) for g in grid for seed in range(100, 105)]
    out = run_all(specs, name)
    scores = []
    for gi, g in enumerate(grid):
        rs = out[gi * 5:(gi + 1) * 5]
        scores.append([float(np.mean([r["viol"] for r in rs])),
                       float(max(r["poll_rate_max"] for r in rs) / RHO0), g])
    scores.sort(key=lambda s_: s_[0])
    best = [sc for sc in scores if sc[2]["S"] == S_FIXED][0]
    print("selected", best)
    return best, scores


def exp_val():
    best, scores = _grid_select("fgw", GRID_FG, "val", dict(DEFAULT_FG))
    json.dump({"rule": f"S fixed to {S_FIXED} polls by design; V, eta, B selected by the lowest mean "
                       "violation ratio on synthetic validation seeds 100-104",
               "fg": best[2], "grid": scores}, open(os.path.join(RES, "val_best.json"), "w"), indent=1)


def exp_val_myopic():
    base = dict(DEFAULT_FG, **json.load(open(os.path.join(RES, "val_best.json")))["fg"])
    best, scores = _grid_select("fgw_myopic", GRID_MYO, "val_myopic", base)
    json.dump({"rule": f"same as FG-Wake (S fixed to {S_FIXED}; V, eta selected on validation seeds)",
               "fg": best[2], "grid": scores}, open(os.path.join(RES, "val_best_myopic.json"), "w"), indent=1)


MAIN_POLS = ["rr", "sched_causal", "sched_best", "rnd", "maxvar", "dps", "fgw", "fgw_shape",
             "fgw_gauss", "fgw_myopic", "fgw_noproj", "fgw_static", "fgw_nocap", "oracle"]


def exp_main():
    specs = [S("synth", s_, p) for s_ in range(20) for p in MAIN_POLS + ["fgw_genie"]]
    specs += [S(src, 0, p, trace=True) for src in TRACES for p in MAIN_POLS]
    run_all(specs, "main")


def exp_budget():
    rhos = [1 / 10, 1 / 15, 1 / 20, 1 / 30, 1 / 45, 1 / 60, 1 / 90, 1 / 120]
    pols = ["rr", "sched_best", "sched_causal", "fgw", "oracle"]
    specs = [S("synth", s_, p, rho=r) for s_ in range(10) for p in pols for r in rhos]
    specs += [S("trace", 0, p, rho=r) for p in pols for r in rhos]
    run_all(specs, "budget")


def exp_alpha():
    Bud = RHO0 * 2.0                              # energy budget per slot (alpha = 1 reference)
    specs = []
    for a in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        rho = Bud / (1 + a)
        for p in ["rr", "fgw", "dps"]:
            specs += [S("synth", s_, p, rho=rho, alpha=a) for s_ in range(3)]
            specs.append(S("trace", 0, p, rho=rho, alpha=a))
    run_all(specs, "alpha")


def exp_headroom():
    specs = [S("synth", s_, p, head=h) for s_ in range(10) for p in ["rr", "fgw", "oracle"]
             for h in [1.0, 1.5, 2.0, 3.0, 5.0]]
    specs += [S("trace", 0, p, head=h) for p in ["rr", "fgw", "oracle"] for h in [1.0, 1.5, 2.0, 3.0, 5.0]]
    run_all(specs, "headroom")


def exp_eps():
    specs = []
    for e in [0.25, 0.5, 1.0]:
        for p in ["rr", "fgw"]:
            specs += [S(src, 0, p, eps=e) for src in TRACES]
            specs += [S("synth", s_, p, eps=e) for s_ in range(5)]
    run_all(specs, "eps")


def exp_robust():
    """FG-Wake with the five best validated configurations at S=10 on both trace windows."""
    val = json.load(open(os.path.join(RES, "val_best.json")))
    top = [g for _, _, g in sorted(val["grid"], key=lambda r: r[0]) if g["S"] == 10][:5]
    out = []
    for g in top:
        for src in TRACES:
            x, _, m, _, _ = get_data(src, 0)
            K = K_of(x.shape[1], RHO0)
            fgw = run_pull(x, "fgw", RHO0, K, EPS, np.random.default_rng(0), fg=dict(FG, **g), trend=True, mask=m)
            out.append(dict(src=src, cfg=g, viol=fgw["viol"]))
    json.dump(out, open(os.path.join(RES, "robust.json"), "w"), indent=1)
    print(out)


def exp_sched():
    """Sched-RR for every exponent (not only the best), at the default budget."""
    out = []
    for src in ("trace", "heldout", "synth"):
        for seed in ([0] if src != "synth" else range(5)):
            x, sig, mask, noise, trend = get_data(src, seed)
            K = K_of(x.shape[1], RHO0)
            rr0 = run_pull(x, "rr", RHO0, K, EPS, np.random.default_rng(seed), noise=noise, warm=SPD,
                           fg=FG, trend=trend, mask=mask, return_trace=True)
            vs, os_ = rr0["viol_slot"][SPD:], rr0["obs_slot"][SPD:]
            hr = (np.arange(len(vs)) % SPD) // 60
            prof = np.array([vs[hr == h].sum() / max(os_[hr == h].sum(), 1) for h in range(24)]) + 0.01
            for stat, pf in (("incr", None), ("causal", None), ("rrviol", prof)):
                for e in SCHED_EXPOS:
                    r = run_pull(x, "schedrr", RHO0, K, EPS, np.random.default_rng(seed), noise=noise,
                                 warm=SPD, fg=dict(FG, sched_expo=e, sched_stat=stat, sched_prof=pf),
                                 trend=trend, mask=mask)
                    out.append(dict(src=src, seed=seed, stat=stat, expo=e, viol=r["viol"],
                                    rate=r["poll_rate"] / RHO0))
    json.dump(out, open(os.path.join(RES, "sched.json"), "w"), indent=1)
    for src in ("trace", "heldout", "synth"):
        for stat in ("incr", "causal", "rrviol"):
            print(src, stat, {e: round(100 * float(np.mean([o["viol"] for o in out if o["src"] == src
                                                           and o["stat"] == stat and o["expo"] == e])), 2)
                              for e in SCHED_EXPOS})


def exp_runtime():
    out = []
    for N in (28, 100, 1000):
        rng = np.random.default_rng(7)
        x, sig = gen_synth(N, 3 * SPD, rng)
        for pol in ("rr", "fgw"):
            K = K_of(N, RHO0)
            r = run_pull(x, pol, RHO0, K, EPS, np.random.default_rng(1), noise=0.02,
                         warm=SPD, fg=FG, record_time=True)
            out.append(dict(N=N, K=K, pol=pol, us=r["slot_us"]))
            print(out[-1], flush=True)
    meta = dict(python=platform.python_version(), numpy=np.__version__,
                cpu=platform.processor() or platform.machine(), cores=os.cpu_count())
    json.dump(dict(rows=out, meta=meta), open(os.path.join(RES, "runtime.json"), "w"), indent=1)


def exp_diag():
    """Hour-of-day allocation, estimator effect at the lowest budget, calibration at poll
    times, and sensitivity to the shape-update variant."""
    x, _, m, _, _ = get_data("trace", 0)
    N = x.shape[1]
    out = {"hourly": {}, "lowbudget": {}, "calib": {}, "shape_variant": {}}
    for pol in ("rr", "fgw"):
        r = run_pull(x, pol, RHO0, K_of(N, RHO0), EPS, np.random.default_rng(0), noise=0.0,
                     warm=SPD, fg=FG, trend=True, mask=m, return_trace=True)
        p, v, o = r["poll_slot"][SPD:], r["viol_slot"][SPD:], r["obs_slot"][SPD:]
        hr = (np.arange(len(p)) % SPD) // 60
        ndays = len(p) / SPD
        out["hourly"][pol] = dict(
            polls_per_sensor_hour=[float(p[hr == h].sum() / ndays / N) for h in range(24)],
            viol=[float(v[hr == h].sum() / max(o[hr == h].sum(), 1)) for h in range(24)],
            viol_count=[float(v[hr == h].sum()) for h in range(24)],
            night_share=float(p[(hr >= 20) | (hr < 7)].sum() / p.sum()))
    rho = 1 / 120
    for pol in ("rr", "fgw"):
        for trend in (True, False):
            r = run_pull(x, pol, rho, K_of(N, rho), EPS, np.random.default_rng(0), noise=0.0,
                         warm=SPD, fg=FG, trend=trend, mask=m)
            out["lowbudget"][f"{pol}_{'trend' if trend else 'zoh'}"] = dict(
                viol=r["viol"], rate=r["poll_rate"] / rho)
    for pol in ("fgw", "fgw_gauss"):
        r = run_pull(x, pol, RHO0, K_of(N, RHO0), EPS, np.random.default_rng(0), noise=0.0,
                     warm=SPD, fg=dict(FG, cal_pairs=True), trend=True, mask=m)
        pr = np.array([p for p, _ in r["cal_pairs"]]); lb = np.array([l_ for _, l_ in r["cal_pairs"]])
        bins = np.minimum((pr * 10).astype(int), 9)
        ece = float(sum(abs(pr[bins == k].mean() - lb[bins == k].mean()) * (bins == k).mean()
                        for k in range(10) if np.any(bins == k)))
        out["calib"][pol] = dict(pred=r["cal_pred"], obs=r["cal_obs"], ece=ece, n=int(len(pr)))
    for src in ("trace", "heldout"):
        xx, _, mm, _, _ = get_data(src, 0)
        for var in ("pre", "post"):
            r = run_pull(xx, "fgw", RHO0, K_of(xx.shape[1], RHO0), EPS, np.random.default_rng(0),
                         noise=0.0, warm=SPD, fg=dict(FG, shape_u=var), trend=True, mask=mm)
            out["shape_variant"][f"{src}_{var}"] = r["viol"]
    vals = []
    for var in ("pre", "post"):
        v = []
        for seed in range(100, 105):
            xs, sg, _, nz, _ = get_data("synth", seed)
            v.append(run_pull(xs, "fgw", RHO0, K_of(100, RHO0), EPS, np.random.default_rng(seed),
                              noise=nz, warm=SPD, sig=sg, fg=dict(FG, shape_u=var))["viol"])
        out["shape_variant"][f"val_{var}"] = float(np.mean(v))
    json.dump(out, open(os.path.join(RES, "diag.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("lowbudget", "calib", "shape_variant")}, indent=1))


if __name__ == "__main__":
    for e in sys.argv[1:]:
        globals()[f"exp_{e}"]()
