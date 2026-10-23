"""Turn ../results/*.json into numbers.tex, table_*.tex and figures in ../tex/.
Every number quoted in the paper is produced here from the raw result files."""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiments as RX  # noqa: E402
import sim as SIM  # noqa: E402

RES = os.path.join(HERE, "..", "results")
TEX = os.path.join(HERE, "..", "tex")
DATA = os.path.join(HERE, "..", "data")
RHO0 = RX.RHO0
macros = {}


def M(name, val):
    macros[name] = val


def fmt(x, d=1):
    return f"{x:.{d}f}"


def ci(a, conf=0.95):
    a = np.asarray(a, float)
    if len(a) < 2:
        return float(a.mean()), 0.0
    h = stats.t.ppf(0.5 + conf / 2, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean()), float(h)


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def group(rows, *keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def red(rr, fg):
    return 100 * (rr - fg) / rr


# ------------------------------------------------------------------ data facts
s = json.load(open(os.path.join(DATA, "intel_summary.json")))
h = json.load(open(os.path.join(DATA, "intel_summary_heldout.json")))
c3 = lambda n: f"{n:,}".replace(",", "{,}")  # noqa: E731
M("IntelRows", c3(s["rows_in_file"]))
M("IntelParsed", c3(s["rows_parsed_mote_1_54"]))
M("IntelClean", c3(s["rows_after_range_and_voltage_filter"]))
M("IntelMotes", str(s["n_motes"]))
M("IntelObs", fmt(100 * s["fraction_minutes_observed_before_interp"]))
M("IntelStart", s["window_start"][:10])
M("HoMotes", str(h["n_motes"]))
M("HoObs", fmt(100 * h["fraction_minutes_observed_before_interp"]))
M("HoStart", h["window_start"][:10])
M("HoShared", str(len(set(s["motes_kept"]) & set(h["motes_kept"]))))

# variance-scaling exponents of the main window: intervals lying entirely in the
# night (20:00-07:00) or day (08:00-19:00) period, observed minutes at both ends
x = np.load(os.path.join(DATA, "intel_temp_1min.npy"))[1440:]
mk = np.load(os.path.join(DATA, "intel_mask_1min.npy"))[1440:]
T = x.shape[0]
tod = (np.arange(T) % 1440) // 60
for nm, per in [("Night", (tod >= 20) | (tod < 7)), ("Day", (tod >= 8) & (tod < 19))]:
    cs = np.concatenate([[0], np.cumsum(per)])
    hs = np.array([1, 2, 4, 8, 16, 32, 64])
    vs = []
    for hh in hs:
        idx = np.arange(T - hh)
        inside = (cs[idx + hh + 1] - cs[idx]) == hh + 1
        dd = (x[idx + hh] - x[idx])
        ok = inside[:, None] & mk[idx] & mk[idx + hh]
        vs.append(dd[ok].var())
    M(f"IntelExp{nm}", fmt(np.polyfit(np.log(hs), np.log(vs), 1)[0], 2))

val = json.load(open(os.path.join(RES, "val_best.json")))
FGsel = dict(SIM.DEFAULT_FG, **val["fg"])
M("ValS", fmt(FGsel["S"], 0)); M("ValV", fmt(FGsel["V"], 0)); M("ValLr", f"{FGsel['lr']:g}")
M("ValB", str(int(FGsel["B"])))
M("ShapeOn", "yes" if FGsel["B"] > 1 else "no")
M("BoundWhole", fmt(FGsel["S"] + 1, 0))
M("BoundWin", fmt(2 * FGsel["S"] + 1, 0))
vm = load("val_best_myopic")
if vm:
    M("MyoS", fmt(vm["fg"]["S"], 0)); M("MyoV", fmt(vm["fg"]["V"], 0)); M("MyoLr", f"{vm['fg']['lr']:g}")

# ------------------------------------------------------------------ main table
main = load("main")
g = group(main, "src", "pol")


def mean(src, pol, key="viol"):
    return float(np.mean([r[key] for r in g[(src, pol)]]))


for src, tag in [("trace", "Tr"), ("heldout", "Ho"), ("synth", "Sy")]:
    rr, fgw = mean(src, "rr"), mean(src, "fgw")
    M(f"{tag}RR", fmt(100 * rr, 2)); M(f"{tag}FG", fmt(100 * fgw, 2))
    for pol, nm in [("maxvar", "MV"), ("oracle", "OR"), ("fgw_gauss", "Gauss"), ("fgw_shape", "Shape"),
                    ("fgw_static", "Static"), ("dps", "DPS"),
                    ("fgw_myopic", "Myo"), ("fgw_noproj", "NoProj"), ("fgw_nocap", "NoCap"), ("rnd", "RND")]:
        if (src, pol) in g:
            M(f"{tag}{nm}", fmt(100 * mean(src, pol), 2))
    M(f"{tag}RateMax", fmt(max(r["poll_rate_max"] for r in g[(src, "fgw")]) / RHO0, 3))
    M(f"{tag}RateMean", fmt(mean(src, "fgw", "poll_rate") / RHO0, 3))
    M(f"{tag}AoiRR", fmt(mean(src, "rr", "aoi"))); M(f"{tag}AoiFG", fmt(mean(src, "fgw", "aoi")))
    for pol, nm in [("sched_best", "SchedB"), ("sched_causal", "SchedC")]:
        M(f"{tag}{nm}", fmt(100 * mean(src, pol), 2))
        M(f"{tag}{nm}Red", fmt(red(rr, mean(src, pol)), 0))
    M(f"{tag}FailRR", fmt(100 * mean(src, "rr", "failed_frac"), 0)) if src != "synth" else None
    M(f"{tag}FailFG", fmt(100 * mean(src, "fgw", "failed_frac"), 0)) if src != "synth" else None
    M(f"{tag}DpsE", fmt(mean(src, "dps", "energy") / (2 * RHO0), 2))
    M(f"{tag}NoCapRate", fmt(max(r["poll_rate_max"] for r in g[(src, "fgw_nocap")]) / RHO0, 3))
    M(f"{tag}NoCapRed", fmt(red(rr, mean(src, "fgw_nocap")), 0))
    if src != "synth":
        M(f"{tag}NoCapFail", fmt(100 * mean(src, "fgw_nocap", "failed_frac"), 0))
    M(f"{tag}MyoRed", fmt(red(rr, mean(src, "fgw_myopic")), 0))
    if src != "synth":
        M(f"{tag}Red", fmt(red(rr, fgw), 0))
M("TrMaeRed", fmt(red(mean("trace", "rr", "mae"), mean("trace", "fgw", "mae")), 0))
M("TrMaeRR", fmt(mean("trace", "rr", "mae"), 3)); M("TrMaeFG", fmt(mean("trace", "fgw", "mae"), 3))
M("TrMaePct", fmt(100 * (mean("trace", "fgw", "mae") / mean("trace", "rr", "mae") - 1), 0))
M("TrStaticDelta", fmt(100 * (mean("trace", "fgw_static") - mean("trace", "fgw")), 1))
_all = []
for _n in ("main", "budget", "alpha", "headroom", "eps"):
    _d = load(_n)
    if _d:
        _all += _d
M("OverMax", fmt(max(r["overshoot_max_polls"] for r in _all if r["pol"] == "fgw"), 0))
M("OverMaxAbl", fmt(max(r.get("overshoot_max_polls", -1e9) for r in _all
                        if r["pol"].startswith("fgw") and r["pol"] != "fgw_nocap"), 0))
M("QrunMax", fmt(max(r["Q_runmax"] for r in main if r["pol"] == "fgw"), 1))
for src, tag in [("trace", "Tr"), ("heldout", "Ho"), ("synth", "Sy")]:
    r0 = g[(src, "fgw")][0]
    M(f"{tag}Head", fmt(r0["K"] / (r0["N"] * RHO0), 2))
    M(f"{tag}K", str(r0["K"]))

# per-day paired comparison on the trace windows (descriptive)
for src, tag in [("trace", "Tr"), ("heldout", "Ho")]:
    d_rr = np.array(g[(src, "rr")][0]["viol_day"]); d_fg = np.array(g[(src, "fgw")][0]["viol_day"])
    diff = 100 * (d_rr - d_fg)
    m_, hw = ci(diff)
    M(f"{tag}Days", str(int((d_fg < d_rr).sum()))); M(f"{tag}Ndays", str(len(diff)))
    M(f"{tag}Diff", fmt(m_, 2)); M(f"{tag}DiffCI", f"[{fmt(m_ - hw, 2)}, {fmt(m_ + hw, 2)}]")

# synthetic: paired per-seed relative reductions (mean and CI)
by_seed = {(r["pol"], r["seed"]): r["viol"] for r in main if r["src"] == "synth"}
seeds = sorted({r["seed"] for r in main if r["src"] == "synth"})
for pol, nm in [("fgw", "Red"), ("fgw_genie", "GenieRed"), ("sched_best", "SchedBRedS"), ("dps", "DpsRed")]:
    rs_ = [red(by_seed[("rr", s_)], by_seed[(pol, s_)]) for s_ in seeds]
    m_, hw = ci(rs_)
    M(f"Sy{nm}", fmt(m_, 0)); M(f"Sy{nm}CI", f"[{fmt(m_ - hw, 1)}, {fmt(m_ + hw, 1)}]")
M("SyNseeds", str(len(seeds)))
dg_ = [100 * (by_seed[("fgw_gauss", s_)] - by_seed[("fgw", s_)]) for s_ in seeds]
m_, hw = ci(dg_)
M("SyGaussDiff", fmt(m_, 2)); M("SyGaussDiffCI", fmt(hw, 2))

shape_label = ("\\quad with time-of-day shape" if FGsel["B"] == 1 else "\\quad without time-of-day shape")
rows = [("RR (periodic)", "rr"), ("Schedule (warm-up day)", "sched_causal"), ("Schedule (hindsight)", "sched_best"),
        ("RND", "rnd"), ("MaxVar", "maxvar"),
        ("DPS (push, hindsight)", "dps"), ("\\textbf{FG-Wake}", "fgw"),
        (shape_label, "fgw_shape"), ("\\quad without calibration (Gaussian)", "fgw_gauss"),
        ("\\quad without renewal index (myopic)", "fgw_myopic"), ("\\quad without projection", "fgw_noproj"),
        ("\\quad frozen after warm-up", "fgw_static"), ("\\quad without debt cap", "fgw_nocap"),
        ("\\quad genie variance", "fgw_genie"), ("Oracle (clairvoyant)", "oracle")]


def erel(src, pol):
    if pol == "dps":
        return mean(src, pol, "energy") / (2 * RHO0)
    return mean(src, pol, "poll_rate") / RHO0


lines = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Violation ratio $\bar v$ (\%), MAE, mean AoI and realised energy relative to the budget "
         r"($E/B$) at $\rho=1/30$, $\alpha=1$, $K=\lceil3N\rho\rceil$. Trace metrics use observed minutes only. "
         r"Synthetic: mean $\pm$ 95\% CI over " + macros["SyNseeds"] + r" seeds. Hindsight baselines tune "
         r"their parameters on the evaluation data itself.}",
         r"\label{tab:main}", r"\setlength{\tabcolsep}{5pt}", r"\small",
         r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Intel main (" + macros["IntelMotes"] + r" motes)} & Held-out ("
         + macros["HoMotes"] + r") & \multicolumn{3}{c}{Synthetic ($N{=}100$)}\\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(l){7-9}",
         r"Policy & $\bar v$ & MAE [$^\circ$C] & AoI [min] & $E/B$ & $\bar v$ & $\bar v$ & MAE & $E/B$\\",
         r"\midrule"]
for label, pol in rows:
    cells = []
    if ("trace", pol) in g:
        cells += [fmt(100 * mean("trace", pol), 2), fmt(mean("trace", pol, "mae"), 3),
                  fmt(mean("trace", pol, "aoi"), 1), fmt(erel("trace", pol), 2)]
    else:
        cells += ["--"] * 4
    cells.append(fmt(100 * mean("heldout", pol), 2) if ("heldout", pol) in g else "--")
    if ("synth", pol) in g:
        mm, hh = ci([100 * r["viol"] for r in g[("synth", pol)]])
        cells += [f"{mm:.2f}$\\pm${hh:.2f}", fmt(mean("synth", pol, "mae"), 3), fmt(erel("synth", pol), 2)]
    else:
        cells += ["--"] * 3
    if pol == "fgw":
        cells = [f"\\textbf{{{c}}}" if i in (0, 4, 5) else c for i, c in enumerate(cells)]
    lines.append(label + " & " + " & ".join(cells) + r"\\")
    if pol in ("dps", "fgw_genie"):
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{-3mm}", r"\end{table*}"]
open(os.path.join(TEX, "table_main.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ parameter table (from the code)
ptab = [r"\begin{table}[t]", r"\centering",
        r"\caption{Simulation parameters (defaults; sweeps in Section~\ref{sec:results}).}",
        r"\label{tab:params}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l@{\hspace{4pt}}>{\raggedright\arraybackslash}p{0.60\columnwidth}@{}}", r"\toprule",
        r"Slot; horizon & 1 min; trace: 10 days (1 warm-up + 9); synthetic: 11 days (1 + 10)\\",
        f"Budget $\\rho$; capacity $K$ & 1/{int(round(1 / RHO0))} polls/slot; $\\lceil {RX.HEAD0:g}N\\rho\\rceil$\\\\",
        f"Tolerance $\\epsilon$; sensing cost $\\alpha$ & {RX.EPS:g} ($^\\circ$C on the trace); 1\\\\",
        r"Trace & " + macros["IntelMotes"] + r" motes ($K{=}" + macros["TrK"] + r"$); held-out "
        + macros["HoMotes"] + r" motes ($K{=}" + macros["HoK"] + r"$)\\",
        f"Synthetic & $N{{=}}{RX.SYN['N']}$, $\\sigma_0{{=}}0.01$, $\\sigma_n{{=}}{RX.SYN['noise']:g}$, "
        f"{macros['SyNseeds']} seeds (sweeps: 3--10)\\\\",
        f"FG-Wake forecaster & $B{{=}}{int(FGsel['B'])}$, $\\beta{{=}}{FGsel['beta']:g}$, "
        f"$\\omega_\\uparrow{{=}}{FGsel['a_up']:g}$, $\\omega_\\downarrow{{=}}{FGsel['a_dn']:g}$\\\\",
        f"Index, price, learning & $\\bar w{{=}}{FGsel['Wcap']:g}$, $S{{=}}{FGsel['S']:g}$, "
        f"$V{{=}}{FGsel['V']:g}$, $\\eta{{=}}{FGsel['lr']:g}$ (validated)\\\\",
        r"Trend smoothing $\gamma$ & 0.3 (trace, all policies)\\",
        f"DPS grid & $T_s\\in\\{{1,2,\\dots\\}}$ up to $1/\\rho$, including $\\lfloor1/\\rho\\rfloor,\\lceil1/\\rho\\rceil$; "
        f"$\\kappa\\in\\{{{','.join(f'{k:g}' for k in SIM.KAPPA_GRID)}\\}}$, "
        f"$S'\\in\\{{{','.join(f'{k:g}' for k in SIM.S_GRID)}\\}}$\\\\",
        f"Schedules & rate$_h\\propto p_h^{{\\,q}}$, $q\\in\\{{{','.join(f'{k:g}' for k in RX.SCHED_EXPOS)}\\}}$\\\\",
        r"\bottomrule", r"\end{tabular}", r"\vspace{-3mm}", r"\end{table}"]
open(os.path.join(TEX, "table_params.tex"), "w").write("\n".join(ptab) + "\n")

# ------------------------------------------------------------------ schedules per exponent / statistic
sch = load("sched")
if sch:
    for src, tag in [("trace", "Tr"), ("heldout", "Ho"), ("synth", "Sy")]:
        for stat, nm in [("incr", "Incr"), ("causal", "Causal"), ("rrviol", "Rrv")]:
            vals = {e: np.mean([o["viol"] for o in sch if o["src"] == src and o["stat"] == stat and o["expo"] == e])
                    for e in RX.SCHED_EXPOS}
            eb = min(vals, key=vals.get)
            M(f"Sch{tag}{nm}", fmt(100 * vals[eb], 2)); M(f"Sch{tag}{nm}Expo", f"{eb:g}")
for src, tag in [("trace", "Tr"), ("heldout", "Ho"), ("synth", "Sy")]:
    rb = g[(src, "sched_best")][0]
    M(f"{tag}SchedBStat", {"incr": "change spread", "rrviol": "RR violation profile"}[rb["sched_stat"]])
    M(f"{tag}SchedBExpo", f"{rb['sched_expo']:g}")
    M(f"{tag}SchedCExpo", f"{g[(src, 'sched_causal')][0]['sched_expo']:g}")
    M(f"{tag}FgVsSchedB", fmt(100 * (mean(src, "fgw") - mean(src, "sched_best")) / mean(src, "sched_best"), 0))
M("TrDpsRed", fmt(red(mean("trace", "rr"), mean("trace", "dps")), 0))
M("HoDpsRed", fmt(red(mean("heldout", "rr"), mean("heldout", "dps")), 0))
M("SyDpsE", fmt(mean("synth", "dps", "energy") / (2 * RHO0), 2))
M("TrAoiPct", fmt(100 * (mean("trace", "fgw", "aoi") / mean("trace", "rr", "aoi") - 1), 0))

# robustness over the five best validated configurations
rob = load("robust")
if rob:
    for src, tag in [("trace", "Tr"), ("heldout", "Ho")]:
        rr = mean(src, "rr")
        reds = [red(rr, o["viol"]) for o in rob if o["src"] == src]
        M(f"Rob{tag}Min", fmt(min(reds), 0)); M(f"Rob{tag}Max", fmt(max(reds), 0))

# ------------------------------------------------------------------ plotting style
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.3,
                     "lines.markersize": 3.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
STY = {"rr": ("RR", "#4c78a8", "s", "-"), "sched_best": ("Schedule (hindsight)", "#9d9d9d", "^", "--"),
       "sched_causal": ("Schedule (warm-up)", "#bcbd22", "x", "--"),
       "fgw": ("FG-Wake", "#d62728", "o", "-"), "oracle": ("Oracle", "#2ca02c", "v", ":"),
       "dps": ("DPS (push)", "#9467bd", "D", "-.")}


def tidy_log_axes(ax, xt):
    ax.set_xticks(xt)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------ budget sweep
bud = load("budget")
if bud:
    gb = group(bud, "src", "pol", "rho")
    rhos = sorted({r["rho"] for r in bud}, reverse=True)
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.95))
    for ax, src, title, tag in [(axs[0], "trace", "Intel trace", "Tr"), (axs[1], "synth", "Synthetic", "Sy")]:
        curves = {}
        for pol in ("rr", "sched_causal", "sched_best", "fgw", "oracle"):
            if src == "synth" and pol == "oracle":
                continue
            xs = np.array([60 * np.mean([r["poll_rate"] for r in gb[(src, pol, rho)]]) for rho in rhos])
            ys = np.array([100 * np.mean([r["viol"] for r in gb[(src, pol, rho)]]) for rho in rhos])
            curves[pol] = (xs, ys)
            lab, col, mk_, ls = STY[pol]
            ax.plot(xs, ys, ls, color=col, marker=mk_, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_xlabel("realised polls per hour")
        tidy_log_axes(ax, [0.5, 1, 2, 4, 6])
        # iso-accuracy saving at the RR operating point rho0 (realised rates)
        target = np.mean([r["viol"] for r in gb[(src, "rr", RHO0)]])
        xr = 60 * np.mean([r["poll_rate"] for r in gb[(src, "rr", RHO0)]])
        xf, yf = curves["fgw"]
        o = np.argsort(yf)
        need_log = float(np.exp(np.interp(np.log(100 * target), np.log(yf[o]), np.log(xf[o]))))
        need_lin = float(np.interp(100 * target, yf[o], xf[o]))
        sv = 100 * (1 - need_log / xr)
        M(f"{tag}Save", fmt(sv, 0)); M(f"{tag}SaveLin", fmt(100 * (1 - need_lin / xr), 0))
        M(f"{tag}Life", fmt(100 * (1 / (1 - sv / 100) - 1), 0))
        # lowest budget at which FG-Wake beats RR
        wins = [60 * rho for rho in rhos if np.mean([r["viol"] for r in gb[(src, "fgw", rho)]])
                < np.mean([r["viol"] for r in gb[(src, "rr", rho)]])]
        M(f"Bud{tag}WinMin", fmt(min(wins), 2) if wins else "n/a")
        lo = rhos[-1]
        M(f"Bud{tag}FgLow", fmt(100 * np.mean([r["viol"] for r in gb[(src, "fgw", lo)]]), 1))
        M(f"Bud{tag}RrLow", fmt(100 * np.mean([r["viol"] for r in gb[(src, "rr", lo)]]), 1))
        M(f"Bud{tag}FgLowRate", fmt(np.mean([r["poll_rate"] for r in gb[(src, "fgw", lo)]]) / lo, 2))
        M(f"Bud{tag}FgBeatsSchedB", str(sum(np.mean([r["viol"] for r in gb[(src, "fgw", rho)]])
                                           < np.mean([r["viol"] for r in gb[(src, "sched_best", rho)]]) for rho in rhos)))
        M(f"Bud{tag}FgBeatsSchedC", str(sum(np.mean([r["viol"] for r in gb[(src, "fgw", rho)]])
                                           < np.mean([r["viol"] for r in gb[(src, "sched_causal", rho)]]) for rho in rhos)))
    M("BudN", str(len(rhos)))
    for pol, nm in [("sched_best", "SchedB"), ("sched_causal", "SchedC")]:
        M(f"BudTr{nm}BeatsFg", str(sum(np.mean([r["viol"] for r in gb[("trace", pol, rho)]])
                                       < np.mean([r["viol"] for r in gb[("trace", "fgw", rho)]]) for rho in rhos)))
    hi = rhos[0]
    M("BudHighRate", fmt(60 * hi, 0))
    M("BudTrFgRateLowMax", fmt(max(np.mean([r["poll_rate"] for r in gb[("trace", "fgw", rho)]]) / rho
                                  for rho in rhos), 2))
    M("BudTrFgSensorLowMax", fmt(max(max(r["poll_rate_max"] for r in gb[("trace", "fgw", rho)]) / rho
                                    for rho in rhos), 2))
    rel = []
    for (src_, pol_, rho_), rs_ in gb.items():
        if src_ == "synth" and pol_ != "oracle":
            mm, hh = ci([100 * r["viol"] for r in rs_])
            rel.append(hh / mm)
    M("BudSyCiMax", fmt(100 * max(rel), 0))
    axs[0].set_ylabel(r"violation ratio $\bar v$ [%]")
    hdl, lbl = axs[0].get_legend_handles_labels()
    fig.legend(hdl, lbl, frameon=False, loc="upper center", ncol=3, handlelength=1.8,
               bbox_to_anchor=(0.5, 1.03), columnspacing=0.8)
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.80))
    fig.savefig(os.path.join(TEX, "fig_budget.pdf"))

# ------------------------------------------------------------------ alpha sweep
al = load("alpha")
if al:
    ga = group(al, "src", "pol", "alpha")
    alphas = sorted({r["alpha"] for r in al})
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75))
    for ax, src, title, tag in [(axs[0], "trace", "Intel trace", "Tr"), (axs[1], "synth", "Synthetic", "Sy")]:
        curves = {}
        for pol in ("rr", "dps", "fgw"):
            ys = [100 * np.mean([r["viol"] for r in ga[(src, pol, a)]]) for a in alphas]
            curves[pol] = np.array(ys)
            lab, col, mk_, ls = STY[pol]
            ax.plot(alphas, ys, ls, color=col, marker=mk_, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_xlabel(r"sensing cost $\alpha$")
        dd = np.log(curves["dps"]) - np.log(curves["fgw"])   # >0: FG-Wake better
        la = np.log(alphas)
        ups, downs = [], []
        for k in range(len(alphas) - 1):
            if (dd[k] <= 0 < dd[k + 1]) or (dd[k] > 0 >= dd[k + 1]):
                xc = float(np.exp(la[k] - dd[k] * (la[k + 1] - la[k]) / (dd[k + 1] - dd[k])))
                (ups if dd[k] <= 0 else downs).append(xc)
        M(f"AlphaLo{tag}", fmt(ups[0], 2) if ups else "n/a")
        M(f"AlphaHi{tag}", fmt(downs[0], 1) if downs else "none")
        M(f"{tag}DpsLowAlpha", fmt(curves["dps"][0], 2)); M(f"{tag}FgLowAlpha", fmt(curves["fgw"][0], 2))
        M(f"{tag}DpsHighAlpha", fmt(curves["dps"][-1], 2)); M(f"{tag}FgHighAlpha", fmt(curves["fgw"][-1], 2))
        M(f"{tag}RrHighAlpha", fmt(curves["rr"][-1], 2))
        M(f"{tag}DpsEmin", fmt(min(np.mean([r["energy"] for r in ga[(src, "dps", a)]]) / (2 * RHO0)
                                  for a in alphas), 2))
    M("AlphaMin", fmt(alphas[0], 2)); M("AlphaMax", fmt(alphas[-1], 0))
    M("BudHighAlpha", fmt(60 * 2 * RHO0 / (1 + alphas[-1]), 2))
    axs[0].set_ylabel(r"violation ratio $\bar v$ [%]")
    hdl, lbl = axs[0].get_legend_handles_labels()
    fig.legend(hdl, lbl, frameon=False, loc="upper center", ncol=3, handlelength=1.8, bbox_to_anchor=(0.5, 1.02))
    for ax in axs:
        tidy_log_axes(ax, [0.1, 0.3, 1, 3, 8])
    axs[0].set_yticks([2, 3, 5, 10, 20, 40]); axs[1].set_yticks([0.5, 1, 2, 5, 10, 20])
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(TEX, "fig_alpha.pdf"))

# ------------------------------------------------------------------ headroom
hd = load("headroom")
if hd:
    gh = group(hd, "src", "pol", "head")
    heads = sorted({r["head"] for r in hd})
    for src, tag in [("trace", "Tr"), ("synth", "Sy")]:
        reds = []
        for hv in heads:
            rr = np.mean([r["viol"] for r in gh[(src, "rr", hv)]])
            fg = np.mean([r["viol"] for r in gh[(src, "fgw", hv)]])
            K = gh[(src, "fgw", hv)][0]["K"]; N = gh[(src, "fgw", hv)][0]["N"]
            nm = {1.0: "One", 1.5: "OneHalf", 2.0: "Two", 3.0: "Three", 5.0: "Five"}[hv]
            M(f"Head{tag}{nm}", fmt(red(rr, fg), 1))
            M(f"HeadCap{tag}{nm}", fmt(K / (N * RHO0), 2))
            M(f"HeadK{tag}{nm}", str(K))
            reds.append(red(rr, fg))
        M(f"Head{tag}Min", fmt(min(reds), 0)); M(f"Head{tag}Max", fmt(max(reds), 0))

# ------------------------------------------------------------------ eps sensitivity
ep = load("eps")
if ep:
    ge = group(ep, "src", "pol", "eps")
    for src, tag in [("trace", "Tr"), ("heldout", "Ho"), ("synth", "Sy")]:
        for e, nm in [(0.25, "Q"), (0.5, "H"), (1.0, "One")]:
            rr = np.mean([r["viol"] for r in ge[(src, "rr", e)]])
            fg = np.mean([r["viol"] for r in ge[(src, "fgw", e)]])
            M(f"Eps{tag}{nm}", fmt(red(rr, fg), 0))
            M(f"EpsV{tag}{nm}RR", fmt(100 * rr, 1)); M(f"EpsV{tag}{nm}FG", fmt(100 * fg, 1))

# ------------------------------------------------------------------ diagnostics
dg = load("diag")
if dg:
    hrs = np.arange(24)
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 2.1), sharex=True)
    for pol in ("rr", "fgw"):
        lab, col, mk_, ls = STY[pol]
        axs[0].step(hrs, dg["hourly"][pol]["polls_per_sensor_hour"], where="mid", color=col, label=lab, ls=ls)
        axs[1].step(hrs, 100 * np.array(dg["hourly"][pol]["viol"]), where="mid", color=col, ls=ls)
    axs[0].set_ylabel("polls per hour", labelpad=2)
    axs[1].set_ylabel(r"$\bar v$ [%]", labelpad=2)
    axs[1].set_xlabel("hour of day")
    axs[1].set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
    axs[0].legend(frameon=False, ncol=2, loc="upper right", handlelength=1.8)
    fig.tight_layout(pad=0.3, h_pad=0.3)
    fig.savefig(os.path.join(TEX, "fig_hourly.pdf"))
    M("NightRR", fmt(100 * dg["hourly"]["rr"]["night_share"], 0))
    M("NightFG", fmt(100 * dg["hourly"]["fgw"]["night_share"], 0))
    pf = dg["hourly"]["fgw"]["polls_per_sensor_hour"]
    pk = int(np.argmax(pf))
    M("PeakFG", fmt(max(pf), 1)); M("PeakHourFG", f"{pk:02d}:00")
    M("RRrate", fmt(dg["hourly"]["rr"]["polls_per_sensor_hour"][12], 0))
    M("PeakVRR", fmt(100 * dg["hourly"]["rr"]["viol"][pk], 1))
    M("PeakVFG", fmt(100 * dg["hourly"]["fgw"]["viol"][pk], 1))
    M("PeakIsRRmax", "yes" if int(np.argmax(dg["hourly"]["rr"]["viol"])) == pk else "no")
    cr = np.array(dg["hourly"]["rr"]["viol_count"]); cf = np.array(dg["hourly"]["fgw"]["viol_count"])
    gain = cr - cf
    M("GainMorning", fmt(100 * gain[7:11].sum() / gain.sum(), 0))
    M("GainMorningGross", fmt(100 * gain[7:11].sum() / gain[gain > 0].sum(), 0))
    M("HoursWorse", str(int((gain < 0).sum())))
    night = [h_ for h_ in range(24) if h_ >= 20 or h_ < 7]
    M("HoursWorseNight", str(int(sum(gain[h_] < 0 for h_ in night))))
    loss = -np.minimum(gain, 0)
    top = np.argsort(-loss)[:3]
    M("LossTopHours", ", ".join(f"{int(h_):02d}:00" for h_ in sorted(top)))
    M("LossTopShare", fmt(100 * loss[top].sum() / loss.sum(), 0))
    hw = int(top[np.argmax(loss[top])])
    M("LossWorstHour", f"{hw:02d}:00")
    M("LossWorstRR", fmt(100 * dg["hourly"]["rr"]["viol"][hw], 1))
    M("LossWorstFG", fmt(100 * dg["hourly"]["fgw"]["viol"][hw], 1))
    lb = dg["lowbudget"]
    for k_, nm in [("rr_trend", "RRtrend"), ("rr_zoh", "RRzoh"), ("fgw_trend", "FGtrend"), ("fgw_zoh", "FGzoh")]:
        M(f"Lb{nm}", fmt(100 * lb[k_]["viol"], 1)); M(f"Lb{nm}Rate", fmt(lb[k_]["rate"], 2))
    cal = dg["calib"]
    M("CalGaussPred", fmt(cal["fgw_gauss"]["pred"], 2)); M("CalGaussObs", fmt(cal["fgw_gauss"]["obs"], 2))
    M("CalFgPred", fmt(cal["fgw"]["pred"], 2)); M("CalFgObs", fmt(cal["fgw"]["obs"], 2))
    M("EceGauss", fmt(cal["fgw_gauss"]["ece"], 2)); M("EceFg", fmt(cal["fgw"]["ece"], 2))
    sv = dg["shape_variant"]
    for k_, nm in [("trace_pre", "TrPre"), ("trace_post", "TrPost"), ("heldout_pre", "HoPre"),
                   ("heldout_post", "HoPost"), ("val_pre", "ValPre"), ("val_post", "ValPost")]:
        M(f"Sv{nm}", fmt(100 * sv[k_], 2))

# ------------------------------------------------------------------ regime threshold on the trace
# largest RR violation ratio among trace configurations where FG-Wake still beats RR;
# by construction, FG-Wake loses in every tested configuration above it.
pairs = []
for name in ("budget", "alpha", "eps", "headroom", "main"):
    d_ = load(name)
    if not d_:
        continue
    gg = group([r for r in d_ if r["src"] in ("trace", "heldout")], "src", "rho", "alpha", "head", "eps")
    for k_, rs_ in gg.items():
        v = {r["pol"]: r["viol"] for r in rs_}
        if "rr" in v and "fgw" in v:
            pairs.append((v["rr"], v["fgw"], name, k_))
win_rr = [p[0] for p in pairs if p[1] < p[0]]
lose_rr = [p[0] for p in pairs if p[1] >= p[0]]
thr = max(win_rr) if win_rr else 0
M("RegimeThr", fmt(np.ceil(100 * thr), 0))
M("RegimeLoseAbove", str(sum(1 for p in pairs if p[0] > thr)))
M("RegimeLoseAboveAll", "yes" if all(p[1] >= p[0] for p in pairs if p[0] > thr) else "no")
M("RegimeLoseBelow", str(sum(1 for p in pairs if p[0] <= thr and p[1] >= p[0])))

# ------------------------------------------------------------------ runtime
rt = load("runtime")
if rt:
    for row in rt["rows"]:
        nm = {28: "small", 100: "mid", 1000: "big"}[row["N"]]
        M(f"Rt{'FG' if row['pol'] == 'fgw' else 'RR'}{nm}", fmt(row["us"], 0))
    M("PyVer", rt["meta"]["python"]); M("NpVer", rt["meta"]["numpy"])

with open(os.path.join(TEX, "numbers.tex"), "w") as f:
    f.write("% Auto-generated by code/make_outputs.py from results/*.json -- do not edit.\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for k, v in sorted(macros.items()):
    print(f"{k:18s} {v}")
