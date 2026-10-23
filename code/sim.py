"""Discrete-time simulator for sink-initiated (pull) wake-up polling of IoT sensors.

Model (paper, Sec. III):
  * N sensors, slotted time t = 0..T-1; sensor i observes x_i(t).
  * The sink wakes at most K sensors per slot through their wake-up receivers.
    A woken sensor senses (cost alpha) and transmits (cost 1): poll cost 1+alpha,
    in units of the energy of one data transmission.
  * Per-sensor energy budget: long-run polls per slot <= rho.
  * The sink keeps an estimate xhat_i(t) (zero-order hold, or linear trend
    extrapolation); all pull policies share the same estimator and differ only
    in *when* they poll.
  * Metrics after warm-up: violation ratio P(|x - xhat| > eps), MAE, mean AoI,
    polls per sensor per slot, energy per sensor per slot.

Pull policies
  rr          periodic polling at the budget rate (staggered phases)
  rnd         Bernoulli(rho) polling
  maxvar      largest predicted error variance among sensors holding a token
              (token bucket rate rho, depth S); KF-trace-like greedy
  fgw         FG-Wake (proposed): forecast -> calibrated violation probability
              -> renewal index W -> per-sensor energy price
  fgw_noshape FG-Wake without the shared time-of-day volatility shape
  fgw_gauss   FG-Wake without the online calibration (Gaussian erfc model)
  fgw_static  FG-Wake with forecaster and calibration frozen after warm-up
  fgw_genie   index computed from the true variance (synthetic only)
  oracle      knows the current true error (clairvoyant reference)
Push reference
  dps         dual prediction scheme with token bucket, sensor side
"""
import time as _time

import numpy as np
from scipy.special import erfc

DEFAULT_FG = dict(a_up=0.5, a_dn=0.2, V=100.0, S=20.0, beta=0.02, B=48,
                  Wcap=2.0, lr=0.02, Qcap=None)
W_INIT = np.array([-0.92, 0.97, 0.0])   # logistic fit of erfc(1/sqrt(2 e^l)) over l


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------
# synthetic process
# --------------------------------------------------------------------------
def gen_synth(N, T, rng, sigma0=0.01, day_mult=6.0, burst_mult=4.0,
              p_burst=1 / 2000, burst_len=60, h_range=(0.5, 2.0),
              slots_per_day=1440):
    """Random walk with diurnal + Markov-burst volatility and heterogeneous scale.
    Returns x (T x N) and the true per-slot standard deviation sig (T x N)."""
    h = np.exp(rng.uniform(np.log(h_range[0]), np.log(h_range[1]), N))
    t = np.arange(T)
    shift = rng.integers(-60, 61, N)
    tod = (t[:, None] + shift[None, :]) % slots_per_day
    day = (tod >= 7 * 60) & (tod < 19 * 60)
    m = np.where(day, day_mult, 1.0)
    b = np.ones((T, N))
    state = np.zeros(N, bool)
    u = rng.random((T, N))
    p_end = 1.0 / burst_len
    for k in range(T):
        state = np.where(state, u[k] > p_end, u[k] < p_burst)
        b[k] = np.where(state, burst_mult, 1.0)
    sig = sigma0 * h[None, :] * m * b
    x = np.cumsum(sig * rng.standard_normal((T, N)), axis=0)
    return x, sig


# --------------------------------------------------------------------------
# pull-based policies
# --------------------------------------------------------------------------
def schedrr_rates(x, rho, slots_per_day=1440, expo=1.0, stat="incr", warm=1440, prof=None):
    """Per-sensor hour-of-day polling rates for the schedule baselines.
    stat="incr":   s_h = std of the change of x over the nominal interval P = 1/rho for
                   intervals starting in hour h, computed in HINDSIGHT on the whole trace;
    stat="causal": the same statistic computed on the warm-up day only (deployable);
    prof given:    s_h = prof (24,) or (24, N), e.g. RR's hourly violation ratio in hindsight.
    rate_h is proportional to s_h**expo and averages rho over the day. Returns (24, N)."""
    T, N = x.shape
    if prof is not None:
        sd = np.asarray(prof, float)
        sd = np.repeat(sd[:, None], N, axis=1) if sd.ndim == 1 else sd
    else:
        P = max(1, int(round(1.0 / rho)))
        xx = x[:warm] if stat == "causal" else x
        d = xx[P:] - xx[:-P]
        hr = (np.arange(xx.shape[0] - P) % slots_per_day) // 60
        sd = np.stack([d[hr == h].std(axis=0) if np.any(hr == h) else np.zeros(N) for h in range(24)])
    sd = np.maximum(sd, 1e-3 * sd.mean(axis=0, keepdims=True) + 1e-12) ** expo
    return rho * 24 * sd / sd.sum(axis=0, keepdims=True)


def run_pull(x, policy, rho, K, eps, rng, alpha=1.0, noise=0.0, warm=1440,
             sig=None, fg=None, trend=False, gamma=0.3, slots_per_day=1440,
             mask=None, record_time=False, return_trace=False):
    """Simulate one pull policy on x (T x N); `mask` (T x N, bool) restricts the
    metrics to observed entries (interpolated trace minutes are not scored)."""
    fg = dict(DEFAULT_FG, **(fg or {}))
    T, N = x.shape
    P = max(1, int(round(1.0 / rho)))
    phase = (np.arange(N) * P) // N
    rng_pol = np.random.default_rng(rng.integers(1 << 31))

    a_up, a_dn, V, S = fg["a_up"], fg["a_dn"], fg["V"], fg["S"]
    beta, lr = fg["beta"], fg["lr"]
    Qcap = S if fg.get("Qcap") is None else fg["Qcap"]     # debt cap (default: = credit floor S)
    shape_post = fg.get("shape_u", "pre") == "post"        # sensitivity variant
    learner = policy.startswith("fgw") or policy == "maxvar"
    calib = policy in ("fgw", "fgw_noshape", "fgw_static", "fgw_myopic", "fgw_noproj")
    project = policy != "fgw_noproj"
    priced = policy.startswith("fgw") or policy == "oracle"
    B = 1 if policy == "fgw_noshape" else int(fg["B"])
    L = slots_per_day // B if B > 1 else T + 1
    Wcap = fg["Wcap"] / rho
    if policy == "schedrr":
        sr = schedrr_rates(x, rho, slots_per_day, fg.get("sched_expo", 1.0),
                           stat=fg.get("sched_stat", "incr"), warm=warm, prof=fg.get("sched_prof"))
        acc = (np.arange(N) + 0.5) / N                    # staggered phases

    yl = x[0] + noise * rng.standard_normal(N)            # initial collection round
    tl = np.zeros(N, dtype=np.int64)
    slope = np.zeros(N)
    s2_prior = eps ** 2 * rho
    shape = np.ones(B)                                    # shared time-of-day shape
    u = np.full(N, s2_prior)                              # per-sensor volatility scale
    vacc = np.zeros(N)                                    # G_i: sum of shape since last poll
    cumvar = np.zeros(N)                                  # true variance since poll (genie)
    Csum = np.zeros(N)                                    # C_i = sum_{h<=age} c_i(h)
    wcal = W_INIT.copy()
    Q = np.zeros(N)                                       # virtual energy queue in [-S, Qcap+1)
    tok = np.zeros(N)                                     # maxvar token bucket
    frozen = False
    c = np.zeros(N)

    viol = np.zeros(N); abserr = np.zeros(N); agesum = np.zeros(N)
    polls = np.zeros(N); nobs = np.zeros(N)
    n_eval = 0
    q_run = 0.0
    cal_pred = cal_lab = cal_n = 0.0
    cal_pairs = [] if fg.get("cal_pairs") else None
    n_failed = 0
    slot_time = 0.0
    vslot = np.zeros(T) if return_trace else None
    oslot = np.zeros(T) if return_trace else None
    pslot = np.zeros(T) if return_trace else None

    for t in range(1, T):
        age = t - tl
        if sig is not None:
            cumvar += sig[t - 1] ** 2
        b_now = ((t - 1) % slots_per_day) // L if B > 1 else 0
        b_nxt = (t % slots_per_day) // L if B > 1 else 0
        vacc += shape[b_now]
        xhat = yl + slope * age if trend else yl
        err = x[t] - xhat

        if record_time:
            t0 = _time.perf_counter()
        # ---------------- decision ----------------
        if policy == "rr":
            cand = np.flatnonzero((t + phase) % P == 0)
            if cand.size > K:
                cand = cand[np.argsort(-age[cand])[:K]]
        elif policy == "schedrr":
            acc += sr[(t % slots_per_day) // 60]
            cand = np.flatnonzero(acc >= 1.0)
            if cand.size > K:
                cand = cand[np.argsort(-acc[cand])[:K]]
            acc[cand] -= 1.0
        elif policy == "rnd":
            cand = np.flatnonzero(rng_pol.random(N) < rho)
            if cand.size > K:
                cand = rng_pol.choice(cand, K, replace=False)
        elif policy == "maxvar":
            tok = np.minimum(tok + rho, S)
            v = u * vacc
            elig = np.flatnonzero(tok >= 1.0)
            cand = elig[np.argsort(-v[elig])[:K]] if elig.size else elig
        else:
            if policy == "oracle":
                ae = np.abs(err)
                p = (ae > eps).astype(float) + 1e-3 * np.minimum(ae / eps, 1.0)
                score = V * p - (Q + S)
            else:
                if policy == "fgw_genie":
                    v = cumvar + 2 * noise ** 2
                    v_nxt = v + (sig[t] ** 2 if t < T else 0.0)
                else:
                    v = u * vacc + 2 * noise ** 2
                    v_nxt = v + u * shape[b_nxt]
                if calib:
                    c = _sig(wcal[0] + wcal[1] * np.log(np.maximum(v, 1e-12) / eps ** 2)
                             + wcal[2] * np.log(age * rho))
                    c_nxt = _sig(wcal[0] + wcal[1] * np.log(np.maximum(v_nxt, 1e-12) / eps ** 2)
                                 + wcal[2] * np.log((age + 1) * rho))
                else:
                    c = erfc(eps / np.sqrt(2.0 * np.maximum(v, 1e-12)))
                    c_nxt = erfc(eps / np.sqrt(2.0 * np.maximum(v_nxt, 1e-12)))
                if policy == "fgw_myopic":                 # ablation: no renewal index
                    score = c_nxt - (Q + S) / V
                else:
                    Csum += c
                    W = np.minimum(age * c_nxt - Csum, Wcap)   # renewal index
                    score = rho * W - (Q + S) / V              # price (Q_i+S)/(rho V)
            score = np.where(Q < Qcap, score, -np.inf)          # no borrowing beyond Qcap
            pos = np.flatnonzero(score > 0)
            cand = pos[np.argpartition(-score[pos], K - 1)[:K]] if pos.size > K else pos

        # ---------------- polls + learning ----------------
        a = np.zeros(N)
        if cand.size:
            a[cand] = 1.0                                   # energy is spent for every poll
            if mask is not None:                            # a poll in an unobserved minute
                cand = cand[mask[t, cand]]                  # fails: no reading reaches the sink
                n_failed += int(a.sum() - cand.size) if t >= warm else 0
        if cand.size:
            y = x[t, cand] + noise * rng.standard_normal(cand.size)
            d = (t - tl[cand]).astype(float)
            pred = yl[cand] + slope[cand] * d if trend else yl[cand]
            r2 = (y - pred) ** 2
            if t >= warm and policy.startswith("fgw"):          # calibration log
                cal_pred += float(c[cand].sum())
                cal_lab += float((np.sqrt(r2) > eps).sum())
                cal_n += cand.size
                if cal_pairs is not None:
                    cal_pairs.extend(zip(c[cand].tolist(), (np.sqrt(r2) > eps).astype(int).tolist()))
            if learner and not frozen:
                if calib:                                      # one SGD step per slot
                    vv = u[cand] * vacc[cand] + 2 * noise ** 2
                    phi = np.stack([np.ones(cand.size),
                                    np.log(np.maximum(vv, 1e-12) / eps ** 2),
                                    np.log(d * rho)], axis=1)
                    lab = (np.sqrt(r2) > eps).astype(float)
                    wcal += lr * ((lab - _sig(phi @ wcal))[:, None] * phi).mean(0)
                    if project:                                # cost non-decreasing
                        wcal[1] = max(wcal[1], 0.2)            # in variance and in age
                        wcal[2] = max(wcal[2], 0.0)
                ratio = r2 / np.maximum(vacc[cand], 1e-12)
                w = np.where(ratio > u[cand], a_up, a_dn)
                u_old = u[cand].copy()
                u[cand] = (1 - w) * u[cand] + w * ratio
                if B > 1:
                    for j, i in enumerate(cand):
                        rel = (r2[j] / d[j]) / max(u[i] if shape_post else u_old[j], 1e-12)
                        b0 = (tl[i] % slots_per_day) // L
                        nb = min(B, int((t - tl[i] - 1) // L) + 2)
                        bins = (b0 + np.arange(nb)) % B
                        shape[bins] = (1 - beta) * shape[bins] + beta * rel
                    shape /= shape.mean()
            if trend:
                slope[cand] = (1 - gamma) * slope[cand] + gamma * (y - yl[cand]) / d
            yl[cand] = y
            tl[cand] = t
            vacc[cand] = 0.0
            cumvar[cand] = 0.0
            Csum[cand] = 0.0
        if policy == "maxvar":
            tok -= a                                        # every attempted poll uses a token
        Q = np.maximum(Q + a - rho, -S)
        if record_time:
            slot_time += _time.perf_counter() - t0
        if policy == "fgw_static" and t == warm:
            frozen = True

        # ---------------- metrics (after warm-up, observed entries only) ----------------
        if t >= warm:
            q_run = max(q_run, float(Q.max()))
            obs = mask[t] if mask is not None else np.ones(N, bool)
            age2 = t - tl
            xhat2 = yl + slope * age2 if trend else yl
            e2 = np.abs(x[t] - xhat2)
            vv2 = (e2 > eps) & obs
            viol += vv2
            abserr += e2 * obs
            agesum += age2 * obs
            nobs += obs
            polls += a
            n_eval += 1
            if return_trace:
                vslot[t] = vv2.sum()
                oslot[t] = obs.sum()
                pslot[t] = a.sum()

    rate = polls / n_eval
    out = dict(
        viol=float(viol.sum() / nobs.sum()),
        viol_worst=float((viol / np.maximum(nobs, 1)).max()),
        mae=float(abserr.sum() / nobs.sum()),
        aoi=float(agesum.sum() / nobs.sum()),
        poll_rate=float(rate.mean()),
        poll_rate_max=float(rate.max()),
        overshoot_max_polls=float((polls - rho * n_eval).max()),
        energy=float(rate.mean() * (1 + alpha)),
        energy_max=float(rate.max() * (1 + alpha)),
        Q_runmax=q_run,
        obs_frac=float(nobs.sum() / (N * n_eval)),
        failed_frac=float(n_failed / max(polls.sum(), 1)),
    )
    if cal_pairs is not None:
        out["cal_pairs"] = cal_pairs
    if cal_n > 0:
        out["cal_pred"] = cal_pred / cal_n
        out["cal_obs"] = cal_lab / cal_n
    if record_time:
        out["slot_us"] = 1e6 * slot_time / (T - 1)
    if return_trace:
        out["viol_slot"] = vslot
        out["obs_slot"] = oslot
        out["poll_slot"] = pslot
        out["wcal"] = wcal.tolist()
    return out


# --------------------------------------------------------------------------
# push-based dual prediction scheme with token bucket (sensor side)
# --------------------------------------------------------------------------
def run_dps(x, Ts, rho, K, eps, rng, alpha=1.0, noise=0.0, warm=1440, S=20.0,
            trend=False, gamma=0.3, return_trace=False, kappa=1.0, mask=None):
    """Sensor samples every Ts slots (cost alpha); transmits (cost 1) when its
    reading deviates from the shared prediction by more than kappa*eps and it
    holds a token (kappa < 1 transmits before the tolerance is violated).
    Tokens accrue at rate B - alpha/Ts with B = rho*(1+alpha), so the
    per-sensor energy budget equals that of the pull policies. With Ts = 1/rho
    and a small kappa, DPS reduces to periodic sensing (= RR)."""
    T, N = x.shape
    Bud = rho * (1 + alpha)
    tok_rate = Bud - alpha / Ts
    if tok_rate <= 0:
        return None
    phase = (np.arange(N) * Ts) // N
    yl = x[0] + noise * rng.standard_normal(N)
    tl = np.zeros(N, dtype=np.int64)
    slope = np.zeros(N)
    tok = np.zeros(N)
    viol = np.zeros(N); abserr = np.zeros(N); agesum = np.zeros(N); nobs = np.zeros(N)
    nsamp = np.zeros(N); ntx = np.zeros(N); n_eval = 0
    vslot = np.zeros(T) if return_trace else None
    oslot = np.zeros(T) if return_trace else None
    for t in range(1, T):
        tok = np.minimum(tok + tok_rate, S)
        samp_all = np.flatnonzero((t + phase) % Ts == 0)
        samp = samp_all[mask[t, samp_all]] if mask is not None else samp_all
        tx = np.empty(0, dtype=np.int64)
        if t >= warm:
            nsamp[samp_all] += 1                            # sensing energy is always spent
        if samp.size:
            y = x[t, samp] + noise * rng.standard_normal(samp.size)
            d = (t - tl[samp]).astype(float)
            pred = yl[samp] + slope[samp] * d if trend else yl[samp]
            dev = np.abs(y - pred)
            idx = np.flatnonzero((dev > kappa * eps) & (tok[samp] >= 1.0))
            if idx.size > K:
                idx = idx[np.argpartition(-dev[idx], K - 1)[:K]]
            tx = samp[idx]
            if tx.size:
                if trend:
                    slope[tx] = (1 - gamma) * slope[tx] + gamma * (y[idx] - yl[tx]) / d[idx]
                yl[tx] = y[idx]
                tl[tx] = t
                tok[tx] -= 1.0
        if t >= warm:
            obs = mask[t] if mask is not None else np.ones(N, bool)
            ntx[tx] += 1
            age2 = t - tl
            xhat = yl + slope * age2 if trend else yl
            e2 = np.abs(x[t] - xhat)
            vv2 = (e2 > eps) & obs
            viol += vv2; abserr += e2 * obs; agesum += age2 * obs; nobs += obs
            n_eval += 1
            if return_trace:
                vslot[t] = vv2.sum()
                oslot[t] = obs.sum()
    en = (alpha * nsamp + ntx) / n_eval
    out = dict(viol=float(viol.sum() / nobs.sum()),
               viol_worst=float((viol / np.maximum(nobs, 1)).max()),
               mae=float(abserr.sum() / nobs.sum()),
               aoi=float(agesum.sum() / nobs.sum()),
               poll_rate=float(ntx.mean() / n_eval),
               energy=float(en.mean()), energy_max=float(en.max()),
               Ts=Ts, kappa=kappa, S=S)
    if return_trace:
        out["viol_slot"] = vslot
        out["obs_slot"] = oslot
    return out


TS_BASE = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 105, 120, 135, 150, 180]
KAPPA_GRID = [0.2, 0.5, 0.7, 1.0]
S_GRID = [10.0, 40.0]


def dps_ts_grid(rho):
    """Sampling periods up to the RR-equivalent period 1/rho, plus 1/rho itself."""
    trr = 1.0 / rho
    g = sorted({ts for ts in TS_BASE if ts <= trr} | {int(np.floor(trr)), int(np.ceil(trr))})
    return [ts for ts in g if ts >= 1]


def run_dps_best(x, rho, K, eps, rng_seed, alpha=1.0, noise=0.0, warm=1440,
                 S=20.0, trend=False, return_trace=False, mask=None):
    """DPS with sampling period, threshold and token depth chosen in hindsight
    (generous to DPS). The argument S is ignored; S_GRID is searched."""
    best = None
    for Ts in dps_ts_grid(rho):
        for kappa in KAPPA_GRID:
            for s_ in S_GRID:
                r = run_dps(x, Ts, rho, K, eps, np.random.default_rng(rng_seed), alpha=alpha,
                            noise=noise, warm=warm, S=s_, trend=trend,
                            return_trace=return_trace, kappa=kappa, mask=mask)
                if r is not None and (best is None or r["viol"] < best["viol"]):
                    best = r
    return best
