"""Variant C: top-8 analogue matching + horizon persistence blend.

Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output). Standard library only,
deterministic, no network.

Method
------
1. Score every launch comparable for similarity to the target
   (unlock-type match, console platforms, scale class, pre-release
   ramp) and keep the top 8.
2. Kernel: per forecast hour, empirical quantiles of the analogues'
   log viewership ratios lr_c(h) = ln(V_c(h) / V_c(1)).
3. Blend the kernel with the house persistence forecast in log
   space, with a horizon-dependent weight, then rescale the spread
   around the blended median. Blend weights and the spread factor
   are constants calibrated by leave-one-out backtest over the
   practice comparables.
"""

import glob
import json
import math
import os
from datetime import datetime, timedelta

# Calibration constants (leave-one-out backtest over the launches:
# micro pinball 0.7552 for this combo vs 0.9297 for pure persistence;
# the grid optimum sat at the lam1 / spread edges of the searched grid).
LAM1 = 0.3   # blend weight on persistence, hours 2-6
LAM2 = 0.0   # blend weight on persistence, hours 7-14
LAM3 = 0.0   # blend weight on persistence, hours 15-24
SPREAD = 1.25  # log-spread multiplier around the blended median

LEVELS_DEFAULT = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_TABLE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
TOP_K = 8
LR_LO, LR_HI = -2.5, 2.5


def parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def classify_unlock(text):
    """Return 'ROLLING', 'GLOBAL' or 'UNKNOWN' for an unlock description."""
    if not text:
        return "UNKNOWN"
    low = text.lower()
    rolling = ("midnight local" in low) or ("rolling" in low) or ("local midnight" in low)
    glob = ("global" in low) or ("simultaneous" in low) or ("single unlock" in low)
    if rolling and not glob:
        return "ROLLING"
    if glob and not rolling:
        return "GLOBAL"
    if rolling and glob:
        # Mixed descriptions (e.g. one platform rolling, another global):
        # a midnight-local component dominates the early UTC hours.
        if "midnight local" in low or "local midnight" in low:
            return "ROLLING"
        return "GLOBAL"
    return "UNKNOWN"


def empirical_quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class Launch:
    """A comparable reduced to what scoring and the kernel need."""

    __slots__ = ("name", "unlock", "platforms", "day", "peak", "v1", "ramp")

    def __init__(self, comp):
        self.name = comp.get("name") or ""
        self.unlock = comp.get("unlock") or ""
        self.platforms = comp.get("platforms") or ""
        self.day = {}  # release-day hour (1..24) -> viewers
        self.v1 = None
        self.peak = 0.0
        self.ramp = None
        try:
            start = parse_time(comp.get("day_start"))
        except Exception:
            return
        pre = []
        for pt in comp.get("hourly") or []:
            try:
                when = parse_time(pt.get("from"))
                viewers = float(pt.get("viewers"))
            except Exception:
                continue
            off = (when - start).total_seconds() / 3600.0
            if off == int(off):
                hour = int(off) + 1
                if 1 <= hour <= 24:
                    self.day[hour] = viewers
                elif -24 <= int(off) < 0:
                    pre.append(viewers)
        if 1 in self.day:
            self.v1 = self.day[1]
        if self.day:
            self.peak = max(self.day.values())
        if self.v1 and self.v1 > 0 and pre:
            mean_pre = sum(pre) / len(pre)
            if mean_pre > 0:
                self.ramp = mean_pre / self.v1


def target_ramp(case, p1):
    """Mean of the last 24 history points before checkpoint, over p1."""
    hist = case.get("history") or []
    if len(hist) < 6 or p1 <= 0:
        return None
    vals = []
    for pt in hist[-24:]:
        try:
            vals.append(float(pt.get("viewers")))
        except Exception:
            continue
    if len(vals) < 6:
        return None
    mean_v = sum(vals) / len(vals)
    if mean_v <= 0:
        return None
    return mean_v / p1


def score_launch(launch, target_class, p1, ramp_t):
    score = 0
    cls = classify_unlock(launch.unlock)
    if target_class != "UNKNOWN" and cls != "UNKNOWN":
        if cls == target_class:
            score += 2
    else:
        score += 1
    if ("PS" in launch.platforms) or ("Xbox" in launch.platforms):
        score += 1
    if launch.peak > 0 and p1 > 0:
        ratio = launch.peak / max(p1, 1.0)
        if 0.2 <= ratio <= 10.0:
            score += 1
    if ramp_t is not None and launch.ramp is not None and launch.ramp > 0 and ramp_t > 0:
        if abs(math.log(ramp_t / launch.ramp)) <= 0.7:
            score += 1
    return score


def select_analogues(case, p1, launches):
    target_class = classify_unlock(case.get("notes") or "")
    ramp_t = target_ramp(case, p1)
    scored = []
    for launch in launches:
        s = score_launch(launch, target_class, p1, ramp_t)
        scored.append((launch, s))
    scored.sort(key=lambda item: (-item[1], -item[0].peak, item[0].name))
    return [launch for launch, _ in scored[:TOP_K]], scored


def lr_values(launches, hour):
    vals = []
    for launch in launches:
        if launch.v1 and launch.v1 > 0 and hour in launch.day:
            v = launch.day[hour]
            if v and v > 0:
                lr = math.log(v / launch.v1)
                vals.append(min(LR_HI, max(LR_LO, lr)))
    vals.sort()
    return vals


def kernel_and_persistence(case):
    """Return (levels, {hour: (kernel log-q list, persistence log-q list)})."""
    levels = case.get("levels") or LEVELS_DEFAULT
    known = case.get("known") or []
    p1 = 1.0
    if known:
        try:
            p1 = float(known[0].get("viewers"))
        except Exception:
            p1 = 1.0
    if not (p1 > 0):
        p1 = 1.0
    p1 = max(p1, 1.0)
    log_p1 = math.log(p1)

    launches = []
    for comp in case.get("comparables") or []:
        name = comp.get("name") or ""
        if "before release" in name:
            continue
        launches.append(Launch(comp))

    selected, _scored = select_analogues(case, p1, launches)

    out = {}
    for hour in range(2, 25):
        # Persistence log-quantiles for this hour.
        pers = []
        for tau in levels:
            z = Z_TABLE.get(tau)
            if z is None:
                # Fallback for unexpected levels: rough normal quantile.
                z = 0.0
            pers.append(log_p1 + z * 0.15 * math.sqrt(hour - 1))
        # Kernel log-quantiles from the selected analogues.
        vals = lr_values(selected, hour)
        if len(vals) < 3:
            vals = lr_values(launches, hour)
        if vals:
            kern = [log_p1 + empirical_quantile(vals, tau) for tau in levels]
        else:
            kern = list(pers)
        out[hour] = (kern, pers)
    return levels, out


def blend_forecast(case, lam1, lam2, lam3, spread):
    levels, per_hour = kernel_and_persistence(case)
    mid_idx = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
    rows = []
    for hour in range(2, 25):
        if hour <= 6:
            lam = lam1
        elif hour <= 14:
            lam = lam2
        else:
            lam = lam3
        kern, pers = per_hour[hour]
        blended = [(1.0 - lam) * k + lam * p for k, p in zip(kern, pers)]
        med = blended[mid_idx]
        qs = []
        for b in blended:
            log_q = med + spread * (b - med)
            q = math.exp(log_q)
            if q < 0.01:
                q = 0.01
            if q > 1e9:
                q = 1e9
            qs.append(q)
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"hour": hour, "q": qs})
    return {"hours": rows}


def forecast_case(case):
    return blend_forecast(case, LAM1, LAM2, LAM3, SPREAD)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, str(case.get("id")) + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)


if __name__ == "__main__":
    main()
