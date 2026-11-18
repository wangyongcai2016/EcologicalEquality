#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 09 — Attribution: SEM + Geodetector + MGWR (H2)   [no-args / VSCode-run]
================================================================================
Paper 2.5. SEM + Geodetector (per-sample table) and MGWR (per-location table
with coordinates) test driver partitioning AND spatial non-stationarity.

HOW TO RUN: put the two CSVs at the configured paths, then run this file.

INPUT (from geo_io.py CONFIG -> add these keys or use defaults under OUT_DIR):
    ATTR_TABLE_CSV : columns AT_change,AP_change,PET_change,AI_change,
                     Lu_change,FP_change,EFI_change,ESI_change,EPI_change,EQI_change
    MGWR_TABLE_CSV : columns x,y,dEQI,precip,temp,landuse,footprint
  (If absent and ALLOW_DEMO, synthetic tables are used.)

OUTPUT (CONFIG['OUT_DIR']): sem_estimates.csv, geodetector_q.csv,
    mgwr_local_coeffs.csv, mgwr_stationarity.csv, 09_mgwr_preview.png
================================================================================
"""
from __future__ import annotations
import os, csv, warnings
import numpy as np
import pandas as pd
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(1)

# default table locations (edit in geo_io CONFIG if you like)
ATTR_TABLE_CSV = CONFIG.get("ATTR_TABLE_CSV", os.path.join(CONFIG["OUT_DIR"], "attribution_table.csv"))
MGWR_TABLE_CSV = CONFIG.get("MGWR_TABLE_CSV", os.path.join(CONFIG["OUT_DIR"], "mgwr_table.csv"))

SEM_MODEL = """
Climate =~ AT_change + AP_change + PET_change + AI_change
Human   =~ Lu_change + FP_change
EQI_change ~ EFI_change + ESI_change + EPI_change
EFI_change ~ Climate + Human
ESI_change ~ Climate + Human
EPI_change ~ Climate + Human
Climate ~~ Human
"""


def run_sem(df, out, tag=""):
    try:
        import semopy
    except Exception as e:
        print(f"[SEM] semopy unavailable ({e}); skipping."); return
    # z-score every indicator: the raw changes differ by 3 orders of magnitude (precipitation slope sd ~5, EFI slope sd
    # ~0.006), which makes the maximum-likelihood optimisation fail (all paths 0, CFI 0.28 on the raw table)
    cols = [c for c in df.columns if c not in ("x", "y", "type", "w")]
    df = df[cols].dropna()
    df = (df - df.mean()) / df.std(ddof=0)
    model = semopy.Model(SEM_MODEL); res = model.fit(df)
    print(f"[SEM] n={len(df):,}, optimiser: {getattr(res, 'message', '')} success={getattr(res, 'success', '?')}")
    est = model.inspect(std_est=True); est.to_csv(os.path.join(out, f"sem_estimates{tag}.csv"), index=False)
    col = "Est. Std" if "Est. Std" in est.columns else "Estimate"
    def beta(l, r):
        row = est[(est["lval"]==l)&(est["op"]=="~")&(est["rval"]==r)]
        return float(row[col].iloc[0]) if len(row) else np.nan
    print(f"[SEM] paths to EQI: EFI={beta('EQI_change','EFI_change'):+.2f} "
          f"ESI={beta('EQI_change','ESI_change'):+.2f} EPI={beta('EQI_change','EPI_change'):+.2f}")
    try:
        st = semopy.calc_stats(model)
        print("[SEM] fit: " + ", ".join(f"{k}={float(st[k].iloc[0]):.3f}" for k in ["CFI","RMSEA","SRMR"] if k in st.columns))
        pd.DataFrame([{"n": len(df), **{k: float(st[k].iloc[0]) for k in ["CFI", "RMSEA", "SRMR"] if k in st.columns}}]).to_csv(
            os.path.join(out, f"sem_fit{tag}.csv"), index=False)
    except Exception:
        pass


def strata(x, k=5):
    x = np.asarray(x, float); s = np.full(x.size, -1, int); fin = np.isfinite(x)
    if fin.sum()==0: return s
    edges = np.unique(np.quantile(x[fin], np.linspace(0,1,k+1)))
    s[fin] = np.clip(np.digitize(x[fin], edges[1:-1]), 0, len(edges)-2)
    return s


def q_stat(y, st):
    y = np.asarray(y, float); m = np.isfinite(y)&(st>=0); y, st = y[m], st[m]
    if y.size<2 or y.var()==0: return np.nan
    within = sum((st==h).sum()*y[st==h].var() for h in np.unique(st))
    return float(1 - within/(y.size*y.var()))


def run_geodetector(df, out, response="EQI_change", tag=""):
    drivers = {"precip":"AP_change","temp":"AT_change","landuse":"Lu_change","footprint":"FP_change"}
    y = df[response].values
    S = {n: strata(df[c].values) for n, c in drivers.items() if c in df}
    q = {n: q_stat(y, s) for n, s in S.items()}
    rows = [["factor", n, f"{q[n]:.3f}"] for n in q]; names = list(S)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            rows.append(["interaction", f"{a} x {b}", f"{q_stat(y, S[a]*(S[b].max()+1)+S[b]):.3f}"])
    with open(os.path.join(out, f"geodetector_q{tag}.csv"), "w", newline="") as f:
        w=csv.writer(f); w.writerow(["kind","driver(s)","q"]); w.writerows(rows)
    print("[GEODET] q: " + ", ".join(f"{k}={v:.2f}" for k, v in q.items()))
    if "precip" in S and "temp" in S:
        print(f"[GEODET] precip x temp q = {q_stat(y, S['precip']*(S['temp'].max()+1)+S['temp']):.2f}")


def run_mgwr(df, out, response="dEQI", n_perm=int(os.environ.get("EQI_MGWR_PERM", 30)), tag="", table=None):
    """Multiscale GWR (one bandwidth per coefficient). mgwr needs numba, which does not support the NumPy of the main
    environment: when it cannot be imported, this function re-runs the script with CONFIG["MGWR_PYTHON"] (a venv)."""
    try:
        from mgwr.gwr import GWR, MGWR
        from mgwr.sel_bw import Sel_BW
    except Exception as e:
        py = CONFIG.get("MGWR_PYTHON")
        if py and os.path.exists(py):
            import subprocess, sys
            print(f"[MGWR] mgwr not importable here ({type(e).__name__}); running it with {py}", flush=True)
            r = subprocess.run([py, "-u", os.path.abspath(__file__), "--mgwr-only", "--table", table or MGWR_TABLE_CSV, "--tag", tag],
                               cwd=os.path.dirname(os.path.abspath(__file__)))
            if r.returncode:
                print(f"[MGWR] subprocess failed with code {r.returncode}")
            return
        print(f"[MGWR] mgwr unavailable ({e}) and no CONFIG['MGWR_PYTHON']; skipping."); return
    need = {"x","y",response,"precip","temp","landuse","footprint"}
    if not need.issubset(df.columns):
        print(f"[MGWR] missing {need - set(df.columns)}; skipping."); return
    df = df.dropna(subset=list(need)).reset_index(drop=True)
    coords = np.column_stack([df["x"].values, df["y"].values])
    Xcols = ["precip","temp","landuse","footprint"]
    if tag:   # type-specific models: a driver that is exactly 0 for >75% of the sample is locally constant -> singular GWR matrices
        dropped = [c for c in Xcols if (df[c] == 0).mean() > 0.75]
        if dropped:
            print(f"[MGWR] {tag.strip('_')}: dropping {dropped} (zero in {[round((df[c]==0).mean()*100) for c in dropped]}% of the sample)", flush=True)
            Xcols = [c for c in Xcols if c not in dropped]
    X = df[Xcols].values
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    y = df[[response]].values; y = (y - y.mean()) / (y.std() + 1e-12)
    import time; t0 = time.time()
    sel = Sel_BW(coords, y, X, multi=True)
    bws = sel.search(); print(f"[MGWR] n={len(df)}, bandwidths (nearest neighbours) intercept+{Xcols}: {np.round(bws, 0).tolist()}  "
                              f"({time.time() - t0:.0f}s)", flush=True)
    res = MGWR(coords, y, X, sel).fit()
    local = res.params; tvals = res.tvalues
    print(f"[MGWR] R2={res.R2:.3f}, adj R2={res.adj_R2:.3f}, AICc={res.aicc:.1f}  ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame([{"n": len(df), "R2": res.R2, "adjR2": res.adj_R2, "AICc": res.aicc,
                   **{f"bw_{n}": float(b) for n, b in zip(["intercept"] + Xcols, bws)}}]).to_csv(
        os.path.join(out, f"mgwr_fit{tag}.csv"), index=False)
    dfl = pd.DataFrame(local, columns=["intercept"]+Xcols); dfl["x"]=df["x"].values; dfl["y"]=df["y"].values
    dfl.to_csv(os.path.join(out, f"mgwr_local_coeffs{tag}.csv"), index=False)
    # stationarity: spread of each local coefficient vs. the spread under a permuted response (same bandwidths)
    rows = []
    for k, name in enumerate(Xcols, start=1):
        sig = np.mean(np.abs(tvals[:,k]) > 1.96)*100; spread = local[:,k].std()
        perm = []
        for _ in range(n_perm):
            gp = GWR(coords, RNG.permutation(y), X, bw=float(bws[k]), kernel="bisquare", fixed=False).fit()
            perm.append(gp.params[:,k].std())
        p = (np.sum(np.array(perm) >= spread) + 1) / (n_perm + 1)
        rows.append([name, f"{sig:.1f}", f"{spread:.3f}", f"{bws[k]:.0f}", f"{p:.3f}"])
        print(f"[MGWR] {name}: bandwidth {bws[k]:.0f}, sig {sig:.0f}%, coefficient sd={spread:.3f}, stationarity p={p:.3f} "
              f"({n_perm} permutations)", flush=True)
    with open(os.path.join(out, f"mgwr_stationarity{tag}.csv"), "w", newline="") as f:
        w=csv.writer(f); w.writerow(["driver","pct_significant","coeff_sd","bandwidth_nn","stationarity_p"]); w.writerows(rows)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        for a, name in zip(ax, ["precip","temp"]):
            k = Xcols.index(name)+1; mx = np.abs(local[:,k]).max()
            sc = a.scatter(df["x"], df["y"], c=local[:,k], cmap="RdBu", vmin=-mx, vmax=mx, s=8)
            a.set_title(f"MGWR local: {name}"); a.set_aspect("equal"); plt.colorbar(sc, ax=a, shrink=.8)
        plt.tight_layout(); plt.savefig(os.path.join(out, f"09_mgwr_preview{tag}.png"), dpi=120)
        print(f"[MGWR] preview -> {out}/09_mgwr_preview.png")
    except Exception as e:
        print(f"[MGWR] (preview skipped: {e})")


def demo_sem_table(n=800):
    AT=RNG.standard_normal(n); AP=RNG.standard_normal(n)
    PET=0.5*AT+0.5*RNG.standard_normal(n); AI=-0.4*AP+0.3*PET+0.4*RNG.standard_normal(n)
    Lu=RNG.standard_normal(n); FP=0.6*Lu+0.5*RNG.standard_normal(n)
    clim=0.6*AP+0.3*AT-0.2*AI; hum=0.7*Lu+0.5*FP
    EFI=0.70*clim-0.06*hum+0.3*RNG.standard_normal(n); ESI=-0.02*clim-0.04*hum+0.3*RNG.standard_normal(n)
    EPI=0.05*clim+0.60*hum+0.3*RNG.standard_normal(n); EQI=0.86*EFI+0.20*ESI-0.76*EPI+0.1*RNG.standard_normal(n)
    return pd.DataFrame(dict(AT_change=AT,AP_change=AP,PET_change=PET,AI_change=AI,Lu_change=Lu,FP_change=FP,
        EFI_change=EFI,ESI_change=ESI,EPI_change=EPI,EQI_change=EQI))


def demo_mgwr_table(n=900):
    gx, gy = np.meshgrid(np.linspace(0,1,30), np.linspace(0,1,30)); x, y = gx.ravel(), gy.ravel()
    precip=RNG.standard_normal(n); temp=RNG.standard_normal(n); landuse=RNG.standard_normal(n); footprint=RNG.standard_normal(n)
    plateau = (x>0.66)&(y>0.5); beta_temp = np.where(plateau, -0.8, -0.05); beta_precip = np.where(y<0.5, 0.6, 0.2)
    dEQI = beta_precip*precip + beta_temp*temp + 0.1*landuse - 0.2*footprint + 0.1*RNG.standard_normal(n)
    return pd.DataFrame(dict(x=x*1000, y=y*1000, dEQI=dEQI, precip=precip, temp=temp, landuse=landuse, footprint=footprint))


def main():
    import sys
    c = CONFIG; out = gio.ensure_outdir()
    if "--mgwr-only" in sys.argv:                       # entered from the main environment through the mgwr venv
        tb = sys.argv[sys.argv.index("--table") + 1] if "--table" in sys.argv else MGWR_TABLE_CSV
        tg = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
        run_mgwr(pd.read_csv(tb), out, tag=tg, table=tb); return
    demo = "--demo" in sys.argv
    if demo:
        print("[ATTR] *** DEMO MODE: synthetic tables, results are meaningless ***")
        sem_df, mgwr_df = demo_sem_table(), demo_mgwr_table()
    else:
        # no silent synthetic fallback: build the tables with 09a_sample_attribution.py
        if not os.path.exists(ATTR_TABLE_CSV):
            raise FileNotFoundError(f"[ATTR] {ATTR_TABLE_CSV} not found; run 09a_sample_attribution.py (or use --demo)")
        sem_df = pd.read_csv(ATTR_TABLE_CSV); print(f"[ATTR] loaded {ATTR_TABLE_CSV} ({len(sem_df):,} rows)")
        mgwr_df = pd.read_csv(MGWR_TABLE_CSV) if os.path.exists(MGWR_TABLE_CSV) else None
        if mgwr_df is not None:
            print(f"[ATTR] loaded {MGWR_TABLE_CSV} ({len(mgwr_df):,} rows)")

    TN = {1: "forest", 2: "shrub", 3: "grassland", 4: "wetland", 5: "sparse"}
    # pooled model: weighted resample (weights = type pixels / sampled pixels) so that the pooled sample is representative
    if "w" in sem_df and "type" in sem_df:
        pooled = sem_df.sample(n=min(40000, len(sem_df)), weights=sem_df["w"], random_state=7, replace=True).reset_index(drop=True)
    else:
        pooled = sem_df
    print("\n--- (a) SEM ---"); run_sem(pooled, out)
    print("\n--- (b) Geodetector ---"); run_geodetector(pooled, out)
    if "type" in sem_df:
        for t, tn in TN.items():
            g = sem_df[sem_df["type"] == t]
            if len(g) < 500:
                print(f"[ATTR] {tn}: only {len(g)} rows, skipped"); continue
            print(f"\n--- by type: {tn} (n={len(g):,}) ---")
            run_sem(g, out, tag="_" + tn); run_geodetector(g, out, tag="_" + tn)
    print("\n--- (c) MGWR ---")
    if "--no-mgwr" in sys.argv: print("[MGWR] skipped (--no-mgwr)")
    elif mgwr_df is not None:
        run_mgwr(mgwr_df, out)
        for nm in ("forest", "grassland"):
            pth = os.path.join(out, f"mgwr_table_{nm}.csv")
            if os.path.exists(pth):
                print(f"\n--- MGWR {nm} ---"); run_mgwr(pd.read_csv(pth), out, tag="_" + nm, table=pth)
    else: print("[MGWR] no mgwr_table.csv; skipping.")
    print("\n[ATTR] done.")


if __name__ == "__main__":
    main()
