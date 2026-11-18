#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect the numbers quoted in the manuscript (results section) into output/facts.json. Read-only on the pipeline outputs."""
import os, json, csv, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_io as gio
import rasterio
from scipy.stats import spearmanr

out = gio.ensure_outdir()
F = {}
TY = {1: "forest", 2: "shrub", 3: "grassland", 4: "wetland", 5: "sparse", 0: "unclassified"}
etype = np.nan_to_num(gio.read_tif(os.path.join(out, "eco_type.tif"))[0], nan=0).astype("uint8")


def L(n):
    return np.load(os.path.join(out, n), mmap_mode="r")


def desc(a, m=None):
    v = np.asarray(a)[m] if m is not None else np.asarray(a).ravel()
    v = v[np.isfinite(v)]
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std()), p5=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)))


efi, esi, epi = np.asarray(L("efi_mean.npy")), np.asarray(L("esi.npy")), np.asarray(L("epi_total.npy"))
eqi = np.asarray(L("eqi_equal.npy"))
base = np.isfinite(efi) & np.isfinite(esi) & np.isfinite(epi)
F["n_analysed"] = int(base.sum())
for nm, a in (("EFI", efi), ("ESI", esi), ("EPI", epi), ("EQI", eqi)):
    F[nm] = {"all": desc(a, base)}
    for t, tn in TY.items():
        m = base & (etype == t)
        if m.sum():
            F[nm][tn] = desc(a, m)
# HJ 1172 classes on EFI x 100
v = efi[np.isfinite(efi)] * 100
F["EFI_class_share"] = {k: float(((v >= lo) & (v < hi)).mean() * 100) for k, (lo, hi) in
                        {"excellent_ge75": (75, 1e9), "good_55_75": (55, 75), "medium_35_55": (35, 55), "low_20_35": (20, 35), "poor_lt20": (-1e9, 20)}.items()}
F["EFI_exact_zero_share_pct"] = float((efi[np.isfinite(efi)] == 0).mean() * 100)
# reliability terciles by type
with rasterio.open(os.path.join(out, "esi_reliability.tif")) as s:
    rel = s.read(1)
F["reliability_by_type"] = {}
for t, tn in TY.items():
    m = base & (etype == t) & np.isfinite(rel)
    if m.sum():
        F["reliability_by_type"][tn] = {f"tercile{int(k)}_pct": float((rel[m] == k).mean() * 100) for k in (0, 1, 2)}
m = base & np.isfinite(rel)
F["lowconf_tercile_share_pct"] = float((rel[m] == 2).mean() * 100)
F["forest_share_of_lowconf_pct"] = float(((rel == 2) & base & (etype == 1)).sum() / max(((rel == 2) & base).sum(), 1) * 100)
# TAC vs DFA
def tif(n):
    return gio.read_tif(os.path.join(out, n))[0]
tac, dfa = tif("esi_tac.tif"), tif("esi_dfa_alpha.tif"); res_ = tif("esi_resistance.tif")
k = base & np.isfinite(tac) & np.isfinite(dfa)
idx = np.flatnonzero(k.ravel()); idx = np.random.default_rng(1).choice(idx, min(idx.size, 500000), replace=False)
F["tac_dfa_spearman"] = float(spearmanr(tac.ravel()[idx], dfa.ravel()[idx]).correlation)
k2 = base & np.isfinite(tac) & np.isfinite(res_)
i2 = np.random.default_rng(2).choice(np.flatnonzero(k2.ravel()), 500000, replace=False)
F["tac_resistance_spearman"] = float(spearmanr(tac.ravel()[i2], res_.ravel()[i2]).correlation)
# changes: correlations among dEFI, dESI, dEPI
dE, dS, dP = np.asarray(L("diag_dEFI.npy")), np.asarray(L("diag_dESI.npy")), np.asarray(L("epi_trend.npy"))
kk = np.isfinite(dE) & np.isfinite(dS) & np.isfinite(dP)
ii = np.random.default_rng(3).choice(np.flatnonzero(kk.ravel()), 500000, replace=False)
a, b, c = dE.ravel()[ii], dS.ravel()[ii], dP.ravel()[ii]
F["change_corr_spearman"] = {"dEFI_dESI": float(spearmanr(a, b).correlation), "dEFI_dEPI": float(spearmanr(a, c).correlation),
                             "dESI_dEPI": float(spearmanr(b, c).correlation)}
F["dEPI"] = dict(mean=float(np.nanmean(dP)), pos_share=float((dP[np.isfinite(dP)] > 0).mean() * 100))
# trends (06) by type
sl, pv, cd, hu, ps = (np.asarray(L(f"trend_{n}.npy")) for n in ("slope", "pvalue", "code", "hurst", "persist"))
vt = np.isfinite(sl)
F["trend"] = {}
for t, tn in [(-1, "all")] + list(TY.items()):
    m = vt & ((etype == t) if t >= 0 else True)
    if not m.sum():
        continue
    F["trend"][tn] = dict(n=int(m.sum()), mean_slope=float(sl[m].mean()), inc=float((sl[m] > 0).mean() * 100), dec=float((sl[m] < 0).mean() * 100),
                          sig_up=float((cd[m] == 1).mean() * 100), sig_dn=float((cd[m] == -1).mean() * 100),
                          hurst_gt05=float((hu[m & np.isfinite(hu)] > 0.5).mean() * 100), persist_decline=float((ps[m] == -2).mean() * 100),
                          persist_incr=float((ps[m] == 2).mean() * 100))
# bfast
nb, bt_, sp0, sp1 = (np.asarray(L(f"bfast_{n}.npy")) for n in ("nbreaks", "break_time", "slope_pre", "slope_post"))
vb = np.isfinite(nb)
F["bfast"] = dict(analysed=int(vb.sum()), with_break=float((nb[vb] > 0).mean() * 100))
hb = vb & (nb > 0)
yrs, cnt = np.unique(bt_[hb & np.isfinite(bt_)].astype(int), return_counts=True)
F["bfast"]["year_share"] = {int(y): float(c / cnt.sum() * 100) for y, c in zip(yrs, cnt)}
F["bfast"]["median_year"] = float(np.median(bt_[hb & np.isfinite(bt_)]))
F["bfast"]["rev_dec_to_inc"] = float((hb & (sp0 <= 0) & (sp1 > 0)).sum() / hb.sum() * 100)
F["bfast"]["rev_inc_to_dec"] = float((hb & (sp0 >= 0) & (sp1 < 0)).sum() / hb.sum() * 100)
F["bfast"]["by_type_with_break"] = {tn: float((nb[vb & (etype == t)] > 0).mean() * 100) for t, tn in TY.items() if (vb & (etype == t)).sum()}
# annual EQI series
rows = list(csv.DictReader(open(os.path.join(out, "eqi_yearly_mean.csv"), encoding="utf-8-sig")))
y = np.array([int(r["year"]) for r in rows]); e_all = np.array([float(r["EQI"]) for r in rows]); e_fx = np.array([float(r["EQI_fixed"]) for r in rows])
def rate(yy, ee, lo, hi):
    m = (yy >= lo) & (yy <= hi); sl_ = np.polyfit(yy[m], ee[m], 1)[0]
    return dict(slope=float(sl_), pct_per_yr=float(sl_ / ee[m].mean() * 100))
F["eqi_rate"] = {f"{n}_{lo}_{hi}": rate(y, e, lo, hi) for n, e in (("all", e_all), ("fixed", e_fx)) for lo, hi in ((2002, 2020), (2003, 2020), (2000, 2024))}
F["eqi_fixed_series"] = {int(a): float(b) for a, b in zip(y, e_fx)}
F["eqi_all_series"] = {int(a): float(b) for a, b in zip(y, e_all)}
F["n_fixed"] = int(rows[0]["n_fixed"])
# EPI vs HFP
hy = range(2002, 2019); acc = None
for yr in hy:
    with rasterio.open(os.path.join(gio.CONFIG["HFP_DIR"], gio.CONFIG["HFP_PATTERN"].format(year=yr))) as s:
        a_ = s.read(1).astype("float32")
    a_[a_ == -9999] = np.nan
    acc = a_ if acc is None else acc + a_
hfp = acc / len(hy)
kh = base & np.isfinite(hfp)
ih = np.random.default_rng(4).choice(np.flatnonzero(kh.ravel()), 500000, replace=False)
F["epi_hfp_spearman"] = float(spearmanr(epi.ravel()[ih], hfp.ravel()[ih]).correlation)
F["epi_human_hfp_spearman"] = float(spearmanr(np.asarray(L("epi_human.npy")).ravel()[ih], hfp.ravel()[ih]).correlation)
F["epi_hfp_spearman_by_type"] = {}
for t, tn in TY.items():
    mt = kh & (etype == t)
    if mt.sum() > 2000:
        it = np.random.default_rng(5).choice(np.flatnonzero(mt.ravel()), min(int(mt.sum()), 200000), replace=False)
        F["epi_hfp_spearman_by_type"][tn] = float(spearmanr(epi.ravel()[it], hfp.ravel()[it]).correlation)
# HFP trend 2002-2018 and EPI trend
F["hfp_mean_2002_2018"] = float(np.nanmean(hfp[base]))
# weights and stable
F["stable_comparison"] = {r["item"]: int(r["pixels"]) for r in csv.DictReader(open(os.path.join(out, "Lu_stable_comparison.csv"), encoding="utf-8-sig"))}
F["weights"] = list(csv.DictReader(open(os.path.join(out, "eqi_weights.csv"), encoding="utf-8-sig")))
# low confidence units
lc = np.asarray(L("efi_lowconf_mask.npy")); F["efi_lowconf_px"] = int((lc & base).sum())
# type areas (km2, 0.25 km2 per pixel)
F["type_area_share_pct"] = {tn: float((base & (etype == t)).sum() / base.sum() * 100) for t, tn in TY.items()}
F["n_units"] = int(len(np.unique(gio.read_units()[base])))
json.dump(F, open(os.path.join(out, "facts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("facts.json written;", len(F), "keys")
