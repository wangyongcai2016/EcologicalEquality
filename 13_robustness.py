#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 13 — Reproducibility and robustness of the function-stability diagnosis (H1)   [read-only on the pipeline outputs]
================================================================================
Why this script exists. 08_diagnostic_space.py classifies every stable natural pixel by the signs of dEFI (Theil-Sen
slope of the 21 annual EFI values, 2000-2020) and dESI (Theil-Sen slope of the 21 five-year ESI windows, centres
2002-2022). The manuscript quotes the resulting shares, so the questions below have to be answered with numbers:

  A  census          full-raster regime shares by ecosystem type, and how many pixels have dEFI == 0 exactly
                     (EFI is truncated at 1 and is 0 where the products are 0, so flat series are common; every such
                     pixel falls into the "dEFI <= 0" regimes)
  B  estimator       does the OLS slope of the ESI windows give the same sign as the Theil-Sen slope used in 08?
  C  reproducibility sign agreement of dESI (and dEFI) between two non-overlapping halves of the record, at the pixel,
                     24-km block, 48-km block and assessment-unit scale
  D  windows         fragile-greening share when the first windows are dropped, the period of dEFI is shifted, or only
                     windows whose data end by 2020 are used
  E  spatial         spatial coherence of dEFI and dESI, regime shares by latitude and longitude band
  F  completeness    why the Monte Carlo decline share (28.2 %) differs from the full-raster share (30.2 %)
  G  context         HJ 1172-2021 class shares of EFI x 100 on the analysed pixels, assessment-unit coverage

Pixel-scale analyses use every 6th pixel in both directions (3 km spacing, ~2.2 million cells), so shares carry a
sampling error of a few tenths of a percentage point; the census in A uses every pixel.

Output (OUT_DIR): robustness_summary.json, robustness_bands.csv
Run:  python 13_robustness.py
================================================================================
"""
from __future__ import annotations
import os, sys, json, csv, glob, struct, warnings
import numpy as np
import rasterio
from scipy.stats import spearmanr
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_io as gio

warnings.filterwarnings("ignore")
DEC = 6
OUT = gio.ensure_outdir()
RES = {}
NAMES = ["true_recovery", "fragile_greening", "low_but_stable", "collapse"]
TYPES = {0: "unclassified", 1: "forest", 2: "shrub", 3: "grassland", 4: "wetland", 5: "sparse"}


def L(n):
    return np.load(os.path.join(OUT, n), mmap_mode="r")


def dec_tif(p, dtype="float32"):
    with rasterio.open(p) as s:
        return s.read(1)[::DEC, ::DEC].astype(dtype)


def ols(Y):
    """OLS slope per column of Y (T, ...); every column must be finite."""
    t = np.arange(Y.shape[0], dtype="float64"); t -= t.mean()
    return np.tensordot(t, Y - Y.mean(0), axes=(0, 0)) / (t ** 2).sum()


def agree(a, b):
    return 100.0 * float((np.sign(a) == np.sign(b)).mean())


def rho(a, b, n=400_000, seed=0):
    if a.size > n:
        i = np.random.default_rng(seed).choice(a.size, n, replace=False); a, b = a[i], b[i]
    return float(spearmanr(a, b).correlation)


# ------------------------------------------------------------------------------ A  census on the full raster
print("[13] A census", flush=True)
dE_f = np.asarray(L("diag_dEFI.npy")); reg_f = np.asarray(L("diag_regime.npy"))
with rasterio.open(os.path.join(OUT, "eco_type.tif")) as s:
    et_f = s.read(1)
h = min(dE_f.shape[0], et_f.shape[0]); w = min(dE_f.shape[1], et_f.shape[1])
dE_f, reg_f, et_f = dE_f[:h, :w], reg_f[:h, :w], et_f[:h, :w]
valid_f = (reg_f <= 3) & np.isfinite(dE_f)
flat_f = valid_f & (dE_f == 0)


def census(m):
    n = int(m.sum()); d = {"n": n, "area_km2": n * 0.25}
    d["flat_dEFI_pct"] = 100 * float((m & flat_f).sum()) / n
    d["dEFI_pos_pct"] = 100 * float((m & (dE_f > 0)).sum()) / n
    nz = m & ~flat_f; nn = int(nz.sum())
    for k, nm in enumerate(NAMES):
        d[nm + "_pct"] = 100 * float((m & (reg_f == k)).sum()) / n
        d[nm + "_flat_pp"] = 100 * float((m & flat_f & (reg_f == k)).sum()) / n
        d[nm + "_excl_flat_pct"] = 100 * float((nz & (reg_f == k)).sum()) / nn
    d["fragile_of_greening_pct"] = 100 * d["fragile_greening_pct"] / d["dEFI_pos_pct"]
    return d


RES["census"] = {"all": census(valid_f)}
for t, nm in TYPES.items():
    RES["census"][nm] = census(valid_f & (et_f == t))
efi_f = np.asarray(L("efi_mean.npy"))[:h, :w]
RES["census"]["flat_mean_EFI"] = float(np.nanmean(efi_f[flat_f]))
RES["census"]["flat_EFI_eq0_pct"] = 100 * float((flat_f & (efi_f == 0)).sum()) / int(flat_f.sum())
RES["census"]["flat_EFI_ge095_pct"] = 100 * float((flat_f & (efi_f >= 0.95)).sum()) / int(flat_f.sum())
e100 = efi_f[valid_f & np.isfinite(efi_f)] * 100
RES["HJ1172_class_pct"] = {"excellent_ge75": 100 * float((e100 >= 75).mean()), "good_55_75": 100 * float(((e100 >= 55) & (e100 < 75)).mean()),
                           "medium_35_55": 100 * float(((e100 >= 35) & (e100 < 55)).mean()), "low_20_35": 100 * float(((e100 >= 20) & (e100 < 35)).mean()),
                           "poor_lt20": 100 * float((e100 < 20).mean()), "exact_zero": 100 * float((e100 == 0).mean())}
del dE_f, reg_f, et_f, efi_f, valid_f, flat_f

# ------------------------------------------------------------------------------ load the decimated layers
print("[13] loading decimated layers", flush=True)
dE = np.asarray(L("diag_dEFI.npy")[::DEC, ::DEC]); dS = np.asarray(L("diag_dESI.npy")[::DEC, ::DEC])
reg = np.asarray(L("diag_regime.npy")[::DEC, ::DEC]).astype("int16")
efi_stack = L("efi_stack.npy")
E = np.stack([np.asarray(efi_stack[i, ::DEC, ::DEC]) for i in range(efi_stack.shape[0])])      # (25, H, W), 2000..2024
trend = np.asarray(L("eqi_trend_core.npy")[::DEC, ::DEC])
et = dec_tif(os.path.join(OUT, "eco_type.tif"), "int16")
U = dec_tif(gio.CONFIG["UNITS_TIF"], "int32")
files = sorted(glob.glob(os.path.join(OUT, "esi_window_20*.tif")))
A = np.stack([dec_tif(f) for f in files]); A[(A < -1) | (A > 2)] = np.nan
with rasterio.open(os.path.join(OUT, "eco_type.tif")) as s:
    TR = s.transform
H, W = dE.shape
H = min(H, A.shape[1], E.shape[1], U.shape[0], et.shape[0]); W = min(W, A.shape[2], E.shape[2], U.shape[1], et.shape[1])
dE, dS, reg, E, A, trend, et, U = dE[:H, :W], dS[:H, :W], reg[:H, :W], E[:, :H, :W], A[:, :H, :W], trend[:H, :W], et[:H, :W], U[:H, :W]
nW = A.shape[0]
print(f"[13] cells {H}x{W}, windows {nW}, EFI years {E.shape[0]}", flush=True)
valid = (reg >= 0) & (reg <= 3) & np.isfinite(dE) & np.isfinite(dS)
okA = np.all(np.isfinite(A), axis=0); okE = np.all(np.isfinite(E[:21]), axis=0)
m = valid & okA & okE
RES["cells"] = {"valid": int(valid.sum()), "with_all_windows_and_EFI": int(m.sum()), "decimation": DEC}

# ------------------------------------------------------------------------------ B  estimator
print("[13] B estimator", flush=True)
X = A[:, m]; Em = E[:21, m]
b_ols = ols(X); b_ts = dS[m]; e_ts = dE[m]
SE = (X - X.mean(0) - np.outer(np.arange(nW) - (nW - 1) / 2, b_ols)).std(0, ddof=2) / np.sqrt(((np.arange(nW) - (nW - 1) / 2) ** 2).sum())
z = np.abs(b_ols) / SE
RES["estimator"] = {
    "sign_agree_TS_OLS": agree(b_ts, b_ols),
    "mean_slope_TS": float(b_ts.mean()), "mean_slope_OLS": float(b_ols.mean()),
    "share_dESI_pos_pipeline": 100 * float((b_ts > 0).mean()),
    "median_abs_z": float(np.median(z)), "share_abs_z_gt2": 100 * float((z > 2).mean()),
    "neg_share_among_abs_z_gt2": 100 * float((b_ols[z > 2] < 0).mean()),
    "fragile_conf_pct": 100 * float(((e_ts > 0) & (b_ols < 0) & (z > 2)).mean()),
    "truerec_conf_pct": 100 * float(((e_ts > 0) & (b_ols > 0) & (z > 2)).mean()),
    "lag1_autocorr_windows": float(np.mean(((X - X.mean(0))[1:] * (X - X.mean(0))[:-1]).sum(0) / ((X - X.mean(0)) ** 2).sum(0))),
    "national_mean_ESI_by_window": [float(v) for v in X.mean(1)],
    "national_mean_EFI_by_year_2000_2020": [float(v) for v in Em.mean(1)],
}
# ------------------------------------------------------------------------------ C  reproducibility of the signs
print("[13] C reproducibility", flush=True)
h1 = list(range(0, 8)); h2 = list(range(13, 21))                 # ESI windows: data 2000-2011 vs 2013-2024
e1 = list(range(0, 10)); e2 = list(range(11, 21))                # EFI years 2000-2009 vs 2011-2020
RES["reproducibility"] = {}
bs1, bs2 = ols(X[h1]), ols(X[h2]); es1, es2 = ols(Em[e1]), ols(Em[e2])
RES["reproducibility"]["pixel"] = {
    "ESI_half_sign_agree": agree(bs1, bs2), "ESI_half_rho": rho(bs1, bs2),
    "EFI_half_sign_agree": agree(es1, es2), "EFI_half_rho": rho(es1, es2),
    "ESI_full_vs_half1": agree(b_ols, bs1), "ESI_full_vs_half2": agree(b_ols, bs2),
    "ESI_neg_share_half1": 100 * float((bs1 < 0).mean()), "ESI_neg_share_half2": 100 * float((bs2 < 0).mean()),
    "EFI_pos_share_half1": 100 * float((es1 > 0).mean()), "EFI_pos_share_half2": 100 * float((es2 > 0).mean())}


def blockify(arr, B, minfrac=0.5):
    T, Hh, Ww = arr.shape; hh = Hh // B * B; ww = Ww // B * B
    a = arr[:, :hh, :ww].reshape(T, hh // B, B, ww // B, B)
    cnt = np.isfinite(a).sum(axis=(2, 4))
    mean = np.nanmean(a, axis=(2, 4))
    mean[cnt < minfrac * B * B] = np.nan
    return mean


Am = np.where(m[None], A, np.nan); Ee = np.where(m[None], E[:21], np.nan)
for B in (4, 8):
    ba = blockify(Am, B); be = blockify(Ee, B)
    ok = np.all(np.isfinite(ba), axis=0) & np.all(np.isfinite(be), axis=0)
    ba, be = ba[:, ok], be[:, ok]
    s1, s2 = ols(ba[h1]), ols(ba[h2]); q1, q2 = ols(be[e1]), ols(be[e2])
    sf, qf = ols(ba), ols(be)
    RES["reproducibility"][f"block_{B * DEC * 0.5:.0f}km"] = {
        "n_blocks": int(ok.sum()), "ESI_half_sign_agree": agree(s1, s2), "ESI_half_rho": rho(s1, s2),
        "ESI_chance_agree": 100 * float((s1 > 0).mean() * (s2 > 0).mean() + (s1 <= 0).mean() * (s2 <= 0).mean()),
        "EFI_chance_agree": 100 * float((q1 > 0).mean() * (q2 > 0).mean() + (q1 <= 0).mean() * (q2 <= 0).mean()),
        "EFI_half_sign_agree": agree(q1, q2), "EFI_half_rho": rho(q1, q2),
        "rho_dEFI_dESI": rho(qf, sf), "fragile_pct": 100 * float(((qf > 0) & (sf < 0)).mean()),
        "true_recovery_pct": 100 * float(((qf > 0) & (sf > 0)).mean()), "dESI_pos_pct": 100 * float((sf > 0).mean()), "dEFI_pos_pct": 100 * float((qf > 0).mean())}
# assessment units (pixels pooled, >= 200 sampled cells)
rows = []
for u in np.unique(U[U > 0]):
    sel = (U == u) & m
    if sel.sum() < 200:
        continue
    rows.append((int(u), int(sel.sum()), A[:, sel].mean(1), E[:21, sel].mean(1)))
ua = np.array([r[2] for r in rows]); ue = np.array([r[3] for r in rows]); un = np.array([r[1] for r in rows])
ucs = ols(ua.T); uce = ols(ue.T)
_u1, _u2 = ols(ua[:, h1].T), ols(ua[:, h2].T); _v1, _v2 = ols(ue[:, e1].T), ols(ue[:, e2].T)
RES["reproducibility"]["unit"] = {
    "ESI_chance_agree": 100 * float((_u1 > 0).mean() * (_u2 > 0).mean() + (_u1 <= 0).mean() * (_u2 <= 0).mean()),
    "EFI_chance_agree": 100 * float((_v1 > 0).mean() * (_v2 > 0).mean() + (_v1 <= 0).mean() * (_v2 <= 0).mean()),
    "n_units": len(rows), "ESI_half_sign_agree": agree(ols(ua[:, h1].T), ols(ua[:, h2].T)), "ESI_half_rho": rho(ols(ua[:, h1].T), ols(ua[:, h2].T)),
    "EFI_half_sign_agree": agree(ols(ue[:, e1].T), ols(ue[:, e2].T)), "EFI_half_rho": rho(ols(ue[:, e1].T), ols(ue[:, e2].T)),
    "dEFI_pos_units": int((uce > 0).sum()), "dESI_pos_units": int((ucs > 0).sum()), "rho_dEFI_dESI": float(spearmanr(uce, ucs).correlation),
    "rho_p": float(spearmanr(uce, ucs).pvalue),
    "fragile_units": int(((uce > 0) & (ucs <= 0)).sum()), "fragile_area_pct": 100 * float(un[(uce > 0) & (ucs <= 0)].sum() / un.sum())}
# distribution of the fragile share over the assessment units (full-pixel counts from 08)
sh = [r for r in csv.DictReader(open(os.path.join(OUT, "diag_regime_shares.csv"), encoding="utf-8-sig")) if r["group"].startswith("unit_") and r["scheme"] == "all"]
nu = np.array([float(r["n_pixels"]) for r in sh]); fg = np.array([float(r["Fragile greening"]) for r in sh]); tr_ = np.array([float(r["True recovery"]) for r in sh])
cl = np.array([float(r["Collapse"]) for r in sh]); big = nu >= 1000
RES["units_fragile"] = {"n_units_total": len(sh), "n_units_ge1000px": int(big.sum()),
                        "pctl": {str(p): float(np.percentile(fg[big], p)) for p in (10, 25, 50, 75, 90)},
                        "units_fragile_gt_true": int((fg[big] > tr_[big]).sum()), "units_fragile_ge50": int((fg[big] >= 50).sum()),
                        "units_fragile_le25": int((fg[big] <= 25).sum()), "units_collapse_gt_fragile": int((cl[big] > fg[big]).sum()),
                        "weighted_mean": float((fg[big] * nu[big]).sum() / nu[big].sum())}

# ------------------------------------------------------------------------------ D  window / period variants
print("[13] D variants", flush=True)
pe = e_ts > 0
base = 100 * float((pe & (b_ts <= 0)).mean())
RES["variants"] = {"baseline_pipeline_TS_on_complete_subset": base, "baseline_OLS": 100 * float((pe & (b_ols <= 0)).mean())}
for lab, idx in (("drop_first_2_windows", list(range(2, 21))), ("drop_first_4_windows", list(range(4, 21))), ("windows_centres_2002_2018", list(range(0, 17))),
                 ("last_15_windows", list(range(6, 21)))):
    s = ols(X[idx])
    RES["variants"][lab] = {"fragile_pct": 100 * float((pe & (s <= 0)).mean()), "dESI_pos_pct": 100 * float((s > 0).mean()), "sign_agree_with_full": agree(s, b_ols)}
for lab, (a, b) in (("dEFI_2002_2022", (2, 23)), ("dEFI_2000_2024", (0, 25))):
    ee = ols(E[a:b, m]) if np.all(np.isfinite(E[a:b][:, m])) else None
    if ee is not None:
        RES["variants"][lab] = {"fragile_pct": 100 * float(((ee > 0) & (b_ts <= 0)).mean()), "dEFI_pos_pct": 100 * float((ee > 0).mean()),
                                "rho": rho(ee, b_ts)}

# ------------------------------------------------------------------------------ E  spatial coherence and bands
print("[13] E spatial", flush=True)
ok2 = valid & np.isfinite(dS)


def lag_rho(F, k):
    a = F[:, :-k]; b = F[:, k:]; mm = ok2[:, :-k] & ok2[:, k:]
    return rho(a[mm], b[mm], 400_000)


RES["spatial_coherence"] = {f"lag_{k * DEC * 0.5:.0f}km": {"dEFI": lag_rho(dE, k), "dESI": lag_rho(dS, k)} for k in (1, 3, 7)}
rr, cc = np.nonzero(valid)
xs = TR.c + (cc * DEC + 0.5) * TR.a; ys = TR.f + (rr * DEC + 0.5) * TR.e
tr = Transformer.from_crs("+proj=aea +lat_0=0 +lon_0=105 +lat_1=25 +lat_2=47 +datum=WGS84 +units=m +no_defs", 4326, always_xy=True)
lon, lat = tr.transform(xs, ys)
rg = reg[valid]; ee = dE[valid]
bands = []
for kind, vals, edges in (("lat", lat, [18, 25, 30, 35, 40, 45, 54]), ("lon", lon, [73, 85, 95, 105, 115, 125, 136])):
    for a, b in zip(edges[:-1], edges[1:]):
        s = (vals >= a) & (vals < b)
        if s.sum() < 2000:
            continue
        bands.append({"band": kind, "from": a, "to": b, "n_cells": int(s.sum()), "true_recovery": 100 * float((rg[s] == 0).mean()), "fragile_greening": 100 * float((rg[s] == 1).mean()),
                      "low_but_stable": 100 * float((rg[s] == 2).mean()), "collapse": 100 * float((rg[s] == 3).mean()), "flat_dEFI": 100 * float((ee[s] == 0).mean())})
with open(os.path.join(OUT, "robustness_bands.csv"), "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=list(bands[0].keys())); wr.writeheader(); wr.writerows(bands)
RES["bands"] = bands
# the type composition of the latitude bands (is the gradient a forest effect?)
tt = et[valid]
RES["band_type_mix"] = []
for a, b in zip([18, 25, 30, 35, 40, 45], [25, 30, 35, 40, 45, 54]):
    s = (lat >= a) & (lat < b)
    RES["band_type_mix"].append({"from": a, "to": b, **{TYPES[t]: 100 * float((tt[s] == t).mean()) for t in (1, 2, 3, 4, 5)}})
    for t in (1, 3):
        st = s & (tt == t)
        if st.sum() > 2000:
            RES["band_type_mix"][-1][f"fragile_in_{TYPES[t]}"] = 100 * float((rg[st] == 1).mean())

# ------------------------------------------------------------------------------ E2 levels vs trends: is the LEVEL of ESI / EFI reproducible between halves?
lev_esi1 = X[h1].mean(0); lev_esi2 = X[h2].mean(0); lev_efi1 = Em[e1].mean(0); lev_efi2 = Em[e2].mean(0)
RES["level_reproducibility"] = {"ESI_level_rho_halves": rho(lev_esi1, lev_esi2), "EFI_level_rho_halves": rho(lev_efi1, lev_efi2),
                                "ESI_level_change_mean": float((lev_esi2 - lev_esi1).mean()), "EFI_level_change_mean": float((lev_efi2 - lev_efi1).mean()),
                                "ESI_level_down_share": 100 * float((lev_esi2 < lev_esi1).mean()), "EFI_level_up_share": 100 * float((lev_efi2 > lev_efi1).mean())}
# chance agreement of the signs given the marginal shares of the two halves
def chance(a, b):
    p, q = (a > 0).mean(), (b > 0).mean(); return 100 * float(p * q + (1 - p) * (1 - q))
RES["reproducibility"]["pixel"]["ESI_chance_agree"] = chance(bs1, bs2)
RES["reproducibility"]["pixel"]["EFI_chance_agree"] = chance(es1, es2)

# ------------------------------------------------------------------------------ F  completeness of the ESI series vs decline share
v = np.isfinite(trend)
anyw = np.any(np.isfinite(A), axis=0)
RES["completeness"] = {"trend_cells": int(v.sum()), "complete_ESI_pct": 100 * float((v & okA).sum() / v.sum()),
                       "decline_all": 100 * float((trend[v] < 0).mean()), "decline_complete_ESI": 100 * float((trend[v & okA] < 0).mean()),
                       "decline_incomplete_ESI": 100 * float((trend[v & ~okA] < 0).mean())}

# ------------------------------------------------------------------------------ G  assessment-unit coverage (value-attribute table of the unit raster)
try:
    p = gio.CONFIG["UNITS_TIF"] + ".vat.dbf"
    with open(p, "rb") as f:
        hdr = f.read(32); n = struct.unpack("<I", hdr[4:8])[0]; hl = struct.unpack("<H", hdr[8:10])[0]; rl = struct.unpack("<H", hdr[10:12])[0]
        flds = []
        while True:
            d = f.read(32)
            if d[0] == 0x0D:
                break
            flds.append((d[:11].split(b"\x00")[0].decode(), d[16]))
        f.seek(hl); tot = 0
        for _ in range(n):
            r = f.read(rl); off = 1; row = {}
            for nm, ln in flds:
                row[nm] = r[off:off + ln].decode("utf-8", "ignore").strip(); off += ln
            tot += float(row["Count"])
    RES["unit_coverage"] = {"n_units": n, "pixels": tot, "area_km2": tot * 0.25}
except Exception as e:                                                   # noqa: BLE001
    RES["unit_coverage"] = {"error": str(e)}

json.dump(RES, open(os.path.join(OUT, "robustness_summary.json"), "w"), indent=1, ensure_ascii=False)
print("[13] wrote robustness_summary.json, robustness_bands.csv")
