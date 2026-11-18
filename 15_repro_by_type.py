#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15 -- Reproducibility of the dESI / dEFI signs by ecosystem type and by NPP reliability tercile (extends 13_robustness.py, section C).
Same decimation (every 6th pixel), same halves (ESI windows 1-8 vs 14-21; EFI years 2000-2009 vs 2011-2020), same chance-agreement definition.
Also reports, per group, the share of pixels whose ESI trend is "confident" (|OLS slope| / SE > 2, windows overlap, so this is only a screening
criterion) and falls in fragile greening or true recovery.
Output (OUT_DIR): repro_by_type.json
"""
from __future__ import annotations
import os, sys, json, glob, warnings
import numpy as np, rasterio
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_io as gio

warnings.filterwarnings("ignore")
DEC = 6
OUT = gio.ensure_outdir()
TYPES = {1: "forest", 2: "shrub", 3: "grassland", 4: "wetland", 5: "sparse"}


def L(n):
    return np.load(os.path.join(OUT, n), mmap_mode="r")


def dec_tif(p, dtype="float32"):
    with rasterio.open(p) as s:
        return s.read(1)[::DEC, ::DEC].astype(dtype)


def ols(Y):
    t = np.arange(Y.shape[0], dtype="float64"); t -= t.mean()
    return np.tensordot(t, Y - Y.mean(0), axes=(0, 0)) / (t ** 2).sum()


def agree(a, b):
    return 100.0 * float((np.sign(a) == np.sign(b)).mean())


def chance(a, b):
    p, q = (a > 0).mean(), (b > 0).mean()
    return 100 * float(p * q + (1 - p) * (1 - q))


dE = np.asarray(L("diag_dEFI.npy")[::DEC, ::DEC]); reg = np.asarray(L("diag_regime.npy")[::DEC, ::DEC]).astype("int16")
ef = L("efi_stack.npy")
E = np.stack([np.asarray(ef[i, ::DEC, ::DEC]) for i in range(21)])
et = dec_tif(os.path.join(OUT, "eco_type.tif"), "int16")
rel = dec_tif(os.path.join(OUT, "esi_reliability.tif"), "int16")
A = np.stack([dec_tif(f) for f in sorted(glob.glob(os.path.join(OUT, "esi_window_20*.tif")))]); A[(A < -1) | (A > 2)] = np.nan
H = min(dE.shape[0], A.shape[1], E.shape[1], et.shape[0], rel.shape[0]); W = min(dE.shape[1], A.shape[2], E.shape[2], et.shape[1], rel.shape[1])
dE, reg, E, A, et, rel = dE[:H, :W], reg[:H, :W], E[:, :H, :W], A[:, :H, :W], et[:H, :W], rel[:H, :W]
m = (reg >= 0) & (reg <= 3) & np.all(np.isfinite(A), 0) & np.all(np.isfinite(E), 0) & np.isfinite(dE)
h1, h2 = list(range(0, 8)), list(range(13, 21)); e1, e2 = list(range(0, 10)), list(range(11, 21))
nW = A.shape[0]
res = {}


def block(sel, name):
    s = sel & m
    n = int(s.sum())
    if n < 2000:
        return
    X = A[:, s]; Em = E[:, s]
    b = ols(X); t = np.arange(nW) - (nW - 1) / 2
    se = (X - X.mean(0) - np.outer(t, b)).std(0, ddof=2) / np.sqrt((t ** 2).sum())
    z = np.abs(b) / se
    a1, a2, q1, q2 = ols(X[h1]), ols(X[h2]), ols(Em[e1]), ols(Em[e2])
    idx = np.random.default_rng(0).choice(n, min(n, 300_000), replace=False)
    d = dE[s]
    res[name] = {
        "n_cells": n,
        "ESI_agree": agree(a1, a2), "ESI_chance": chance(a1, a2), "ESI_rho": float(spearmanr(a1[idx], a2[idx]).correlation),
        "EFI_agree": agree(q1, q2), "EFI_chance": chance(q1, q2), "EFI_rho": float(spearmanr(q1[idx], q2[idx]).correlation),
        "dESI_pos_pct": 100 * float((b > 0).mean()), "dEFI_pos_pct": 100 * float((d > 0).mean()), "flat_dEFI_pct": 100 * float((d == 0).mean()),
        "confident_trend_pct": 100 * float((z > 2).mean()),
        "fragile_confident_pct": 100 * float(((d > 0) & (b < 0) & (z > 2)).mean()),
        "truerec_confident_pct": 100 * float(((d > 0) & (b > 0) & (z > 2)).mean()),
        "fragile_nominal_pct": 100 * float(((d > 0) & (b <= 0)).mean()),
    }
    print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)


block(np.ones_like(m), "all")
for k, nm in TYPES.items():
    block(et == k, nm)
for k, nm in ((2, "tercile_low_NPP_reliable"), (1, "tercile_mid"), (0, "tercile_high_NPP_lowconf")):
    block(rel == k, nm)
block(rel != 0, "excluding_lowconf")
json.dump(res, open(os.path.join(OUT, "repro_by_type.json"), "w"), indent=1)
print("wrote repro_by_type.json")
