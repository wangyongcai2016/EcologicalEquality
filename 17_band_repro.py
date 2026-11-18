#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""17 -- Is the latitudinal / longitudinal pattern of the dESI sign reproducible between the two non-overlapping halves of the record?
Share of pixels with a negative dESI (OLS slope) per 5-degree latitude band and 10-degree longitude band, computed separately from the ESI windows
1-8 (data 2000-2011) and 14-21 (data 2013-2024); Pearson/Spearman correlation of the band values between halves. 3-km sample.
Output: band_repro.json"""
from __future__ import annotations
import os, sys, json, glob, warnings
import numpy as np, rasterio
from scipy.stats import spearmanr, pearsonr
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_io as gio
warnings.filterwarnings("ignore")
DEC = 6
OUT = gio.ensure_outdir()
L = lambda n: np.load(os.path.join(OUT, n), mmap_mode="r")
def dec(p, dt="float32"):
    with rasterio.open(p) as s:
        return s.read(1)[::DEC, ::DEC].astype(dt)
def ols(Y):
    t = np.arange(Y.shape[0], dtype="float64"); t -= t.mean()
    return np.tensordot(t, Y - Y.mean(0), axes=(0, 0)) / (t ** 2).sum()
reg = np.asarray(L("diag_regime.npy")[::DEC, ::DEC]).astype(int)
A = np.stack([dec(f) for f in sorted(glob.glob(os.path.join(OUT, "esi_window_20*.tif")))]); A[(A < -1) | (A > 2)] = np.nan
et = dec(os.path.join(OUT, "eco_type.tif"), "int16")
with rasterio.open(os.path.join(OUT, "eco_type.tif")) as s:
    TR = s.transform
H = min(reg.shape[0], A.shape[1]); W = min(reg.shape[1], A.shape[2])
reg, A, et = reg[:H, :W], A[:, :H, :W], et[:H, :W]
m = (reg >= 0) & (reg <= 3) & np.all(np.isfinite(A), 0)
rr, cc = np.nonzero(m)
xs = TR.c + (cc * DEC + 0.5) * TR.a; ys = TR.f + (rr * DEC + 0.5) * TR.e
lon, lat = Transformer.from_crs("+proj=aea +lat_0=0 +lon_0=105 +lat_1=25 +lat_2=47 +datum=WGS84 +units=m +no_defs", 4326, always_xy=True).transform(xs, ys)
X = A[:, m]; b1 = ols(X[0:8]); b2 = ols(X[13:21]); bf = ols(X); tp = et[m]
res = {}
for kind, v, edges in (("lat", lat, np.arange(20, 55, 5)), ("lon", lon, np.arange(75, 135, 10))):
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (v >= a) & (v < b)
        if s.sum() < 3000: continue
        rows.append((int(a), int(b), int(s.sum()), 100 * float((b1[s] < 0).mean()), 100 * float((b2[s] < 0).mean()), 100 * float((bf[s] < 0).mean()),
                     float(b1[s].mean()), float(b2[s].mean())))
    r = np.array(rows)
    res[kind] = {"bands": rows, "n_bands": len(rows),
                 "pearson_neg_share": float(pearsonr(r[:, 3], r[:, 4])[0]), "spearman_neg_share": float(spearmanr(r[:, 3], r[:, 4]).correlation),
                 "pearson_mean_slope": float(pearsonr(r[:, 6], r[:, 7])[0])}
    print(kind, [(x[0], x[1], round(x[3], 1), round(x[4], 1)) for x in rows], {k: round(v, 2) for k, v in res[kind].items() if k.startswith("p") or k.startswith("s")})
json.dump(res, open(os.path.join(OUT, "band_repro.json"), "w"), indent=1)
