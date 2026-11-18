#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""18 -- Census (every pixel) of the regime shares against the share expected if dEFI and dESI were independent, by ecosystem type,
with and without the pixels whose dEFI is exactly zero. Output: null_by_type.json"""
import os, sys, json, warnings
import numpy as np, rasterio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_io as gio
warnings.filterwarnings("ignore")
OUT = gio.ensure_outdir()
L = lambda n: np.asarray(np.load(os.path.join(OUT, n), mmap_mode="r"))
dE, dS, reg = L("diag_dEFI.npy"), L("diag_dESI.npy"), L("diag_regime.npy")
with rasterio.open(os.path.join(OUT, "eco_type.tif")) as s:
    et = s.read(1)
with rasterio.open(os.path.join(OUT, "esi_reliability.tif")) as s:
    rel = s.read(1)
h = min(dE.shape[0], et.shape[0], rel.shape[0]); w = min(dE.shape[1], et.shape[1], rel.shape[1])
dE, dS, reg, et, rel = dE[:h, :w], dS[:h, :w], reg[:h, :w], et[:h, :w], rel[:h, :w]
valid = (reg <= 3) & np.isfinite(dE) & np.isfinite(dS)
NM = {0: "unclassified", 1: "forest", 2: "shrub", 3: "grassland", 4: "wetland", 5: "sparse"}
res = {}
def block(m, name):
    d = {}
    for tag, mm in (("all", m), ("noflat", m & (dE != 0))):
        n = int(mm.sum())
        pE = float((dE[mm] > 0).mean()); pS = float((dS[mm] > 0).mean())
        obs = {k: 100 * float((reg[mm] == i).mean()) for i, k in enumerate(["TR", "FG", "LS", "CO"])}
        exp = {"TR": 100 * pE * pS, "FG": 100 * pE * (1 - pS), "LS": 100 * (1 - pE) * pS, "CO": 100 * (1 - pE) * (1 - pS)}
        d[tag] = {"n": n, "dEFI_pos_pct": 100 * pE, "dESI_pos_pct": 100 * pS, "obs": obs, "exp": exp}
    res[name] = d
    print(name, {t: (round(d[t]["obs"]["FG"], 2), round(d[t]["exp"]["FG"], 2)) for t in d}, flush=True)
block(valid, "all")
for k, nm in NM.items():
    block(valid & (et == k), nm)
block(valid & (rel != 0), "excl_lowconf_all")
json.dump(res, open(os.path.join(OUT, "null_by_type.json"), "w"), indent=1)
