#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""14 -- national mean of the raw GLASS FVC, LAI and GPP layers by year over the stable natural pixels (every 6th pixel),
to check whether the EFI increase after 2011 enters through a step between 2011 and 2012. Output: product_step.json"""
import os, sys, json, glob
import numpy as np, rasterio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_io as gio
DEC = 6
OUT = gio.ensure_outdir()
with rasterio.open(os.path.join(OUT, "Lu_stable.tif")) as s:
    st = s.read(1)[::DEC, ::DEC] > 0
res = {}
for var, d in (("FVC", gio.CONFIG["FVC_DIR"]), ("LAI", gio.CONFIG["LAI_DIR"]), ("GPP", gio.CONFIG["GPP_DIR"])):
    res[var] = {}
    for y in range(2000, 2025):
        fs = glob.glob(os.path.join(d, f"*_{y}_*.tif")) or glob.glob(os.path.join(d, f"*{y}*.tif"))
        if not fs:
            continue
        with rasterio.open(fs[0]) as s:
            a = s.read(1)[::DEC, ::DEC].astype("float64")
            nd = s.nodata
        h = min(a.shape[0], st.shape[0]); w = min(a.shape[1], st.shape[1])
        a = a[:h, :w]; m = st[:h, :w] & np.isfinite(a) & (a >= 0)
        if nd is not None:
            m &= a != nd
        res[var][y] = float(a[m].mean())
    print(var, {k: round(v, 3) for k, v in res[var].items()}, flush=True)
json.dump(res, open(os.path.join(OUT, "product_step.json"), "w"), indent=1)
