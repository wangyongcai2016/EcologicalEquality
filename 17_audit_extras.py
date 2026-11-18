#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17 -- Extra statistics quoted in the manuscript after the pre-submission audit (no hand-typed numbers).

  npp_zero_by_month   share of exactly-zero GLASS NPP among stable natural pixels, by calendar month
                      (every 12th pixel = 6 km, years 2012, 2016, 2020, 2024)
  forest_lowconf_by_lat  share of forest pixels in the low-confidence ESI stratum (esi_reliability == 0) per latitude band
                      (3 km sample), and its Spearman correlation with the fragile-greening share in forest per band
  mc_rank_corr        Spearman correlation of each Monte Carlo parameter with the declining share and with the mean slope
  input_fixes         size of the two input corrections (FVC 2001; LAI and GPP 2022) from the _backup_orig files
Output (OUT_DIR): audit_extras.json
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_io as gio

OUT = gio.ensure_outdir()
BANDS = ((18, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 54))
ALB = "+proj=aea +lat_0=0 +lon_0=105 +lat_1=25 +lat_2=47 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def read_dec(path, D):
    with rasterio.open(path) as s:
        return s.read(1, out_shape=(-(-s.height // D), -(-s.width // D)), resampling=Resampling.nearest)


def npp_zero_by_month(D=12, years=(2012, 2016, 2020, 2024)):
    m = gio.read_mask()[::D, ::D]
    res = {}
    for mo in range(1, 13):
        z = t = 0
        for yr in years:
            a = read_dec(gio.monthly_path("npp", yr, mo, aligned=True), D)
            h, w = min(a.shape[0], m.shape[0]), min(a.shape[1], m.shape[1])
            v = a[:h, :w][m[:h, :w]]; v = v[np.isfinite(v)]
            z += int((v == 0).sum()); t += int(v.size)
        res[mo] = 100 * z / t
    return {"by_month_pct": res, "nov_mar_pct": float(np.mean([res[k] for k in (11, 12, 1, 2, 3)])),
            "jun_aug_pct": float(np.mean([res[k] for k in (6, 7, 8)])), "years": list(years), "step_px": D}


def forest_lowconf_by_lat(D=6):
    from pyproj import Transformer
    with rasterio.open(os.path.join(OUT, "esi_reliability.tif")) as s:
        rel = s.read(1)[::D, ::D]; tf = s.transform
    with rasterio.open(os.path.join(OUT, "eco_type.tif")) as s:
        et = s.read(1)[::D, ::D]
    h, w = min(rel.shape[0], et.shape[0]), min(rel.shape[1], et.shape[1]); rel, et = rel[:h, :w], et[:h, :w]
    rr, cc = np.mgrid[0:h, 0:w]
    x = tf.c + (cc * D + 0.5) * tf.a; y = tf.f + (rr * D + 0.5) * tf.e
    _, lat = Transformer.from_crs(ALB, 4326, always_xy=True).transform(x, y)
    fo = (et == 1) & np.isin(rel, [0, 1, 2])
    out = []
    for a, b in BANDS:
        mm = fo & (lat >= a) & (lat < b)
        out.append({"from": a, "to": b, "n": int(mm.sum()), "lowconf_pct": 100 * float((rel[mm] == 0).mean())})
    mix = json.load(open(os.path.join(OUT, "robustness_summary.json"), encoding="utf-8"))["band_type_mix"]
    fg = [r["fragile_in_forest"] for r in mix]
    rho = float(spearmanr(fg, [o["lowconf_pct"] for o in out]).correlation)
    return {"bands": out, "fragile_in_forest": fg, "spearman_fragile_vs_lowconf": rho}


def mc_rank_corr():
    import csv
    rows = list(csv.DictReader(open(os.path.join(OUT, "mc_draws.csv"), encoding="utf-8-sig")))[1:]   # row 0 = base run
    P = ("q", "lambda", "human_scale", "w_efi", "w_esi", "w_epi")
    col = lambda k: np.array([float(r[k]) for r in rows])
    return {stat: {p: float(spearmanr(col(p), col(stat)).correlation) for p in P} for stat in ("decline_pct", "slope_mean")} | {"n_draws": len(rows)}


def input_fixes():
    R = lambda p: rasterio.open(glob.glob(p)[0]).read(1).astype("float64")
    out = {}
    f01o, f00, f02 = R(r"G:\FVC\_backup_orig\*2001*.tif"), R(r"G:\FVC\FVC_Align\*2000*.tif"), R(r"G:\FVC\FVC_Align\*2002*.tif")
    ok = lambda a: np.isfinite(a) & (a >= 0) & (a < 1e30)        # nodata is a large negative value
    out["FVC_2001_orig_mean"] = float(f01o[ok(f01o)].mean())
    out["FVC_2000_mean"] = float(f00[ok(f00)].mean()); out["FVC_2002_mean"] = float(f02[ok(f02)].mean())
    for key, pat in (("LAI", r"G:\LAI\{}\*2022*.tif"), ("GPP", r"G:\GPP\{}\*2022*.tif")):
        a, b = R(pat.format("_backup_orig")), R(pat.format(f"{key}_Align"))
        v = ok(a) & ok(b)
        out[f"{key}_2022_changed_pct"] = 100 * float(((a != b) & v).sum() / v.sum())
    return out


if __name__ == "__main__":
    res = {"npp_zero": npp_zero_by_month(), "forest_lowconf_by_lat": forest_lowconf_by_lat(), "mc_rank_corr": mc_rank_corr(), "input_fixes": input_fixes()}
    json.dump(res, open(os.path.join(OUT, "audit_extras.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))
