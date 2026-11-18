#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""19 -- (i) split-half reproducibility of the annual-EQI trend sign (core years: 2002-2010 vs 2012-2020), (ii) VIF of the three indices,
(iii) share of the variance of the pixel-scale EQI change carried by each component (sampled attribution table), (iv) OLS slopes of the national
means of the raw GLASS FVC/LAI/GPP before and after 2011. 3-km sample. Output: eqi_repro.json"""
import os, sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_io as gio
warnings.filterwarnings("ignore")
DEC = 6
OUT = gio.ensure_outdir(); res = {}
L = lambda n: np.load(os.path.join(OUT, n), mmap_mode="r")
def ols(Y):
    t = np.arange(Y.shape[0], dtype="float64"); t -= t.mean()
    return np.tensordot(t, Y - Y.mean(0), axes=(0, 0)) / (t ** 2).sum()
def chance(a, b):
    p, q = (a > 0).mean(), (b > 0).mean(); return 100 * float(p * q + (1 - p) * (1 - q))
def agree(a, b): return 100 * float((np.sign(a) == np.sign(b)).mean())
reg = np.asarray(L("diag_regime.npy")[::DEC, ::DEC]).astype(int)
for tag, fn, sl in (("EQI", "eqi_stack.npy", slice(2, 21)), ("EFI", "efi_stack.npy", slice(2, 21))):
    st = L(fn); S = np.stack([np.asarray(st[i, ::DEC, ::DEC]) for i in range(2, 21)])
    H = min(reg.shape[0], S.shape[1]); W = min(reg.shape[1], S.shape[2])
    ok = (reg[:H, :W] <= 3) & np.all(np.isfinite(S[:, :H, :W]), 0)
    X = S[:, :H, :W][:, ok]
    a, b = ols(X[0:9]), ols(X[10:19])          # 2002-2010 vs 2012-2020
    full = ols(X)
    res[tag] = {"n": int(ok.sum()), "agree": agree(a, b), "chance": chance(a, b), "pos_first": 100 * float((a > 0).mean()), "pos_second": 100 * float((b > 0).mean()),
                "pos_full": 100 * float((full > 0).mean())}
    print(tag, {k: round(v, 2) for k, v in res[tag].items()}, flush=True)
# VIF
e = np.asarray(L("efi_mean.npy")[::DEC, ::DEC]); s = np.asarray(L("esi.npy")[::DEC, ::DEC]); p = np.asarray(L("epi_total.npy")[::DEC, ::DEC])
H = min(e.shape[0], s.shape[0], p.shape[0]); W = min(e.shape[1], s.shape[1], p.shape[1])
ok = np.isfinite(e[:H, :W]) & np.isfinite(s[:H, :W]) & np.isfinite(p[:H, :W])
M = np.stack([e[:H, :W][ok], s[:H, :W][ok], p[:H, :W][ok]], 1)
Rm = np.corrcoef(M.T); res["VIF"] = [float(x) for x in np.diag(np.linalg.inv(Rm))]; res["corr_means"] = {"EFI_ESI": float(Rm[0, 1]), "EFI_EPI": float(Rm[0, 2]), "ESI_EPI": float(Rm[1, 2])}
print("VIF", res["VIF"], res["corr_means"])
# variance share of the pixel EQI change
d = pd.read_csv(os.path.join(OUT, "attribution_table.csv"))
q = (d["EFI_change"] + d["ESI_change"] - d["EPI_change"]) / 3
res["var_share"] = {k: float(np.cov(sg * d[k] / 3, d["EQI_change"])[0, 1] / d["EQI_change"].var()) for k, sg in (("EFI_change", 1), ("ESI_change", 1), ("EPI_change", -1))}
res["sd_change"] = {k: float(d[k].std()) for k in ("EFI_change", "ESI_change", "EPI_change", "EQI_change")}
res["eqi_change_check_corr"] = float(np.corrcoef(q, d["EQI_change"])[0, 1]); res["attr_rows"] = int(len(d))
print(res["var_share"], res["sd_change"], res["eqi_change_check_corr"])
# raw products before / after
ps = json.load(open(os.path.join(OUT, "product_step.json")))
res["products"] = {}
for k, v in ps.items():
    yrs = np.array(sorted(int(y) for y in v)); val = np.array([v[str(y)] for y in yrs])
    def sl(a, b):
        m = (yrs >= a) & (yrs <= b); return float(np.polyfit(yrs[m], val[m], 1)[0])
    res["products"][k] = {"slope_2000_2011": sl(2000, 2011), "slope_2012_2024": sl(2012, 2024), "mean_2000_2011": float(val[(yrs <= 2011)].mean()),
                          "rel_slope_2000_2011_pct": 100 * sl(2000, 2011) / float(val[(yrs <= 2011)].mean()), "rel_slope_2012_2024_pct": 100 * sl(2012, 2024) / float(val[(yrs <= 2011)].mean())}
    print(k, {a: round(b, 4) for a, b in res["products"][k].items()})
json.dump(res, open(os.path.join(OUT, "eqi_repro.json"), "w"), indent=1)
