#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16 -- Sensitivity of the regime shares to the period of dEFI, and the 2020 EFI anomaly used in the benchmarking section.
dESI is the pipeline value (Theil-Sen of the 21 windows centred 2002-2022, data 2000-2024). dEFI is recomputed here as an OLS slope of the annual
EFI over (a) 2000-2020 (pipeline period), (b) 2002-2022 (aligned with the window centres), (c) 2000-2024 (all data years), on the 3 km sample
(every 6th pixel) of pixels with a complete series in all three periods.  Output (OUT_DIR): variants_period.json
"""
from __future__ import annotations
import os, sys, json, csv, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_io as gio

warnings.filterwarnings("ignore")
DEC = 6
OUT = gio.ensure_outdir()


def L(n):
    return np.load(os.path.join(OUT, n), mmap_mode="r")


def ols(Y):
    t = np.arange(Y.shape[0], dtype="float64"); t -= t.mean()
    return np.tensordot(t, Y - Y.mean(0), axes=(0, 0)) / (t ** 2).sum()


reg = np.asarray(L("diag_regime.npy")[::DEC, ::DEC]).astype(int)
dS = np.asarray(L("diag_dESI.npy")[::DEC, ::DEC])
ef = L("efi_stack.npy")
E = np.stack([np.asarray(ef[i, ::DEC, ::DEC]) for i in range(ef.shape[0])])
ok = (reg >= 0) & (reg <= 3) & np.isfinite(dS) & np.all(np.isfinite(E), axis=0)
print("complete cells", int(ok.sum()), "of", int(((reg >= 0) & (reg <= 3)).sum()), flush=True)
s = dS[ok]; X = E[:, ok]
res = {"n_cells": int(ok.sum()), "n_regime_valid": int(((reg >= 0) & (reg <= 3)).sum())}
for lab, (a, b) in (("2000-2020", (0, 21)), ("2002-2022", (2, 23)), ("2000-2024", (0, 25))):
    d = ols(X[a:b]); pos = d > 0; sp = s > 0
    res[lab] = {"dEFI_pos_pct": 100 * float(pos.mean()), "dESI_pos_pct": 100 * float(sp.mean()),
                "true_recovery_pct": 100 * float((pos & sp).mean()), "fragile_greening_pct": 100 * float((pos & ~sp).mean()),
                "low_but_stable_pct": 100 * float((~pos & sp).mean()), "collapse_pct": 100 * float((~pos & ~sp).mean()),
                "expected_fragile_under_independence_pct": 100 * float(pos.mean() * (1 - sp.mean())),
                "flat_pct": 100 * float((d == 0).mean())}
    print(lab, {k: round(v, 2) for k, v in res[lab].items()}, flush=True)
# 2020 EFI anomaly of the national mean (all valid pixels, eqi_yearly_mean.csv)
rows = list(csv.DictReader(open(os.path.join(OUT, "eqi_yearly_mean.csv"), encoding="utf-8-sig")))
efi = {int(r["year"]): float(r["EFI"]) for r in rows}
m0020 = float(np.mean([efi[y] for y in range(2000, 2021)]))
res["EFI_2020_vs_mean_2000_2020_pct"] = 100 * (efi[2020] / m0020 - 1)
res["EFI_2020"] = efi[2020]; res["EFI_mean_2000_2020"] = m0020
print("EFI 2020 anomaly %", res["EFI_2020_vs_mean_2000_2020_pct"])
json.dump(res, open(os.path.join(OUT, "variants_period.json"), "w"), indent=1)
