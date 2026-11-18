#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 05 — Composite EQI & weighting schemes   [no-args / VSCode-run]
================================================================================
Paper 2.2 & 2.6, Eq. 1:  EQI = W1*EFI + W2*ESI + W3*(1-EPI)
Weighting: equal / entropy / AHP-entropy. Reports VIF and agreement with equal.

HOW TO RUN: run 01_efi, 02_esi, 03_spei, 04_epi first (they write to OUT_DIR),
then run this file. Paths come from geo_io.py CONFIG['OUT_DIR'].

INPUT (from CONFIG['OUT_DIR']): efi_mean.npy, esi.npy, epi_total.npy
        for the annual EQI also efi_stack.npy, esi_stack.npy, epi_stack.npy
OUTPUT: eqi_equal/entropy/ahp_entropy .npy + .tif (if a reference .tif exists)  -- whole-period EQI
        eqi_weights.csv, 05_eqi_preview.png
        ANNUAL EQI, 2000..END_YEAR (CONFIG["EQI_EXTEND"], default True):
          eqi_YYYY.tif, eqi_stack.npy (n_years,H,W), eqi_trend.tif/.npy (Theil-Sen slope per year, all years),
          eqi_trend_core.tif/.npy (same, only the years in which all three components are observed, normally
          2002-2020), eqi_yearly_mean.csv (national mean EQI/EFI/ESI/EPI per year + extrapolation flags).
          Outside a component's record the nearest observed value is used: ESI windows are centred 2002-2022, so
          2000-2001 take the 2002 window and 2023-2024 the 2022 window; EPI needs a land-cover map (ESA-CCI
          2000-2020 only), so 2021-2024 reuse the 2020 map. EQI_EXTEND=False keeps only the observed years.
        CONFIG["EQI_YEARLY_SCHEMES"] (default ("equal",)) lists the weighting schemes to write annually; the first
        is primary, the others get a '_<scheme>' suffix. Weights are the whole-period ones, fixed over years.
        EFI is annual, EPI is annual (human-pressure part), ESI is a 5-year moving window labelled by its centre
        year, so the annual EQI changes slowly through ESI.
================================================================================
"""
from __future__ import annotations
import os, csv, warnings
import numpy as np
from scipy.stats import spearmanr
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)
RNG = np.random.default_rng(3)


def equal_weights():
    return np.array([1/3, 1/3, 1/3])


def entropy_weights(stack, mask):
    K = stack.shape[0]; X = stack.reshape(K, -1)[:, mask.reshape(-1)]
    ent = np.zeros(K)
    for k in range(K):
        col = X[k][np.isfinite(X[k])]
        if col.size == 0:
            ent[k] = 1.0; continue
        col = col - col.min() + 1e-9; p = col / col.sum()
        ent[k] = -np.sum(p * np.log(p)) / np.log(col.size)
    d = 1 - ent
    return d / d.sum() if d.sum() else equal_weights()


def ahp_weights(pairwise=None):
    if pairwise is None:
        pairwise = np.array([[1, 1.5, 1.5], [1/1.5, 1, 1], [1/1.5, 1, 1]])
    vals, vecs = np.linalg.eig(pairwise)
    w = np.abs(vecs[:, np.argmax(vals.real)].real)
    return w / w.sum()


def ahp_entropy_weights(stack, mask):
    w = ahp_weights() * entropy_weights(stack, mask)
    return w / w.sum()


def compute_eqi(efi, esi, epi, weights, mask):
    eqi = weights[0]*efi + weights[1]*esi + weights[2]*(1 - epi)
    eqi[~mask] = np.nan
    return eqi


def vif(efi, esi, epi, mask):
    X = np.column_stack([a.reshape(-1)[mask.reshape(-1)] for a in (efi, esi, epi)])
    X = X[np.all(np.isfinite(X), axis=1)]; out = []
    for k in range(3):
        y = X[:, k]; others = np.delete(X, k, axis=1)
        A = np.column_stack([others, np.ones(len(others))])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - np.sum((y - A@beta)**2) / (np.sum((y - y.mean())**2) + 1e-12)
        out.append(1 / (1 - r2 + 1e-12))
    return np.array(out)


def load_inputs():
    out = CONFIG["OUT_DIR"]
    def L(n):
        p = os.path.join(out, n)
        if not os.path.exists(p):
            raise FileNotFoundError(f"[EQI] {p} not found; run the upstream script first")
        return np.load(p)
    efi, esi, epi = L("efi_mean.npy"), L("esi.npy"), L("epi_total.npy")
    grass = np.isfinite(efi) & np.isfinite(esi) & np.isfinite(epi)
    # reference profile for GeoTIFF output, if efi_mean.tif exists
    prof = None
    tif = os.path.join(out, "efi_mean.tif")
    if gio._HAS_RIO and os.path.exists(tif):
        import rasterio
        with rasterio.open(tif) as s:
            prof = s.profile
    return efi, esi, epi, grass, prof


# ------------------------------------------------------------ annual EQI
def _theil_sen_cols(E):
    """E : (nY, n) -> Theil-Sen slope per column (NaN-aware; needs >= 3 finite years)."""
    nY, n = E.shape
    ii, jj = np.triu_indices(nY, 1)
    S = (E[jj] - E[ii]) / (jj - ii)[:, None]
    out = np.full(n, np.nan, "float32")
    full = ~np.isnan(S).any(0)
    if full.any():
        out[full] = np.median(S[:, full], axis=0)
    part = ~full & (np.isfinite(E).sum(0) >= 3)
    if part.any():
        out[part] = np.nanmedian(S[:, part], axis=0)
    return out


def stack_years(out, c):
    """Open efi_stack / esi_stack / epi_stack (memory-mapped) and return ({name: array}, {name: [years]}).

    EFI  : annual, START_YEAR .. END_YEAR                                   (01_efi_new.py)
    ESI  : 5-year moving windows, labelled by the window centre year        (02_esi.py: START_YEAR + k + ESI_WINDOW//2)
    EPI  : annual, LU_START_YEAR .. LU_END_YEAR (land-cover record)         (04_epi.py)
    Returns (None, reason) when a stack is missing."""
    names = {"efi": "efi_stack.npy", "esi": "esi_stack.npy", "epi": "epi_stack.npy"}
    arrs = {}
    for k, fn in names.items():
        p = os.path.join(out, fn)
        if not os.path.exists(p):
            return None, f"{fn} not found"
        arrs[k] = np.load(p, mmap_mode="r")
    s0, win = c["START_YEAR"], c["ESI_WINDOW"]
    years = {"efi": [s0 + i for i in range(arrs["efi"].shape[0])],
             "esi": [s0 + win // 2 + i for i in range(arrs["esi"].shape[0])],
             "epi": [c.get("LU_START_YEAR", s0) + i for i in range(arrs["epi"].shape[0])]}
    if years["efi"][-1] != c["END_YEAR"]:
        raise ValueError(f"[EQI] efi_stack has {arrs['efi'].shape[0]} years but CONFIG says "
                         f"{c['START_YEAR']}-{c['END_YEAR']}; rerun 01_efi_new.py")
    if years["epi"][-1] != c.get("LU_END_YEAR", c["END_YEAR"]):
        raise ValueError(f"[EQI] epi_stack has {arrs['epi'].shape[0]} years but CONFIG says "
                         f"{c.get('LU_START_YEAR')}-{c.get('LU_END_YEAR')}; rerun 04_epi.py")
    shp = {a.shape[1:] for a in arrs.values()}
    if len(shp) != 1:
        raise ValueError(f"[EQI] annual stacks differ in grid shape: { {k: a.shape for k, a in arrs.items()} }")
    return arrs, years


def yearly_eqi(out, c, schemes, mask, prof, rows=128):
    """Annual EQI = w1*EFI_t + w2*ESI_t + w3*(1-EPI_t) for the years all three components cover.

    * weights are the whole-period weights of each scheme, FIXED for all years (so years are comparable);
    * mask is the whole-period analysis mask (stable natural pixels); a pixel-year is NaN when any component is;
    * ESI_t is the 5-year window centred on t, i.e. a slowly varying quantity, not an independent annual value;
    * EPI_t varies by year through the human-pressure part only (the drought part is one value per pixel).
    The first scheme in `schemes` is the primary one (eqi_YYYY.tif, eqi_stack.npy, eqi_trend); the others get a
    '_<scheme>' suffix. Also writes eqi_yearly_mean.csv (national mean per year and component)."""
    arrs, years = stack_years(out, c)
    if arrs is None:
        print(f"[EQI] annual EQI skipped: {years}. Run 01_efi_new, 02_esi and 04_epi first.")
        return
    core = sorted(set(years["efi"]) & set(years["esi"]) & set(years["epi"]))     # all three observed
    if len(core) < 3:
        print(f"[EQI] annual EQI skipped: only {len(core)} common year(s) "
              f"(EFI {years['efi'][0]}-{years['efi'][-1]}, ESI {years['esi'][0]}-{years['esi'][-1]}, "
              f"EPI {years['epi'][0]}-{years['epi'][-1]})")
        return
    extend = bool(c.get("EQI_EXTEND", True))
    common = list(years["efi"]) if extend else core          # EFI is the longest record: 2000..END_YEAR

    def _pick(k, y):
        """Index of year y in component k; outside its record the nearest observed year is used."""
        yk = years[k]
        return yk.index(min(max(y, yk[0]), yk[-1]))
    ix = {k: [_pick(k, y) for y in common] for k in arrs}
    esi_ext = [not (years["esi"][0] <= y <= years["esi"][-1]) for y in common]
    epi_ext = [not (years["epi"][0] <= y <= years["epi"][-1]) for y in common]
    core_i = [i for i in range(len(common)) if not esi_ext[i] and not epi_ext[i]]
    nY = len(common); H, W = mask.shape
    print(f"[EQI] annual EQI {common[0]}-{common[-1]} ({nY} years; EFI {years['efi'][0]}-{years['efi'][-1]}, "
          f"ESI windows centred {years['esi'][0]}-{years['esi'][-1]}, EPI {years['epi'][0]}-{years['epi'][-1]})")
    if extend:
        print(f"[EQI]   extrapolated: ESI = nearest window in {[y for y, e in zip(common, esi_ext) if e]}, "
              f"EPI = {years['epi'][-1]} map in {[y for y, e in zip(common, epi_ext) if e]}; "
              f"core (all observed) {common[core_i[0]]}-{common[core_i[-1]]}")
    names = list(schemes)
    primary = names[0]
    suffix = {n: ("" if n == primary else f"_{n}") for n in names}
    stacks = {n: np.lib.format.open_memmap(os.path.join(out, f"eqi_stack{suffix[n]}.npy"), mode="w+",
                                           dtype="float32", shape=(nY, H, W)) for n in names}
    trend = np.full((H, W), np.nan, "float32")           # all years (incl. extrapolated components)
    trend_core = np.full((H, W), np.nan, "float32")      # only years in which ESI and EPI are observed
    sums = {k: np.zeros(nY) for k in ("eqi", "efi", "esi", "epi")}
    cnt = np.zeros(nY)
    # fixed pixel set: pixels with a valid EQI in EVERY year, so the national mean is not driven by a changing
    # pixel population (e.g. pixels whose ESI only exists from the 2012 window on)
    sums_fx = {k: np.zeros(nY) for k in ("eqi", "efi", "esi", "epi")}
    n_fx = 0
    fixed_mask = np.zeros((H, W), bool)
    for r0 in range(0, H, rows):
        r1 = min(H, r0 + rows)
        m = mask[r0:r1]
        if not m.any():
            for n in names:
                stacks[n][:, r0:r1] = np.nan
            continue
        efi = np.asarray(arrs["efi"][ix["efi"], r0:r1], "float32")
        esi = np.asarray(arrs["esi"][ix["esi"], r0:r1], "float32")
        epi = np.asarray(arrs["epi"][ix["epi"], r0:r1], "float32")
        ok = m[None] & np.isfinite(efi) & np.isfinite(esi) & np.isfinite(epi)
        for n in names:
            w = schemes[n]
            E = (w[0] * efi + w[1] * esi + w[2] * (1 - epi)).astype("float32")
            E[~ok] = np.nan
            stacks[n][:, r0:r1] = E
            if n == primary:
                v = ok.any(0)
                if v.any():
                    t = np.full(v.shape, np.nan, "float32")
                    t[v] = _theil_sen_cols(E[:, v])
                    trend[r0:r1] = t
                    if extend:
                        tc = np.full(v.shape, np.nan, "float32")
                        tc[v] = _theil_sen_cols(E[core_i][:, v])
                        trend_core[r0:r1] = tc
                for key, a in (("eqi", E), ("efi", efi), ("esi", esi), ("epi", epi)):
                    sums[key] += np.where(ok, a, 0.0).sum(axis=(1, 2), dtype="float64")
                cnt += ok.sum(axis=(1, 2))
                fx = ok.all(0)
                fixed_mask[r0:r1] = fx
                if fx.any():
                    for key, a in (("eqi", E), ("efi", efi), ("esi", esi), ("epi", epi)):
                        sums_fx[key] += np.where(fx[None], a, 0.0).sum(axis=(1, 2), dtype="float64")
                    n_fx += int(fx.sum())
    for n in names:
        stacks[n].flush()
    np.save(os.path.join(out, "eqi_trend.npy"), trend)
    if extend:
        np.save(os.path.join(out, "eqi_trend_core.npy"), trend_core)
    with open(os.path.join(out, "eqi_yearly_mean.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["year", "n_pixels", "EQI", "EFI", "ESI", "EPI", "ESI_extrapolated", "EPI_extrapolated",
                     "n_fixed", "EQI_fixed", "EFI_fixed", "ESI_fixed", "EPI_fixed"])
        for i, y in enumerate(common):
            d = max(cnt[i], 1)
            wr.writerow([y, int(cnt[i])] + [f"{sums[k][i] / d:.4f}" for k in ("eqi", "efi", "esi", "epi")]
                        + [int(esi_ext[i]), int(epi_ext[i]), n_fx]
                        + [f"{sums_fx[k][i] / max(n_fx, 1):.4f}" for k in ("eqi", "efi", "esi", "epi")])
    print(f"[EQI] fixed pixel set (valid in all {nY} years): {n_fx:,} px "
          f"({100 * n_fx / max(int(mask.sum()), 1):.1f}% of the analysis mask); "
          f"EQI_fixed {sums_fx['eqi'][0] / max(n_fx, 1):.3f} ({common[0]}) -> "
          f"{sums_fx['eqi'][-1] / max(n_fx, 1):.3f} ({common[-1]})")
    if prof is not None:
        gio.write_tif(os.path.join(out, "eqi_fixed_mask.tif"), fixed_mask.astype("uint8"), prof, "uint8", 0)
    for i, y in enumerate(common):
        if i in (0, nY // 2, nY - 1):
            print(f"[EQI]   {y}: EQI {sums['eqi'][i] / max(cnt[i], 1):.3f}  (valid px {int(cnt[i]):,})")
    for lbl, tr in (("all years", trend), ("core", trend_core)):
        tv = tr[np.isfinite(tr)]
        if tv.size:
            print(f"[EQI] EQI trend ({primary}, {lbl}): mean {tv.mean():+.4f}/yr, "
                  f"{100 * (tv > 0).mean():.1f}% pixels increasing")
    if prof is not None:
        for n in names:
            for i, y in enumerate(common):
                gio.write_tif(os.path.join(out, f"eqi_{y}{suffix[n]}.tif"), np.asarray(stacks[n][i]), prof)
        gio.write_tif(os.path.join(out, "eqi_trend.tif"), trend, prof)
        if extend:
            gio.write_tif(os.path.join(out, "eqi_trend_core.tif"), trend_core, prof)
        print(f"[EQI] wrote eqi_{common[0]}..{common[-1]}.tif ({', '.join(names)}), eqi_trend.tif"
              f"{', eqi_trend_core.tif' if extend else ''}, eqi_stack.npy, eqi_yearly_mean.csv")
    else:
        print("[EQI] wrote eqi_stack.npy, eqi_trend.npy, eqi_yearly_mean.csv (no reference GeoTIFF: no .tif)")


def generate_demo():
    H = W = 40; yy, xx = np.mgrid[0:H, 0:W]; grass = np.ones((H, W), bool)
    base = 0.45 + 0.25 * RNG.random((H, W))
    efi = np.clip(base + 0.1*RNG.standard_normal((H, W)), 0, 1)
    esi = np.clip(0.7 - 0.3*((xx >= 2*W//3) & (yy >= H//2)) + 0.08*RNG.standard_normal((H, W)), 0, 1)
    epi = np.clip(0.2 + 0.1*RNG.random((H, W)), 0, 1)
    return efi, esi, epi, grass, None


def main():
    c = CONFIG; out = gio.ensure_outdir()
    demo = False
    try:
        efi, esi, epi, grass, prof = load_inputs()
    except FileNotFoundError as e:
        if c["ALLOW_DEMO"]:
            print(f"{e}\n[EQI] running synthetic demo")
            efi, esi, epi, grass, prof = generate_demo()
            demo = True
        else:
            raise
    print(f"[EQI] grid {grass.shape}, grass px {int(grass.sum())}")

    v = vif(efi, esi, epi, grass)
    print(f"[EQI] VIF (EFI,ESI,EPI) = {np.round(v,2)} (require all < 5)")

    stack = np.stack([efi, esi, 1 - epi], 0)
    schemes = {"equal": equal_weights(), "entropy": entropy_weights(stack, grass),
               "ahp_entropy": ahp_entropy_weights(stack, grass)}
    fields = {}
    for name, w in schemes.items():
        fields[name] = compute_eqi(efi, esi, epi, w, grass)
        np.save(os.path.join(out, f"eqi_{name}.npy"), fields[name].astype(np.float32))
        if prof is not None:
            gio.write_tif(os.path.join(out, f"eqi_{name}.tif"), fields[name], prof)

    base = fields["equal"][grass]; rows = []
    for name, w in schemes.items():
        r = np.corrcoef(base, fields[name][grass])[0, 1]
        rho = spearmanr(base, fields[name][grass]).correlation
        rows.append([name, *[f"{x:.3f}" for x in w], f"{r:.3f}", f"{rho:.3f}"])
        print(f"[EQI] {name:12s} w={np.round(w,3)} mean={np.nanmean(fields[name]):.3f} r={r:.3f} rho={rho:.3f}")
    with open(os.path.join(out, "eqi_weights.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["scheme","W_EFI","W_ESI","W_EPI","r_vs_equal","rho_vs_equal"])
        wr.writerows(rows)
    if prof is not None:
        print("[EQI] wrote GeoTIFFs: eqi_equal.tif, eqi_entropy.tif, eqi_ahp_entropy.tif")

    # annual EQI from the annual stacks (skipped when a stack is missing, e.g. demo mode)
    yearly = [s for s in c.get("EQI_YEARLY_SCHEMES", ("equal",)) if s in schemes]
    if yearly and not demo:
        yearly_eqi(out, c, {s: schemes[s] for s in yearly}, grass, prof)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
        for a, name in zip(ax, ["equal", "entropy", "ahp_entropy"]):
            im = a.imshow(fields[name], cmap="RdYlGn", vmin=0, vmax=1); a.set_title(f"EQI ({name})")
            plt.colorbar(im, ax=a, shrink=.8)
        plt.tight_layout(); plt.savefig(os.path.join(out, "05_eqi_preview.png"), dpi=120)
        print(f"[EQI] preview -> {out}/05_eqi_preview.png")
    except Exception as e:
        print(f"[EQI] (preview skipped: {e})")
    print("[EQI] done.")


if __name__ == "__main__":
    main()
