#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 04 — Ecological Pressure Index (EPI): human sources + drought
      [annual ESA-CCI land cover 2000-2020, 300 m -> 500 m | per-year EPI | memmap + tiled]
================================================================================
Paper section 2.5, Eqs. 8-10.  Pressure sources and sensitivities come from the coefficient workbook
(CONFIG["EPI_COEF_XLSX"], default input/系数表.xlsx; InVEST habitat-quality layout):

    Sheet1  threat table      ThreadFactors | ID | Weight | Maximum Distance of Influence/km | Type of Decay over Space
    Sheet2  sensitivity table Landuse | Chinese | Code | HabitatSuitability | <one column per threat>

    SI(x,y) = A * sum_i  w_i * D_i(x,y) * S_{c(x,y), i}               (human pressure, A = global min-max coefficient)
    D_i = exp(-2.99 * d / d_max)  (Exponential, d <= d_max)   or   1 - d / d_max  (Linear, d <= d_max)   [InVEST]
    EPI = lambda * SI_norm + (1 - lambda) * P_drought,  lambda = 0.6   (P_drought from 03_spei.py)

d = distance (km) from the pixel to the nearest source pixel of threat i, S_{c,i} = Sheet2 value of the pixel's ESA-CCI
class c for threat i. The weights are used as given; the single global min-max constant A absorbs InVEST's w_i / sum(w).
HabitatSuitability (Sheet2) belongs to the InVEST habitat-quality score and is NOT used by the EPI.

SOURCES: every threat in Sheet1 is an ESA-CCI class group, so all of them are extracted from the annual land cover.
The sheet gives one representative ID per threat; the codes that make up each threat are geo_io.ESACCI_SOURCE_CODES
(author decision -- check it):
    Urbanland          190              (Sheet1 ID 190)
    Cropland           10, 11, 12       (rain-fed cropland and its sub-classes; ID 10)
    IrrigatedCropland  20               (irrigated cropland; ID 20)
    Other              30, 40           (cropland / natural-vegetation mosaics; ID 30)
Each ID must lie inside its group, and the groups must not overlap.

DESERT AND BARE LAND ARE NEVER PRESSURE SOURCES. ESA-CCI 200-202 (bare areas) and 140, 150-153 (lichens/mosses, sparse
vegetation) are listed in geo_io.ESACCI_NON_SOURCE_CODES; source_masks_from_codes raises if a source group contains
them. Bare land is also not an assessed ecosystem (not in ESACCI_NATURAL_CODES), so it gets no EPI value.

ASSESSED PIXELS: ESA-CCI natural codes (ESACCI_NATURAL_CODES) inside an assessment unit. Every natural code must have a
row in Sheet2, otherwise the run stops.

--------------------------------------------------------------------------------
DATA INPUT  (paths set ONCE in geo_io.py CONFIG; NO command-line args)
--------------------------------------------------------------------------------
  CONFIG['EPI_COEF_XLSX']                                                         coefficient workbook
  CONFIG['LANDUSE_DIR'], ['LU_NAME'], ['LU_PATTERN'], ['LU_START_YEAR'], ['LU_END_YEAR']  annual ESA-CCI (300 m)
  CONFIG['PIXEL_KM'] = 0.5
  drought_pressure.tif/.npy from 03_spei.py in OUT_DIR (static field, blended into every year).

OUTPUTS (CONFIG['OUT_DIR'])
    epi_stack.npy            (n_years, H, W) float32 annual EPI (NaN outside assessed natural pixels)
    epi_total.tif/.npy       time-mean EPI  [read by 05_eqi.py, 06_trends.py]
    epi_human.npy, epi_human_mean.tif   time-mean normalised human pressure [read by 11_monte_carlo.py]
    epi_trend.tif/.npy       Theil-Sen slope of annual EPI (dEPI per year)
    epi_YYYY.tif             per-year EPI (if WRITE_YEARLY_TIFS)
    epi_sources.csv, epi_sensitivity_mapping.csv (the values actually used), 04_epi_preview.png

Run: press ▶ in VSCode (edit geo_io.py CONFIG first), or  python 04_epi.py
Pilot: EQI_ROI / EQI_OUT_DIR environment variables, as for 02_esi.py.
================================================================================
"""
from __future__ import annotations
import os, csv, glob, time, warnings
import numpy as np
import geo_io as gio
from geo_io import CONFIG, ESACCI_NATURAL_CODES, ESACCI_SOURCE_CODES, ESACCI_NON_SOURCE_CODES

warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.ndimage import distance_transform_edt
RNG = np.random.default_rng(5)

LAMBDA = 0.6
BLOCK_ROWS = 128               # rows per tile in the second pass
MIN_YEARS_TREND = 5
WRITE_YEARLY_TIFS = True
KEEP_RAW = False               # keep the temporary un-normalised human-pressure stack (about 6-7 GB nationally)


# ------------------------------------------------------------ coefficient workbook
def _decay_kind(text):
    t = str(text).strip().lower()
    if t.startswith("exp"):
        return "exp"
    if t.startswith("lin"):
        return "lin"
    raise ValueError(f"[EPI] unknown decay type '{text}' in Sheet1 (use Exponential or Linear)")


def _rows(ws):
    return [r for r in ws.iter_rows(values_only=True) if r and any(c is not None for c in r)]


def load_coefficients(path=None):
    """Read Sheet1 (threats) and Sheet2 (sensitivities). Returns a dict:
       names, w, dmax, kind, ids   per threat (Sheet1 order)
       lut     (256, n) sensitivity by ESA-CCI code
       valid   (256,)   assessed natural codes
       landuse {code: Landuse name}
    """
    import openpyxl
    path = path or CONFIG["EPI_COEF_XLSX"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"[EPI] coefficient workbook not found: {path} (CONFIG['EPI_COEF_XLSX'])")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        ws1 = wb["Sheet1"] if "Sheet1" in wb.sheetnames else wb.worksheets[0]
        ws2 = wb["Sheet2"] if "Sheet2" in wb.sheetnames else wb.worksheets[1]
        r1, r2 = _rows(ws1), _rows(ws2)
    finally:
        wb.close()

    # ---- Sheet1: threat | ID | weight | d_max (km) | decay
    names, ids, w, dmax, kind = [], [], [], [], []
    for r in r1[1:]:
        name = str(r[0]).strip()
        if name in names:
            raise ValueError(f"[EPI] Sheet1 lists threat '{name}' twice")
        names.append(name); ids.append(int(r[1])); w.append(float(r[2])); dmax.append(float(r[3]))
        kind.append(_decay_kind(r[4]))
    if not names:
        raise ValueError("[EPI] Sheet1 has no threat rows")
    unknown = [n for n in names if n not in ESACCI_SOURCE_CODES]
    if unknown:
        raise ValueError(f"[EPI] Sheet1 threats {unknown} have no ESA-CCI code group; add them to "
                         f"geo_io.ESACCI_SOURCE_CODES (known: {sorted(ESACCI_SOURCE_CODES)})")
    for n, i in zip(names, ids):
        if i not in ESACCI_SOURCE_CODES[n]:
            raise ValueError(f"[EPI] Sheet1 ID {i} of '{n}' is not in its ESA-CCI group {sorted(ESACCI_SOURCE_CODES[n])}")
    if min(w) <= 0 or min(dmax) <= 0:
        raise ValueError("[EPI] Sheet1 weights and maximum distances must be > 0")
    seen = {}
    for n in names:
        for c in ESACCI_SOURCE_CODES[n]:
            if c in seen:
                raise ValueError(f"[EPI] ESA-CCI code {c} belongs to both '{seen[c]}' and '{n}'")
            seen[c] = n

    # ---- Sheet2: Landuse | Chinese | Code | HabitatSuitability | one column per threat
    head = [str(c).strip().lower() if c is not None else "" for c in r2[0]]
    if "code" not in head:
        raise ValueError("[EPI] Sheet2 has no 'Code' column")
    ic = head.index("code")
    icol = []
    for n in names:
        if n.lower() not in head:
            raise ValueError(f"[EPI] Sheet2 has no sensitivity column for threat '{n}'")
        icol.append(head.index(n.lower()))
    lut = np.zeros((256, len(names)), "float32")
    have = set()
    landuse = {}
    for r in r2[1:]:
        if r[ic] is None:
            continue
        code = int(r[ic])
        if not 0 <= code < 256:
            raise ValueError(f"[EPI] Sheet2 code {code} outside 0-255")
        if code in have:
            raise ValueError(f"[EPI] Sheet2 lists ESA-CCI code {code} twice")
        have.add(code)
        vals = [float(r[j]) for j in icol]
        if min(vals) < 0 or max(vals) > 1:
            raise ValueError(f"[EPI] Sheet2 sensitivities for code {code} must lie in [0, 1]: {vals}")
        lut[code] = vals
        landuse[code] = str(r[0]).strip() if r[0] is not None else ""
    missing = sorted(set(ESACCI_NATURAL_CODES) - have)
    if missing:
        raise ValueError(f"[EPI] Sheet2 lacks sensitivities for natural ESA-CCI codes {missing}")
    valid = np.zeros(256, bool)
    valid[list(ESACCI_NATURAL_CODES)] = True
    return dict(names=names, ids=ids, w=w, dmax=dmax, kind=kind, lut=lut, valid=valid, landuse=landuse)


# ------------------------------------------------------------ distance decay and pressure
def decay(dist, d_max, kind):
    """InVEST: exp(-2.99 d / d_max) or 1 - d / d_max; zero beyond d_max."""
    dist = np.asarray(dist, "float32")
    if kind == "exp":
        out = np.exp(np.float32(-2.99 / d_max) * dist)
    else:
        out = 1.0 - dist / np.float32(d_max)
    out[dist > d_max] = 0.0
    return np.maximum(out, 0.0).astype("float32")


def decay_field(mask, d_max, kind, pixel_km):
    """Influence field of one source mask (True = source pixel) on the grid; None when there is no source."""
    if not mask.any():
        return None
    dist = distance_transform_edt(~mask) * pixel_km
    return decay(dist, d_max, kind)


def human_year(code, coef, pixel_km, inunit=None):
    """Raw (un-normalised) human pressure of one year: sum_i w_i * S_{c,i} * D_i, NaN outside assessed natural pixels."""
    code = np.asarray(code, "uint8")
    groups = {n: ESACCI_SOURCE_CODES[n] for n in coef["names"]}
    masks, _, _ = gio.source_masks_from_codes(code, groups)   # raises if a group contains bare land / desert
    total = np.zeros(code.shape, "float32")
    for k, n in enumerate(coef["names"]):
        dec = decay_field(masks[n], coef["dmax"][k], coef["kind"][k], pixel_km)
        if dec is None:
            continue
        total += np.float32(coef["w"][k]) * coef["lut"][code, k] * dec
    total[~coef["valid"][code]] = np.nan
    if inunit is not None:
        total[~inunit] = np.nan
    return total


# ------------------------------------------------------------ trend (vectorised Theil-Sen, columns = pixels)
def theil_sen_cols(E, chunk=100_000):
    """E : (n, m) -> median of pairwise slopes per column (NaN-aware); NaN where fewer than MIN_YEARS_TREND values."""
    n, m = E.shape
    ii, jj = np.triu_indices(n, 1)
    dt = (jj - ii).astype("float32")[:, None]
    out = np.full(m, np.nan, "float32")
    for s in range(0, m, chunk):
        sl = slice(s, min(m, s + chunk))
        D = (E[jj, sl] - E[ii, sl]) / dt
        fin = np.isfinite(D)
        c = fin.sum(0)
        S = np.sort(np.where(fin, D, np.inf), axis=0)
        lo_i = np.maximum((c - 1) // 2, 0)[None]
        hi_i = (c // 2)[None]
        hi_i = np.minimum(hi_i, S.shape[0] - 1)
        med = 0.5 * (np.take_along_axis(S, lo_i, 0)[0] + np.take_along_axis(S, hi_i, 0)[0])
        nv = np.isfinite(E[:, sl]).sum(0)
        med[(c == 0) | (nv < MIN_YEARS_TREND)] = np.nan
        out[sl] = med
    return out


# ------------------------------------------------------------ second-pass tile worker
def epi_block(r0, r1, human_path, epi_path, drought_blk, lo, hi, lam):
    h = np.asarray(np.load(human_path, mmap_mode="r")[:, r0:r1, :], "float32")
    n, b, W = h.shape
    hn = np.clip((h - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    epi = (lam * hn + (1 - lam) * drought_blk[None]).astype("float32")
    epi[~np.isfinite(h)] = np.nan
    mm = np.load(epi_path, mmap_mode="r+")
    mm[:, r0:r1, :] = epi
    mm.flush()
    del mm
    epi_mean = np.nanmean(epi, axis=0)
    hum_mean = np.nanmean(hn, axis=0)
    flat = epi.reshape(n, -1)
    ok = np.isfinite(flat).sum(0) >= MIN_YEARS_TREND
    idx = np.flatnonzero(ok)
    trend = np.full(b * W, np.nan, "float32")
    if idx.size:
        trend[idx] = theil_sen_cols(flat[:, idx])
    return r0, r1, epi_mean.astype("float32"), hum_mean.astype("float32"), trend.reshape(b, W)


# ------------------------------------------------------------ loaders
def lu_path(year):
    c = CONFIG
    p = os.path.join(c["LANDUSE_DIR"], c["LU_PATTERN"].format(var=c["LU_NAME"], year=year))
    if os.path.exists(p):
        return p
    cand = glob.glob(os.path.join(c["LANDUSE_DIR"], f"*{c['LU_NAME']}*{year}*.tif"))
    if not cand:
        raise FileNotFoundError(f"[EPI] missing land cover for {year}: {p}")
    return sorted(cand)[0]


def load_drought(out_dir, shape):
    """Drought pressure from 03_spei outputs (GeoTIFF or .npy), NaN -> 0; zeros when absent (human pressure only)."""
    tif = os.path.join(out_dir, "drought_pressure.tif")
    npy = os.path.join(out_dir, "drought_pressure.npy")
    d = None
    if os.path.exists(tif):
        d, _ = gio.read_tif(tif)
    elif os.path.exists(npy):
        d = np.load(npy)
    if d is None:
        print("[EPI] no drought_pressure from 03_spei.py; EPI = lambda * human pressure only (drought term 0)")
        return np.zeros(shape, "float32"), False
    if d.shape != tuple(shape):
        raise ValueError(f"[EPI] drought_pressure {d.shape} differs from the analysis grid {tuple(shape)}; "
                         "rerun 03_spei.py with the same ROI")
    return np.clip(np.nan_to_num(d, nan=0.0), 0, 1).astype("float32"), True


def generate_demo():
    """Synthetic annual land cover: forest / grassland with growing urban and cropland, and a bare-land patch
    (which must exert no pressure)."""
    c = CONFIG
    H = W = 80
    yy, xx = np.mgrid[0:H, 0:W]
    years = list(range(c.get("LU_START_YEAR", 2000), c.get("LU_END_YEAR", 2020) + 1))
    codes = []
    for t, yr in enumerate(years):
        lc = np.full((H, W), 130, "uint8")
        lc[:, : W // 3] = 61
        lc[(yy > 60) & (xx > 50)] = 200                       # bare land
        r = 2 + t // 3
        lc[10:10 + r, 10:10 + r] = 190
        lc[H - 20 - r:H - 20, 40:40 + r] = 10
        codes.append(lc)
    drought = np.clip(0.15 + 0.4 * ((xx >= 2 * W // 3) & (yy >= H // 2)) + 0.05 * RNG.standard_normal((H, W)), 0, 1)
    return codes, years, drought.astype("float32")


# ------------------------------------------------------------ pipeline
def run_pipeline(get_code, years, shape, inunit, drought, drought_real, coef, out, prof, keep_raw=KEEP_RAW):
    from joblib import Parallel, delayed
    c = CONFIG
    H, W = shape
    n = len(years)
    raw_path = os.path.join(out, "epi_human_raw.npy")
    epi_path = os.path.join(out, "epi_stack.npy")

    # pass 1: raw human pressure per year (full grid so that distance decay crosses tile borders)
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype="float32", shape=(n, H, W))
    lo, hi = np.inf, -np.inf
    t0 = time.time()
    for t, yr in enumerate(years):
        code = get_code(t)
        hy = human_year(code, coef, c["PIXEL_KM"], inunit)
        raw[t] = hy
        if np.isfinite(hy).any():
            lo = min(lo, float(np.nanmin(hy))); hi = max(hi, float(np.nanmax(hy)))
        print(f"[EPI] {yr}: human pressure done  ({time.time() - t0:.0f}s)", flush=True)
    raw.flush(); del raw
    if not np.isfinite(lo):
        raise RuntimeError("[EPI] no assessed natural pixel found; check LANDUSE_DIR and UNITS_TIF")
    print(f"[EPI] A (joint min-max over {n} years): raw human pressure in [{lo:.4f}, {hi:.4f}]")

    # pass 2: normalise, blend drought, write the annual stack, time-mean and Theil-Sen trend (row blocks)
    es = np.lib.format.open_memmap(epi_path, mode="w+", dtype="float32", shape=(n, H, W)); del es
    epi_mean = np.full((H, W), np.nan, "float32")
    hum_mean = np.full((H, W), np.nan, "float32")
    trend = np.full((H, W), np.nan, "float32")
    blocks = [(r, min(r + BLOCK_ROWS, H)) for r in range(0, H, BLOCK_ROWS)]
    nj = c["N_JOBS"] if H * W > 2_000_000 else 1
    gen = Parallel(n_jobs=nj, return_as="generator_unordered")(
        delayed(epi_block)(r0, r1, raw_path, epi_path, drought[r0:r1], lo, hi, LAMBDA) for r0, r1 in blocks)
    for i, (r0, r1, em, hm, tr) in enumerate(gen, 1):
        epi_mean[r0:r1] = em; hum_mean[r0:r1] = hm; trend[r0:r1] = tr
        if i % 10 == 0 or i == len(blocks):
            print(f"[EPI] blocks {i}/{len(blocks)}  ({time.time() - t0:.0f}s)", flush=True)
    if not keep_raw:
        try:
            os.remove(raw_path)
        except OSError as e:
            print(f"[EPI] (could not remove {raw_path}: {e})")
    return epi_mean, hum_mean, trend, epi_path


def write_tables(out, coef):
    names = coef["names"]
    with open(os.path.join(out, "epi_sensitivity_mapping.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ESA-CCI code", "Landuse (Sheet2)", "assessed", *[f"S_{n}" for n in names]])
        for code in sorted(coef["landuse"]):
            w.writerow([code, coef["landuse"][code], int(coef["valid"][code]),
                        *[f"{v:.2f}" for v in coef["lut"][code]]])
    with open(os.path.join(out, "epi_sources.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["threat", "Sheet1 ID", "weight w_i", "d_max (km)", "decay", "ESA-CCI codes used as source"])
        for k, n in enumerate(names):
            w.writerow([n, coef["ids"][k], coef["w"][k], coef["dmax"][k],
                        "exponential" if coef["kind"][k] == "exp" else "linear",
                        ",".join(str(c) for c in sorted(ESACCI_SOURCE_CODES[n]))])
        w.writerow([])
        w.writerow(["never a source: ESA-CCI " + ",".join(str(c) for c in sorted(ESACCI_NON_SOURCE_CODES)) +
                    " (bare areas 200-202, lichens/mosses 140, sparse vegetation 150-153)"])


# ------------------------------------------------------------ main
def main():
    c = CONFIG
    out = gio.ensure_outdir()
    coef = load_coefficients()
    print("[EPI] coefficient workbook: " + os.path.basename(c["EPI_COEF_XLSX"]))
    for k, n in enumerate(coef["names"]):
        print(f"[EPI]   {n:<18s} ID {coef['ids'][k]:<4d} w={coef['w'][k]:<4g} d_max={coef['dmax'][k]:<4g} km  "
              f"{'exponential' if coef['kind'][k] == 'exp' else 'linear':<11s} codes {sorted(ESACCI_SOURCE_CODES[n])}")
    lu_dir = c.get("LANDUSE_DIR", "")
    real = bool(lu_dir) and gio.files_available(lu_dir) and gio.files_available(c["UNITS_TIF"])
    if real:
        years = list(range(c["LU_START_YEAR"], c["LU_END_YEAR"] + 1))
        grid = gio.analysis_grid()
        shape = (grid["height"], grid["width"])
        inunit = gio.read_units() > 0
        drought, has_drought = load_drought(out, shape)
        get_code = lambda t: gio.resample_to_ref(lu_path(years[t]), grid, categorical=True).astype("uint8")
        prof = grid
    elif c["ALLOW_DEMO"]:
        print("[EPI] LANDUSE_DIR / UNITS_TIF not found; running synthetic demo (edit paths in geo_io.py CONFIG)")
        codes, years, drought = generate_demo()
        shape = codes[0].shape
        inunit = np.ones(shape, bool)
        has_drought = True
        get_code = lambda t: codes[t]
        prof = None
    else:
        raise FileNotFoundError("[EPI] LANDUSE_DIR not found and ALLOW_DEMO=False")

    H, W = shape
    print(f"[EPI] grid {H}x{W}{' (ROI)' if c.get('ROI') else ''}, land-use years {years[0]}-{years[-1]} ({len(years)}), "
          f"lambda {LAMBDA}")
    write_tables(out, coef)

    epi_mean, hum_mean, trend, epi_path = run_pipeline(get_code, years, shape, inunit, drought, has_drought,
                                                       coef, out, prof)
    np.save(os.path.join(out, "epi_total.npy"), epi_mean)
    np.save(os.path.join(out, "epi_trend.npy"), trend)
    np.save(os.path.join(out, "epi_human.npy"), hum_mean)         # time-mean normalised human pressure (11_monte_carlo)
    if prof is not None:
        gio.write_tif(os.path.join(out, "epi_total.tif"), epi_mean, prof)
        gio.write_tif(os.path.join(out, "epi_human_mean.tif"), hum_mean, prof)
        gio.write_tif(os.path.join(out, "epi_trend.tif"), trend, prof)
        if WRITE_YEARLY_TIFS:
            es = np.load(epi_path, mmap_mode="r")
            for t, yr in enumerate(years):
                gio.write_tif(os.path.join(out, f"epi_{yr}.tif"), np.asarray(es[t]), prof)
            del es
        print(f"[EPI] wrote epi_total / epi_human_mean / epi_trend" +
              (f" + epi_{years[0]}..{years[-1]}.tif" if WRITE_YEARLY_TIFS else "") + f" to {out}")

    fin = np.isfinite(epi_mean)
    v = epi_mean[fin]
    if v.size:
        q = np.percentile(v[:: max(1, v.size // 2_000_000)], [5, 50, 95, 99])
        print(f"[EPI] EPI valid px {v.size:,}; mean {v.mean():.3f}; P5/P50/P95/P99 = {np.round(q, 3)}")
        hv = hum_mean[np.isfinite(hum_mean)]
        print(f"[EPI] human pressure mean {hv.mean():.3f}; drought term mean {float(drought[fin].mean()):.3f}"
              f"{'' if has_drought else ' (no drought file)'}")
        tv = trend[np.isfinite(trend)]
        if tv.size:
            print(f"[EPI] pressure RISING (dEPI>0) over {100 * np.mean(tv > 0):.1f}% of assessed pixels")
    if real:
        stable = gio.read_mask()
        print(f"[EPI] coverage of the stable natural mask: {100 * np.mean(fin[stable]):.1f}% of {int(stable.sum()):,} px")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sd = max(1, H // 900)
        es = np.load(epi_path, mmap_mode="r")
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
        im0 = ax[0].imshow(np.asarray(es[0, ::sd, ::sd]), cmap="OrRd", vmin=0, vmax=1)
        ax[0].set_title(f"EPI {years[0]}"); plt.colorbar(im0, ax=ax[0], shrink=.8)
        im1 = ax[1].imshow(np.asarray(es[-1, ::sd, ::sd]), cmap="OrRd", vmin=0, vmax=1)
        ax[1].set_title(f"EPI {years[-1]}"); plt.colorbar(im1, ax=ax[1], shrink=.8)
        im2 = ax[2].imshow(trend[::sd, ::sd], cmap="RdBu_r", vmin=-0.02, vmax=0.02)
        ax[2].set_title("dEPI (trend)"); plt.colorbar(im2, ax=ax[2], shrink=.8)
        for a in ax:
            a.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(out, "04_epi_preview.png"), dpi=120)
        print(f"[EPI] preview -> {out}/04_epi_preview.png")
    except Exception as e:
        print(f"[EPI] (preview skipped: {e})")
    print("[EPI] done.")


if __name__ == "__main__":
    main()
