#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 geo_io.py — shared GeoTIFF I/O + central data-path configuration
================================================================================
ALL scripts (01-12) import this module. To run in VSCode WITHOUT command-line
arguments, edit the CONFIG block below ONCE with your data paths; every script
then reads from here. No --data / --out flags are needed.

Data organisation assumed:
  * Annual vegetation (FVC/LAI/GPP): separate folder & pattern per variable
        <FVC_DIR>/GLASS_FVC_Mean_500m_Albers_2000_clip.tif ...
        <LAI_DIR>/GLASS_LAI_Mean_500m_Albers_2000_clip.tif ...
        <GPP_DIR>/GLASS_GPP_Sum_500m_Albers_2000_clip.tif ...
  * Monthly NPP: one file per month
        <NPP_DIR>/npp_200001.tif ... npp_202412.tif      (yyyymm)
  * Monthly climate for ESI/SPEI: precip & PET, separate folders per variable
        <PRECIP_DIR>/pre_200001.tif ...  and  <PET_DIR>/pet_200001.tif ...
  * Land use: ANNUAL ESA-CCI GeoTIFFs, ONE FILE PER YEAR (300 m)
        <LANDUSE_DIR>/ESACCI-LC-2000.tif ... ESACCI-LC-2020.tif
  * Assessment units & stable mask: single GeoTIFFs
        <UNITS_TIF>, <STABLE_TIF>

Everything is assumed co-registered (same grid/CRS/extent). Helpers here verify
shapes and abort with a clear message otherwise.
================================================================================
"""
from __future__ import annotations
import glob
import os
import numpy as np

try:
    import rasterio
    _HAS_RIO = True
except Exception:
    _HAS_RIO = False


# ============================================================================
#  CONFIG — EDIT THESE PATHS, THEN RUN ANY SCRIPT DIRECTLY (F5 in VSCode)
# ============================================================================
CONFIG = {
    # --- root output directory (all scripts write here) ---
    "OUT_DIR": "./output",

    # --- study period ---
    "START_YEAR": 2000,
    "END_YEAR":   2024,

    # --- annual vegetation GeoTIFFs (01_efi) ---
    "FVC_DIR":    r"G:\FVC\FVC_Align",          # 存放 FVC 年均值影像文件夹
    "LAI_DIR":    r"G:\LAI\LAI_Align",          # 存放 LAI 年均值影像文件夹
    "GPP_DIR":    r"G:\GPP\GPP_Align",          # 存放 GPP 年总和影像文件夹

    # 植被变量目录映射（方便通用调用）
    "VEG_DIRS": {
        "fvc": r"G:\FVC\FVC_Align",
        "lai": r"G:\LAI\LAI_Align",
        "gpp": r"G:\GPP\GPP_Align",
    },
    
    "FVC_NAME":   "FVC",                     # 对应文件名占位符 {var}
    "LAI_NAME":   "LAI",
    "GPP_NAME":   "GPP",

    # 单独为各变量指定命名模板 (FVC/LAI 使用 Mean, GPP 使用 Sum)
    "VEG_PATTERNS": {
        "fvc": "GLASS_{var}_Mean_500m_Albers_{year}_clip.tif",
        "lai": "GLASS_{var}_Mean_500m_Albers_{year}_clip.tif",
        "gpp": "GLASS_{var}_Sum_500m_Albers_{year}_clip.tif",
    },
    "VEG_PATTERN": "GLASS_{var}_Mean_500m_Albers_{year}_clip.tif",  # 回退默认模板

    # --- monthly NPP GeoTIFFs (02_esi, 10_early_warning) ---
    "NPP_DIR":     r"G:\NPP\GLASS_NPP_500m_Month_clip",             # holds npp_YYYYMM.tif
    "NPP_NAME":    "NPP",
    "NPP_PATTERN": "GLASS_{var}_500m_{year}{month:02d}_clip.tif",   # e.g. npp_200001.tif

    # --- monthly climate GeoTIFFs (02_esi driver, 03_spei) ---
    "PRECIP_DIR":  r"E:\中国1km分辨率逐月气温和降水\中国1km分辨率逐月降水量数据集\Month_500m",     # 存放降水 pre 月度影像文件夹
    "PET_DIR":     r"E:\中国1km分辨率逐月气温和降水\中国1km逐月潜在蒸散发数据集\Month_500m",     # 存放潜在蒸散 pet 月度影像文件夹
    
    "CLIM_DIRS": {
        "pre": r"E:\中国1km分辨率逐月气温和降水\中国1km分辨率逐月降水量数据集\Month_500m",
        "pet": r"E:\中国1km分辨率逐月气温和降水\中国1km逐月潜在蒸散发数据集\Month_500m",
        "tmp": r"E:\中国1km分辨率逐月气温和降水\中国1km分辨率逐月平均气温数据集\Month_500m",   # used by 09a (AT_change)
    },
    
    "PRECIP_NAME": "pre",
    "PET_NAME":    "pet",
    "CLIM_PATTERN": "{var}_{year}_{month:02d}_500m.tif",

    # --- land use: ANNUAL ESA-CCI GeoTIFFs, ONE FILE PER YEAR (300 m) ---
    "LANDUSE_DIR":     r"H:\HabitatRisk\Landcover_Albers",     # holds ESACCI-LC-2000.tif ...
    "LU_NAME":         "LC",                                   # token in <token>_<year>.tif
    "LU_PATTERN":      "ESACCI-{var}-{year}.tif",
    "LU_START_YEAR":   2000,                                   # land-use record may be shorter than veg/npp
    "LU_END_YEAR":     2020,                                   # ESA-CCI here is 2000-2020
    # --- annual Human Footprint (global 1 km Mollweide, 0-50; Mu et al. 2022 for 2000-2018, operational update after)
    #     00_prepare_hfp.py: hfp<year>.zip -> aligned 500 m Albers GeoTIFFs on the UNITS_TIF grid, used by 09_attribution
    "HFP_ZIP_DIR": r"H:\数据\2024\人类足迹数据集",
    "HFP_DIR":     r"G:\HFP\HFP_Align",
    "HFP_PATTERN": "HFP_Mean_500m_Albers_{year}_clip.tif",
    "HFP_TMP":     r"G:\HFP\_tmp_extract",

    # raw global 300 m WGS84 ESA-CCI (one file per year, name contains "-<year>-"). Used by 08 to build the ecosystem-
    # type map when LANDUSE_DIR has no <year> file (H:\HabitatRisk\Landcover_Albers was removed in Oct 2026).
    "LANDUSE_RAW_DIR": r"E:\ESA_LandCover\ESA_LandCover_2000-2020",
    "LU_RESAMPLE_TO":  "veg",                                  # "veg" (use a veg GeoTIFF as 500 m ref) or a path
    "LANDUSE_TIF":     "",

    # --- stable natural pixel (01_efi_new.get_or_create_stable_mask) ---
    # ESA-CCI v2.0.7 (to 2015) and v2.1.1 (2016 on) are produced from different sensor sets, so natural/non-natural
    # labels jump between 2015 and 2016 (grassland <-> bare / mosaic mostly). "breakpoint" judges the two segments
    # 2000..LU_BREAK_YEAR-1 and LU_BREAK_YEAR..LU_END_YEAR separately; "single" is the old one-period rule.
    "STABLE_RULE":       "breakpoint",    # "breakpoint" | "single"
    "LU_BREAK_YEAR":     2016,            # first year of the second ESA-CCI segment
    "STABLE_MIN_SHARE":  0.80,            # natural-year share required (per segment, or over the whole period)
    "STABLE_BREAK_MODE": "either",        # "either": >= share in ANY segment; "both": in EVERY segment
    "STABLE_HARD_SHARE": 0.80,            # "either" only: drop pixels whose OTHER segment is >= this share cropland/
                                          # irrigated cropland/urban (a real conversion, not a product break); None = off

    # --- assessment units & stable mask (01_efi) ---
    "UNITS_TIF":  r"G:\全国生态质量评估\分区\生态分区_植被图_mask.tif",     # int class id 1..212; 0/NoData=masked
    "STABLE_TIF": r"./output/Lu_stable.tif",                            # >0 = land-cover-invariant natural pixel
    "SINGLE_UNIT": False,                                               # True -> ignore UNITS_TIF, treat all as 1 unit

    # --- grid ---
    "PIXEL_KM": 0.5,                                                    # pixel size in km (500 m -> 0.5) for distance-decay EPI

    # --- EPI coefficient workbook (04_epi): Sheet1 = pressure sources (ID, weight, d_max km, decay),
    #     Sheet2 = ecosystem sensitivity by ESA-CCI code. See the header of 04_epi.py.
    "EPI_COEF_XLSX": r"G:\全国生态质量评估\代码和文章初稿\input\系数表.xlsx",

    # --- ESI options ---
    "ESI_WINDOW": 5,                                                    # sliding-window length (years) for the ESI trend (dESI)
    "STL_PERIOD": 12,                                                   # monthly seasonality

    # --- aligned copies of the monthly rasters on the UNITS_TIF grid (written by 00_align_monthly.py;
    #     02_esi / 03_spei read these when present, else the raw dirs, which must already match the grid) ---
    "NPP_ALIGN_DIR":    r"G:\NPP\NPP_Align",
    "PRECIP_ALIGN_DIR": r"G:\PRE\PRE_Align",
    "PET_ALIGN_DIR":    r"G:\PET\PET_Align",

    # --- tiled processing (02_esi, 03_spei) ---
    "TILE":        256,       # tile edge in pixels; memory per worker ~ TILE^2 * months * 4 B * ~5
    "N_JOBS":      32,        # parallel tile workers
    "N_JOBS_ALIGN": 6,        # parallel files in 00_align_monthly
    "STL_ROBUST":  True,      # robust STL (≈6x slower; set False for a quick pass)
    "MAX_MISSING_MONTHS": 6,  # tolerated missing monthly files (filled with NaN, then interpolated by STL)
    "ROI": None,              # pilot run: (row0, row1, col0, col1) on the UNITS_TIF grid; None = whole grid

    # --- demo fallback ---
    "ALLOW_DEMO": True,

    # --- 06 trend / 07 breakpoint series: "core" = years in which ESI and EPI are observed (normally 2002-2020);
    #     "all" = every year of eqi_stack.npy (2000-2024, ESI/EPI extrapolated at the ends).
    "MGWR_PYTHON": r"G:\全国生态质量评估\代码和文章初稿\deepened_code\_venv_mgwr\Scripts\python.exe",   # numpy 2.2 venv: mgwr needs numba
    "TREND_YEARS": "core",
    "BREAK_PENALTY_C": 2.0,    # 07_bfast.py PELT penalty = c * sigma_d^2 * ln(n); sigma_d = noise sd of the pixel's
                               # differenced series (from its 2nd difference). Simulation, 19 yr, real-pixel noise:
                               #   c   false positive (linear, no break)   detected (V, slope flip +-0.02/yr)
                               #   1   10.5%                               88%
                               #   2    1.7%                               68%
                               #   3    0.2%                               53%
                               #   4    0.0%                               39%
                               # larger c = fewer, more reliable breaks; quote the value used in the paper.
}

# Relative paths are resolved against the project root (parent of scripts/), so results
# land in the same folder whatever the VSCode working directory is.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(p):
    return p if (not p or os.path.isabs(p)) else os.path.normpath(os.path.join(PROJECT_ROOT, p))


for _k in ("OUT_DIR", "STABLE_TIF"):
    CONFIG[_k] = _abs(CONFIG[_k])

# Environment overrides (parallel workers re-import this module, so a CONFIG edited at run time in the
# parent would be lost; environment variables are inherited). Handy for pilots without editing the file:
#   EQI_ROI="7050,7350,6300,6600"   EQI_OUT_DIR="D:/pilot_out"
if os.environ.get("EQI_ROI"):
    CONFIG["ROI"] = tuple(int(v) for v in os.environ["EQI_ROI"].split(","))
if os.environ.get("EQI_OUT_DIR"):
    CONFIG["OUT_DIR"] = _abs(os.environ["EQI_OUT_DIR"])
# ============================================================================


# ------------------------------------------------------------------ ESA-CCI
# 1. 人类活动压力源。压力源名称、权重、最大影响距离、衰减形式见系数表 Sheet1 (CONFIG["EPI_COEF_XLSX"]),
#    键名必须与 Sheet1 的 ThreadFactors 一致。Sheet1 只给每个压力源一个代表性 ID, 下面是各压力源实际包含的
#    ESA-CCI 编码 (作者设定, 请核对): ID 必须落在对应集合内, 各集合不得相交。
#    荒漠与裸地 (ESA-CCI 200-202 裸地, 以及 140、150-153 稀疏植被/地衣苔藓) 不得作为压力源。
ESACCI_SOURCE_CODES = {
    "Urbanland":         {190},          # 城镇 / 建设用地 (Sheet1 ID 190)
    "Cropland":          {10, 11, 12},   # 雨养耕地及其子类 (ID 10)
    "IrrigatedCropland": {20},           # 灌溉耕地 (ID 20)
    "Other":             {30, 40},       # 耕地与自然植被镶嵌 (ID 30; 40 为自然植被为主的镶嵌, 作者设定归入此类)
}
# 任何压力源编码集合不得与此集合相交 (source_masks_from_codes 会校验)
ESACCI_NON_SOURCE_CODES = {140, 150, 151, 152, 153, 200, 201, 202}

# 2. 全类型自然生态系统编码 (森林、灌丛、草地、苔原苔藓、稀疏植被、淹没水湿生态系统)
ESACCI_NATURAL_CODES = {
    # 森林林地 (Broadleaved / Needleleaved / Mixed Trees, 包含郁闭与疏林)
    50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100,
    # 灌丛与灌草混交 (Mosaic / Evergreen & Deciduous Shrubland)
    110, 120, 121, 122,
    # 天然草地 (Grassland)
    130,
    # 地衣与苔藓 (Lichens and mosses)
    140,
    # 稀疏植被 (Sparse vegetation: tree / shrub / herbaceous)
    150, 151, 152, 153,
    # 湿地 / 淹没森林与沼泽灌草 (Flooded tree / shrub / herbaceous cover)
    160, 170, 180,
}

# 3. 纯草地与灌草丛细分编码 (保留给草地子课题或敏感性专项分析)
ESACCI_GRASSLAND_CODES = {110, 120, 121, 122, 130, 140, 150, 152, 153}


# ------------------------------------------------------------------ IO
def _require_rio():
    if not _HAS_RIO:
        raise RuntimeError("rasterio is required to read/write GeoTIFFs. "
                           "Install with: pip install rasterio")


def _resolve_var_dir(data_dir, var: str) -> str:
    """Resolve directory path when data_dir can be a string, dict, or CONFIG key."""
    if isinstance(data_dir, dict):
        return data_dir.get(var.lower(), data_dir.get(var, "."))
    if isinstance(data_dir, str) and data_dir in CONFIG:
        val = CONFIG[data_dir]
        if isinstance(val, dict):
            return val.get(var.lower(), val.get(var, "."))
        return val
    return data_dir


def _resolve_pattern(pattern, var: str) -> str:
    """Resolve filename pattern when pattern can be a string, dict, or CONFIG key."""
    if isinstance(pattern, dict):
        return pattern.get(var.lower(), pattern.get(var, CONFIG.get("VEG_PATTERN", "{var}_{year}.tif")))
    if isinstance(pattern, str) and pattern in CONFIG:
        val = CONFIG[pattern]
        if isinstance(val, dict):
            return val.get(var.lower(), val.get(var, "{var}_{year}.tif"))
        return val
    return pattern


def read_tif(path: str):
    """Read single-band GeoTIFF -> (float32 array with NaN nodata, profile)."""
    _require_rio()
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nod = src.nodata
        prof = src.profile
    if nod is not None:
        arr[arr == nod] = np.nan
    return arr, prof


def write_tif(path: str, arr: np.ndarray, ref_profile, dtype="float32", nodata=np.nan, tags=None):
    """Write single-band GeoTIFF using a reference grid/CRS profile (optional metadata tags)."""
    _require_rio()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prof = ref_profile.copy()
    prof.update(count=1, dtype=dtype, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


def read_tif_tags(path: str) -> dict:
    """Metadata tags of a GeoTIFF (empty dict when there are none)."""
    _require_rio()
    with rasterio.open(path) as src:
        return dict(src.tags())


def stack_annual(data_dir, var: str, start: int, end: int, pattern=None):
    """Stack annual GeoTIFFs -> (n_years, H, W), profile, years."""
    _require_rio()
    target_dir = _resolve_var_dir(data_dir, var)
    pattern = pattern or CONFIG.get("VEG_PATTERNS", CONFIG.get("VEG_PATTERN", "{var}_{year}.tif"))
    actual_pattern = _resolve_pattern(pattern, var)
    
    arrs, years, prof = [], [], None
    for yr in range(start, end + 1):
        fn = actual_pattern.format(var=var, year=yr)
        path = os.path.join(target_dir, fn)
        if not os.path.exists(path):
            # 容错模糊匹配
            cand = glob.glob(os.path.join(target_dir, f"*{var}*{yr}*.tif"))
            if cand:
                path = sorted(cand)[0]
            else:
                raise FileNotFoundError(
                    f"missing {var} for {yr}: expected '{fn}' in {target_dir}")
        a, p = read_tif(path)
        prof = prof or p
        arrs.append(a)
        years.append(yr)
    return np.stack(arrs, 0), prof, years


def stack_monthly(data_dir, var: str, start: int, end: int, pattern=None):
    """Stack monthly GeoTIFFs -> (n_months, H, W), profile, ym."""
    _require_rio()
    target_dir = _resolve_var_dir(data_dir, var)
    pattern = pattern or CONFIG.get("CLIM_PATTERN", "{var}_{year}{month:02d}.tif")
    actual_pattern = _resolve_pattern(pattern, var)
    
    arrs, ym, prof = [], [], None
    for yr in range(start, end + 1):
        for mo in range(1, 13):
            fn = actual_pattern.format(var=var, year=yr, month=mo)
            path = os.path.join(target_dir, fn)
            if not os.path.exists(path):
                cand = glob.glob(os.path.join(target_dir, f"*{var}*{yr}{mo:02d}*.tif"))
                if cand:
                    path = sorted(cand)[0]
                else:
                    raise FileNotFoundError(
                        f"missing {var} for {yr}-{mo:02d}: expected '{fn}' in {target_dir}")
            a, p = read_tif(path)
            prof = prof or p
            arrs.append(a)
            ym.append(yr * 100 + mo)
    return np.stack(arrs, 0), prof, ym


def source_masks_from_codes(code, source_codes=None):
    """Build pressure-source masks + natural cover from an in-memory code array."""
    source_codes = source_codes or ESACCI_SOURCE_CODES
    bad = {n: sorted(set(c) & ESACCI_NON_SOURCE_CODES) for n, c in source_codes.items()
           if set(c) & ESACCI_NON_SOURCE_CODES}
    if bad:
        raise ValueError(f"desert / bare land / sparse vegetation cannot be pressure sources: {bad}")
    code = np.asarray(code)
    source_masks = {name: np.isin(code, list(codes))
                    for name, codes in source_codes.items()}
    natural = np.isin(code, list(ESACCI_NATURAL_CODES))
    cover_class = np.where(natural, 2, 0).astype(int)
    return source_masks, natural, cover_class


def landuse_source_masks(landuse_tif: str, source_codes=None):
    """Read one ESA-CCI GeoTIFF -> dict of boolean source masks + natural mask."""
    _require_rio()
    lc, prof = read_tif(landuse_tif)
    code = np.nan_to_num(lc, nan=0).astype(int)
    source_masks, natural, cover_class = source_masks_from_codes(code, source_codes)
    return source_masks, natural, cover_class, prof


def resample_to_ref(src_path: str, ref_profile, categorical=True):
    """Reproject/resample a raster onto a reference grid (e.g., 300 m to 500 m)."""
    _require_rio()
    from rasterio.warp import reproject
    from rasterio.enums import Resampling
    resampling = Resampling.nearest if categorical else Resampling.bilinear

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        dst = np.zeros((ref_profile["height"], ref_profile["width"]),
                       dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
            resampling=resampling,
            src_nodata=src_nodata, dst_nodata=0,
        )
    return np.nan_to_num(dst, nan=0).astype(int)


def iter_landuse_annual(landuse_dir: str, var: str, start: int, end: int, pattern: str,
                        ref_profile=None):
    """Yield (year, code array) one ANNUAL ESA-CCI land-cover GeoTIFF at a time (low memory)."""
    _require_rio()
    for yr in range(start, end + 1):
        fn = pattern.format(var=var, year=yr)
        path = os.path.join(landuse_dir, fn)
        if not os.path.exists(path):
            cand = glob.glob(os.path.join(landuse_dir, f"*{var}*{yr}*.tif"))
            if cand:
                path = sorted(cand)[0]
            else:
                raise FileNotFoundError(
                    f"missing land use for {yr}: expected '{fn}' in {landuse_dir}")
        if ref_profile is not None:
            code = resample_to_ref(path, ref_profile, categorical=True)
        else:
            lc, _ = read_tif(path)
            code = np.nan_to_num(lc, nan=0).astype(int)
        yield yr, code


def stack_landuse_annual(landuse_dir: str, var: str, start: int, end: int, pattern: str,
                         ref_profile=None):
    """Read ANNUAL ESA-CCI land-cover GeoTIFFs, optionally resampled to a grid. Stacks every year in memory:
    on the national 500 m grid use iter_landuse_annual instead."""
    _require_rio()
    arrs, years, src_prof = [], [], None
    for yr, code in iter_landuse_annual(landuse_dir, var, start, end, pattern, ref_profile):
        if src_prof is None:
            fn = pattern.format(var=var, year=yr)
            p = os.path.join(landuse_dir, fn)
            if not os.path.exists(p):
                p = sorted(glob.glob(os.path.join(landuse_dir, f"*{var}*{yr}*.tif")))[0]
            with rasterio.open(p) as s:
                src_prof = s.profile
        arrs.append(code)
        years.append(yr)
    return np.stack(arrs, 0), src_prof, years


def reference_profile(which="veg"):
    """Return a rasterio profile of the 500 m analysis grid to resample onto."""
    _require_rio()
    if which and which != "veg" and os.path.exists(which):
        with rasterio.open(which) as s:
            return s.profile
            
    # 优先从 FVC 目录或 VEG_DIRS['fvc'] 中获取 500 m 基准参考底图
    veg_dir = CONFIG.get("FVC_DIR") or CONFIG.get("VEG_DIRS", {}).get("fvc", CONFIG.get("VEG_DIR", "."))
    yr = CONFIG.get("LU_START_YEAR", CONFIG["START_YEAR"])
    fvc_pattern = _resolve_pattern(CONFIG.get("VEG_PATTERNS", CONFIG.get("VEG_PATTERN")), "fvc")
    fn = fvc_pattern.format(var=CONFIG.get("FVC_NAME", "FVC"), year=yr)
    path = os.path.join(veg_dir, fn)
    
    if not os.path.exists(path):
        cand = sorted(glob.glob(os.path.join(veg_dir, f"*{CONFIG.get('FVC_NAME', 'FVC')}*.tif")))
        if not cand:
            raise FileNotFoundError(
                f"reference_profile: no veg GeoTIFF found in {veg_dir} to define "
                f"the 500 m grid; set LU_RESAMPLE_TO to a 500 m reference file.")
        path = cand[0]
    with rasterio.open(path) as s:
        return s.profile


def ensure_outdir() -> str:
    out = CONFIG["OUT_DIR"]
    os.makedirs(out, exist_ok=True)
    return out


def eqi_annual_years(out=None, stack=None):
    """Calendar years of the annual EQI stack written by 05_eqi.py (eqi_stack.npy).

    05 writes 2000..END_YEAR when CONFIG["EQI_EXTEND"] is True, and only the common (core) years otherwise, so the
    first year cannot be assumed to be START_YEAR. The year list is taken from eqi_yearly_mean.csv (column 'year'),
    which 05 writes together with the stack; if the CSV is absent, END_YEAR - n + 1 .. END_YEAR is used.
    Returns (years ndarray, core_mask bool ndarray): core_mask is True for the years in which ESI and EPI are both
    observed (ESI_extrapolated == EPI_extrapolated == 0), i.e. the years a trend test should normally use."""
    import csv
    out = out or CONFIG["OUT_DIR"]
    n = stack.shape[0] if stack is not None else np.load(os.path.join(out, "eqi_stack.npy"), mmap_mode="r").shape[0]
    p = os.path.join(out, "eqi_yearly_mean.csv")
    if os.path.exists(p):
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if len(rows) == n:
            years = np.array([int(r["year"]) for r in rows])
            core = np.array([int(r.get("ESI_extrapolated", 0)) == 0 and int(r.get("EPI_extrapolated", 0)) == 0
                             for r in rows])
            return years, core
        print(f"[geo_io] eqi_yearly_mean.csv has {len(rows)} rows but eqi_stack.npy has {n} years; "
              f"falling back to END_YEAR-based labels")
    years = np.arange(CONFIG["END_YEAR"] - n + 1, CONFIG["END_YEAR"] + 1)
    return years, np.ones(n, bool)


def files_available(*paths) -> bool:
    """True if every path (file, dir, or dict of paths) exists."""
    for p in paths:
        if p is None:
            return False
        if isinstance(p, dict):
            if not all(files_available(v) for v in p.values()):
                return False
            continue
        if os.path.isdir(p):
            if not os.listdir(p):
                return False
        elif not os.path.exists(p):
            return False
    return True

# ------------------------------------------------------------------ master grid & tiled monthly access
# UNITS_TIF defines the analysis grid. Monthly NPP / precipitation / PET come on other grids
# (different origin, extent and Albers latitude of origin), so 00_align_monthly.py resamples
# them onto this grid once; 02_esi.py then reads the aligned copies tile by tile.
MONTHLY_SPECS = {
    # key: (raw dir key, aligned dir key, name key, pattern key)
    "npp": ("NPP_DIR",    "NPP_ALIGN_DIR",    "NPP_NAME",    "NPP_PATTERN"),
    "pre": ("PRECIP_DIR", "PRECIP_ALIGN_DIR", "PRECIP_NAME", "CLIM_PATTERN"),
    "pet": ("PET_DIR",    "PET_ALIGN_DIR",    "PET_NAME",    "CLIM_PATTERN"),
}


def align_dir(key: str) -> str:
    """Aligned-copy directory. Always full-grid: a pilot ROI reads a window of these same files."""
    return CONFIG[key]


def monthly_path(key: str, year: int, month: int, aligned: bool = False) -> str:
    raw_k, al_k, name_k, pat_k = MONTHLY_SPECS[key]
    d = align_dir(al_k) if aligned else CONFIG[raw_k]
    return os.path.join(d, CONFIG[pat_k].format(var=CONFIG[name_k], year=year, month=month))


def analysis_grid(full: bool = False):
    """Profile (size / transform / CRS) of the analysis grid = UNITS_TIF, cropped to CONFIG['ROI'] if set
    (full=True ignores the ROI)."""
    _require_rio()
    from rasterio.windows import Window, transform as wtransform
    with rasterio.open(CONFIG["UNITS_TIF"]) as s:
        prof = s.profile.copy()
        tr = s.transform
    roi = None if full else CONFIG.get("ROI")
    if roi:
        r0, r1, c0, c1 = roi
        r1 = min(r1, prof["height"]); c1 = min(c1, prof["width"])
        prof.update(height=r1 - r0, width=c1 - c0,
                    transform=wtransform(Window(c0, r0, c1 - c0, r1 - r0), tr))
    return prof


def _master_window(win=None):
    """Window given relative to the analysis grid -> window on the full UNITS_TIF grid."""
    from rasterio.windows import Window
    roi = CONFIG.get("ROI")
    r0, c0 = (roi[0], roi[2]) if roi else (0, 0)
    if win is None:
        g = analysis_grid()
        return Window(c0, r0, g["width"], g["height"])
    return Window(win.col_off + c0, win.row_off + r0, win.width, win.height)


def read_units(win=None, full: bool = False):
    """Assessment-unit ids (int32) on the analysis grid (full=True: whole UNITS_TIF, ignoring ROI);
    nodata and negatives -> 0."""
    _require_rio()
    from rasterio.windows import Window
    with rasterio.open(CONFIG["UNITS_TIF"]) as s:
        w = Window(0, 0, s.width, s.height) if full else _master_window(win)
        a = s.read(1, window=w).astype("int32")
        nod = s.nodata
    if nod is not None and np.isfinite(nod):
        a[a == int(nod)] = 0
    a[a < 0] = 0
    return a


def read_mask(win=None):
    """Pixels analysed by 02_esi: inside an assessment unit and, when STABLE_TIF exists (written by
    01_efi_new.py), land-cover-stable natural vegetation. Without STABLE_TIF every unit pixel is used."""
    m = read_units(win) > 0
    st = CONFIG.get("STABLE_TIF")
    if st and os.path.exists(st):
        with rasterio.open(st) as s:
            with rasterio.open(CONFIG["UNITS_TIF"]) as u:
                if (s.height, s.width) != (u.height, u.width):
                    raise ValueError(f"STABLE_TIF {s.height}x{s.width} differs from UNITS_TIF "
                                     f"{u.height}x{u.width}; regenerate it with 01_efi_new.py")
            b = s.read(1, window=_master_window(win))
        m &= np.nan_to_num(b.astype("float32")) > 0
    return m


def _fill_missing_months(arr, missing):
    """Linear interpolation (copy at the ends) across whole months whose file is absent."""
    miss = set(missing)
    have = [k for k in range(arr.shape[0]) if k not in miss]
    for k in missing:
        before = [h for h in have if h < k]
        after = [h for h in have if h > k]
        if before and after:
            b, a = before[-1], after[0]
            f = (k - b) / (a - b)
            arr[k] = (1 - f) * arr[b] + f * arr[a]
        elif after:      # leading gap: same calendar month of the next year if available (keeps seasonality)
            arr[k] = arr[k + 12] if (k + 12) in have else arr[after[0]]
        else:            # trailing gap
            arr[k] = arr[k - 12] if (k - 12) in have else arr[before[-1]]


def read_monthly_window(key: str, start: int, end: int, win=None, fill: bool = True):
    """Aligned monthly stack for one tile -> ((n_months, h, w) float32, missing month indices).
    Months whose file does not exist are NaN, or interpolated in time when fill=True."""
    _require_rio()
    from rasterio.windows import Window
    g = analysis_grid()
    if win is None:
        win = Window(0, 0, g["width"], g["height"])
    h, w = int(win.height), int(win.width)
    mw = _master_window(win)          # aligned files are full-grid; shift by the ROI offset
    T = (end - start + 1) * 12
    out = np.full((T, h, w), np.nan, "float32")
    missing, k = [], 0
    for yr in range(start, end + 1):
        for mo in range(1, 13):
            p = monthly_path(key, yr, mo, aligned=True)
            if os.path.exists(p):
                with rasterio.open(p) as s:
                    out[k] = s.read(1, window=mw)
            else:
                missing.append(k)
            k += 1
    if len(missing) == T:
        raise FileNotFoundError(f"no aligned '{key}' files in {align_dir(MONTHLY_SPECS[key][1])}; "
                                f"run 00_align_monthly.py first")
    if fill and missing:
        _fill_missing_months(out, missing)
    return out, missing

if os.environ.get("EQI_TILE"):
    CONFIG["TILE"] = int(os.environ["EQI_TILE"])
if os.environ.get("EQI_N_JOBS"):                      # parallel tile workers without editing CONFIG
    CONFIG["N_JOBS"] = int(os.environ["EQI_N_JOBS"])
