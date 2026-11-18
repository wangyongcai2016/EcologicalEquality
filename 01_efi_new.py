#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 01 — 全类型生态系统功能指数 (Ecological Function Index, EFI)
================================================================================
计算公式:
    EFI_ij = 1/3 * min(FVC_ij / FVC_ref, 1) + 1/3 * min(LAI_ij / LAI_ref, 1)
           + 1/3 * min(GPP_ij / GPP_ref, 1)

基准参照值 (Reference Value):
    每个生态分区内部，在多期时序中保持稳定的自然生态系统（森林、灌木、草地、湿地等）
    像元多年均值的前 10% 分位数（Top-10th-percentile）均值，经 1000 次 Bootstrap 检验。
================================================================================
"""
from __future__ import annotations
import csv
import os
import warnings
import geo_io as gio
from geo_io import CONFIG, ESACCI_NATURAL_CODES
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---- 核心参数 ----
REF_PERCENTILE = 90.0
N_BOOTSTRAP    = 1000
MIN_REF_PIXELS = 30
RNG            = np.random.default_rng(42)


def unit_reference_bootstrap(param_mean, unit_labels, stable_mask):
    """按生态分区提取自然稳定像元的前 10% 参照基准值及 95% 置信区间"""
    ref, ref_lo, ref_hi, n_ref = {}, {}, {}, {}
    for u in np.unique(unit_labels):
        if u <= 0:
            continue
        # 筛选条件：当前分区 + 稳定自然生态像元 + 数值有效
        sel = (unit_labels == u) & stable_mask & np.isfinite(param_mean)
        vals = param_mean[sel]
        
        # 若稳定像元极少，保底使用该分区内所有有效像元
        if vals.size < MIN_REF_PIXELS:
            fallback_sel = (unit_labels == u) & np.isfinite(param_mean)
            vals = param_mean[fallback_sel]
            
        if vals.size == 0:
            continue
            
        thr = np.percentile(vals, REF_PERCENTILE)
        top = vals[vals >= thr]
        if top.size == 0:
            top = vals
            
        n_ref[int(u)] = int(top.size)
        ref[int(u)] = float(top.mean())
        
        boot = np.array([RNG.choice(top, size=top.size, replace=True).mean()
                         for _ in range(N_BOOTSTRAP)])
        ref_lo[int(u)] = float(np.percentile(boot, 2.5))
        ref_hi[int(u)] = float(np.percentile(boot, 97.5))
        
    return ref, ref_lo, ref_hi, n_ref


def _nanmean_time(stack):
    """沿时间轴的 nanmean, 逐年累加, 避免 np.nanmean 复制整栈 (全国网格 25 年约 8 GB)。全 NaN 像元返回 NaN。"""
    s = np.zeros(stack.shape[1:], "float64")
    k = np.zeros(stack.shape[1:], "int16")
    for t in range(stack.shape[0]):
        a = stack[t]
        ok = np.isfinite(a)
        s[ok] += a[ok]
        k += ok
    return np.where(k > 0, s / np.maximum(k, 1), np.nan).astype("float32")


def _ref_field(unit_labels, ref):
    out = np.full(unit_labels.shape, np.nan, "float32")
    for u, v in ref.items():
        out[unit_labels == u] = v
    return out


def capped_ratio(value, reference):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(value / reference, 0.0, 1.0)


def compute_efi(fvc, lai, gpp, unit_labels, refs):
    r1 = capped_ratio(fvc, _ref_field(unit_labels, refs["fvc"]))
    r2 = capped_ratio(lai, _ref_field(unit_labels, refs["lai"]))
    r3 = capped_ratio(gpp, _ref_field(unit_labels, refs["gpp"]))
    efi = (r1 + r2 + r3) / 3.0
    efi[unit_labels <= 0] = np.nan
    return efi


# ------------------------------------------------------------------ 稳定自然像元 (含 ESA-CCI 2015→2016 断点校正)
def _hard_conversion_codes():
    """耕地 / 灌溉耕地 / 城镇编码: 另一时段被其占据, 视为真实转换而非产品断点。"""
    from geo_io import ESACCI_SOURCE_CODES
    return sorted(ESACCI_SOURCE_CODES["Urbanland"] | ESACCI_SOURCE_CODES["Cropland"]
                  | ESACCI_SOURCE_CODES["IrrigatedCropland"])


def segment_years(c):
    """返回 (第一时段年份列表, 第二时段年份列表); 规则为 single 或断点年不在研究期内时第二时段为空。"""
    s, e = c["LU_START_YEAR"], c["LU_END_YEAR"]
    yrs = list(range(s, e + 1))
    b = c.get("LU_BREAK_YEAR")
    if c.get("STABLE_RULE", "breakpoint") != "breakpoint" or b is None or not (s < b <= e):
        return yrs, []
    return [y for y in yrs if y < b], [y for y in yrs if y >= b]


def classify_stable(nat1, nat2, hard1, hard2, len1, len2, min_share=0.80, mode="either", hard_share=0.80):
    """由各时段自然/人类利用年数计数得到稳定掩膜 (纯函数, 便于测试)。

    nat*/hard* : 各时段内自然 / 人类利用(耕地、灌溉耕地、城镇)的年数 (数组);  len* : 时段年数 (len2 == 0 -> 单时段)
    单时段    : 全期自然年占比 >= min_share
    either    : 任一时段自然年占比 >= min_share, 但剔除另一时段被人类利用占据 >= hard_share 的像元 (真实转换)
    both      : 每个时段自然年占比均 >= min_share
    """
    eps = 1e-9
    if len2 == 0:
        return (nat1 / float(len1)) >= min_share - eps
    ok1 = (nat1 / float(len1)) >= min_share - eps
    ok2 = (nat2 / float(len2)) >= min_share - eps
    if mode == "both":
        return ok1 & ok2
    st = ok1 | ok2
    if hard_share is not None:
        conv = (ok1 & ~ok2 & (hard2 / float(len2) >= hard_share - eps)) | \
               (ok2 & ~ok1 & (hard1 / float(len1) >= hard_share - eps))
        st &= ~conv
    return st


def _stable_signature(c, seg2):
    return {"stable_rule": c.get("STABLE_RULE", "breakpoint"),
            "break_year": c.get("LU_BREAK_YEAR") if seg2 else "none",
            "min_share": c.get("STABLE_MIN_SHARE", 0.80),
            "break_mode": c.get("STABLE_BREAK_MODE", "either"),
            "hard_share": c.get("STABLE_HARD_SHARE", 0.80),
            "years": f"{c['LU_START_YEAR']}-{c['LU_END_YEAR']}"}


def build_stable_masks(c, ref_prof):
    """逐年读取 ESA-CCI (流式, 内存只放计数器), 同时给出断点校正规则与原单时段规则的掩膜。"""
    seg1, seg2 = segment_years(c)
    first = set(seg1)
    nat_codes = list(ESACCI_NATURAL_CODES)
    hard_codes = _hard_conversion_codes()
    shp = (ref_prof["height"], ref_prof["width"])
    nat1 = np.zeros(shp, "uint8"); nat2 = np.zeros(shp, "uint8")
    hard1 = np.zeros(shp, "uint8"); hard2 = np.zeros(shp, "uint8")
    for yr, code in gio.iter_landuse_annual(c["LANDUSE_DIR"], c["LU_NAME"], c["LU_START_YEAR"],
                                            c["LU_END_YEAR"], c["LU_PATTERN"], ref_prof):
        n, h = np.isin(code, nat_codes), np.isin(code, hard_codes)
        if yr in first:
            nat1 += n; hard1 += h
        else:
            nat2 += n; hard2 += h
        print(f"[EFI]   ESA-CCI {yr} 已计数", flush=True)
    ms = c.get("STABLE_MIN_SHARE", 0.80)
    new = classify_stable(nat1, nat2, hard1, hard2, len(seg1), len(seg2), ms,
                          c.get("STABLE_BREAK_MODE", "either"), c.get("STABLE_HARD_SHARE", 0.80))
    old = classify_stable(nat1.astype("int16") + nat2, 0, 0, 0, len(seg1) + len(seg2), 0, ms)
    return new, old, (nat1, nat2, len(seg1), len(seg2))


def _write_stable_report(c, out_csv, new, old, parts, unit_labels):
    nat1, nat2, l1, l2 = parts
    ms = c.get("STABLE_MIN_SHARE", 0.80)
    inu = unit_labels > 0
    ok1 = nat1 / float(l1) >= ms - 1e-9
    ok2 = (nat2 / float(l2) >= ms - 1e-9) if l2 else np.zeros_like(ok1)
    rows = [("single_rule_all", old.sum()), ("breakpoint_rule_all", new.sum()),
            ("kept_from_single_rule", (new & old).sum()), ("added_by_breakpoint", (new & ~old).sum()),
            ("removed_vs_single_rule", (old & ~new).sum()),
            ("added_natural_seg1_only", (new & ~old & ok1 & ~ok2).sum()),
            ("added_natural_seg2_only", (new & ~old & ok2 & ~ok1).sum()),
            ("single_rule_in_units", (old & inu).sum()), ("breakpoint_rule_in_units", (new & inu).sum()),
            ("unit_pixels", inu.sum())]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item", "pixels"])
        for k, v in rows:
            w.writerow([k, int(v)])
    for k, v in rows:
        print(f"[EFI]   {k:<28s} {int(v):>12,d}")


def get_or_create_stable_mask(c, ref_prof, unit_labels):
    """获取或生成全类型自然生态系统的稳定像元掩膜 (ESA-CCI 2015→2016 断点校正)。

    规则 CONFIG["STABLE_RULE"]:
      "breakpoint" (默认): 2000..LU_BREAK_YEAR-1 与 LU_BREAK_YEAR..LU_END_YEAR 两个时段分别统计自然年占比,
                          任一时段 >= STABLE_MIN_SHARE 即稳定 (STABLE_BREAK_MODE="both" 则要求两段均满足);
                          "either" 下另一时段被耕地/城镇占据 >= STABLE_HARD_SHARE 的像元按真实转换剔除。
      "single": 原规则, 全期自然年占比 >= STABLE_MIN_SHARE。
    缓存的 STABLE_TIF 带规则标签; 标签与当前配置不一致 (或旧文件无标签) 时改名备份后重新生成, 不会静默沿用。
    """
    import time
    stable_tif_path = c.get("STABLE_TIF", "")
    seg1, seg2 = segment_years(c)
    sig = _stable_signature(c, seg2)
    desc = (f"{seg1[0]}-{seg1[-1]} + {seg2[0]}-{seg2[-1]} 分段" if seg2 else f"{seg1[0]}-{seg1[-1]} 单时段")

    # 1. 读取缓存 (规则标签一致才复用)
    if stable_tif_path and os.path.exists(stable_tif_path):
        tags = gio.read_tif_tags(stable_tif_path)
        if all(tags.get(k) == str(v) for k, v in sig.items()):
            print(f"[EFI] 读取已有自然生态稳定掩膜 ({desc}, 规则标签一致): {stable_tif_path}")
            s_arr, _ = gio.read_tif(stable_tif_path)
            return np.nan_to_num(s_arr, nan=0) > 0
        bak = stable_tif_path[:-4] + "_prev_" + time.strftime("%Y%m%d%H%M%S") + ".tif"
        os.replace(stable_tif_path, bak)
        print(f"[EFI] 已有掩膜的规则标签与当前配置不一致 (旧文件无标签视同不一致), 已备份为 {bak}, 将重新生成。")

    # 2. 从逐年土地利用生成
    lu_dir = c.get("LANDUSE_DIR")
    if gio.files_available(lu_dir):
        print(f"[EFI] 从 {lu_dir} 提取稳定自然像元: 规则={sig['stable_rule']} ({desc}), "
              f"阈值={sig['min_share']}, 模式={sig['break_mode']}")
        try:
            new, old, parts = build_stable_masks(c, ref_prof)
            print(f"[EFI] 稳定自然像元提取完成: 共计 {int(new.sum()):,} 个像元; 与原单时段规则对比:")
            if stable_tif_path:
                stem = stable_tif_path[:-4] if stable_tif_path.endswith(".tif") else stable_tif_path
                _write_stable_report(c, stem + "_comparison.csv", new, old, parts, unit_labels)
                try:
                    gio.write_tif(stable_tif_path, new.astype("uint8"), ref_prof, dtype="uint8", nodata=0, tags=sig)
                    gio.write_tif(stem + "_single.tif", old.astype("uint8"), ref_prof, dtype="uint8", nodata=0,
                                  tags={"stable_rule": "single", "years": sig["years"],
                                        "min_share": sig["min_share"]})
                    print(f"[EFI] 已保存断点校正掩膜 {stable_tif_path}\n"
                          f"[EFI]   及原单时段规则掩膜 {stem}_single.tif、对比表 {stem}_comparison.csv")
                except Exception as e:
                    print(f"[EFI] 缓存写入失败 (跳过保存): {e}")
            return new
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[EFI] 动态生成稳定掩膜失败: {e}")

    # 3. 降级保底
    print("[EFI] 警告: 未找到 STABLE_TIF 且无法动态生成，使用评估单元内全部有效像元。")
    return unit_labels > 0


# ------------------------------------------------------------------ 数据读取
def load_geotiffs():
    c = CONFIG
    print(f"[EFI] 读取全区多年植被参数序列 ({c['START_YEAR']}-{c['END_YEAR']})...")
    
    veg_dirs = c.get("VEG_DIRS") or c.get("VEG_DIR", ".")
    patterns = c.get("VEG_PATTERNS", c.get("VEG_PATTERN"))

    fvc_dir = c.get("FVC_DIR") or veg_dirs
    lai_dir = c.get("LAI_DIR") or veg_dirs
    gpp_dir = c.get("GPP_DIR") or veg_dirs

    fvc, prof, yrs = gio.stack_annual(fvc_dir, c["FVC_NAME"],
                                      c["START_YEAR"], c["END_YEAR"], patterns)
    lai, _, _ = gio.stack_annual(lai_dir, c["LAI_NAME"],
                                 c["START_YEAR"], c["END_YEAR"], patterns)
    gpp, _, _ = gio.stack_annual(gpp_dir, c["GPP_NAME"],
                                 c["START_YEAR"], c["END_YEAR"], patterns)
    
    if not (fvc.shape == lai.shape == gpp.shape):
        raise ValueError(f"[EFI] 空间维度不匹配: fvc{fvc.shape} lai{lai.shape} gpp{gpp.shape}")
    H, W = fvc.shape[1:]

    if c["SINGLE_UNIT"]:
        valid = np.isfinite(fvc).any(0) | np.isfinite(lai).any(0) | np.isfinite(gpp).any(0)
        unit_labels = valid.astype("int32")
        print("[EFI] SINGLE_UNIT=True: 全区作为单一单元计算")
    else:
        u_arr, _ = gio.read_tif(c["UNITS_TIF"])
        unit_labels = np.nan_to_num(u_arr, nan=0).astype("int32")
        if unit_labels.shape != (H, W):
            raise ValueError(f"[EFI] 评估单元尺寸 {unit_labels.shape} 与分析网格 {(H,W)} 不一致")

    stable = get_or_create_stable_mask(c, prof, unit_labels)
    return fvc, lai, gpp, unit_labels, stable, prof, yrs


def generate_demo():
    c = CONFIG
    H = W = 40
    yrs = list(range(c["START_YEAR"], c["END_YEAR"] + 1))
    n = len(yrs)
    yy, xx = np.mgrid[0:H, 0:W]
    unit_labels = (1 + (xx // 10) + 4 * (yy // 10)).astype("int32")
    natural = np.ones((H, W), bool)
    natural[(xx - W + 3) ** 2 + (yy - 2) ** 2 < 6] = False
    unit_labels[~natural] = 0
    stable = natural & (RNG.random((H, W)) > 0.2)
    base = 0.4 + 0.3 * RNG.random((H, W))
    fvc = np.empty((n, H, W), "float32")
    lai = np.empty_like(fvc)
    gpp = np.empty_like(fvc)
    for t in range(n):
        lvl = np.clip(base + 0.005 * t + 0.03 * RNG.standard_normal((H, W)), 0.05, 0.95)
        fvc[t] = lvl
        lai[t] = lvl * (2.5 + 0.5 * RNG.random((H, W)))
        gpp[t] = lvl * (900 + 200 * RNG.random((H, W)))
    fvc[:, ~natural] = np.nan
    lai[:, ~natural] = np.nan
    gpp[:, ~natural] = np.nan
    return fvc, lai, gpp, unit_labels, stable, None, yrs


# ------------------------------------------------------------------ 主流程
def main():
    c = CONFIG
    out = gio.ensure_outdir()
    
    veg_check = c.get("VEG_DIRS") or c.get("VEG_DIR")
    have = gio.files_available(veg_check) and (c["SINGLE_UNIT"] or gio.files_available(c.get("UNITS_TIF")))
    if have:
        fvc, lai, gpp, unit_labels, stable, prof, yrs = load_geotiffs()
    elif c["ALLOW_DEMO"]:
        print("[EFI] 未找到配置数据，正在运行合成 Demo...")
        fvc, lai, gpp, unit_labels, stable, prof, yrs = generate_demo()
    else:
        raise FileNotFoundError("[EFI] 未找到配置数据且 ALLOW_DEMO=False")

    n = fvc.shape[0]
    H, W = fvc.shape[1:]
    print(f"[EFI] 分析网格: {H}x{W}, 时段: {yrs[0]}-{yrs[-1]} ({n}年), "
          f"生态分区数: {len(np.unique(unit_labels))-1}, 稳定基准像元数: {int(stable.sum())}")

    # 分参数提取各生态分区基准值
    refs, refs_lo, refs_hi, n_ref = {}, {}, {}, {}
    for name, arr in [("fvc", fvc), ("lai", lai), ("gpp", gpp)]:
        pm = _nanmean_time(arr)
        refs[name], refs_lo[name], refs_hi[name], n_ref[name] = \
            unit_reference_bootstrap(pm, unit_labels, stable)

    # 逐年合成 EFI
    efi_stack = np.empty((n, H, W), "float32")
    for t in range(n):
        efi_stack[t] = compute_efi(fvc[t], lai[t], gpp[t], unit_labels, refs)
    efi_mean = _nanmean_time(efi_stack)
    # 置信区间: 逐年累加求均值 (与 np.nanmean(np.stack(...)) 等价, 但不再多占一整栈内存)
    def _mean_efi(rf):
        s = np.zeros(unit_labels.shape, "float64")
        k = np.zeros(unit_labels.shape, "int16")
        for t in range(n):
            e = compute_efi(fvc[t], lai[t], gpp[t], unit_labels, rf)
            ok = np.isfinite(e)
            s[ok] += e[ok]
            k += ok
        return np.where(k > 0, s / np.maximum(k, 1), np.nan).astype("float32")

    efi_lo = _mean_efi({"fvc": refs_hi["fvc"], "lai": refs_hi["lai"], "gpp": refs_hi["gpp"]})
    efi_hi = _mean_efi({"fvc": refs_lo["fvc"], "lai": refs_lo["lai"], "gpp": refs_lo["gpp"]})

    # 低置信度标记
    lowconf = np.zeros(unit_labels.shape, bool)
    for name in ("fvc", "lai", "gpp"):
        for u, cnt in n_ref[name].items():
            if cnt < MIN_REF_PIXELS:
                lowconf[unit_labels == u] = True

    # 保存 NumPy 数组
    np.save(os.path.join(out, "efi_stack.npy"), efi_stack.astype(np.float32))
    np.save(os.path.join(out, "efi_mean.npy"), efi_mean.astype(np.float32))
    np.save(os.path.join(out, "efi_ci_low.npy"), efi_lo.astype(np.float32))
    np.save(os.path.join(out, "efi_ci_high.npy"), efi_hi.astype(np.float32))
    np.save(os.path.join(out, "efi_lowconf_mask.npy"), lowconf)

    # 保存 GeoTIFF
    if prof is not None:
        gio.write_tif(os.path.join(out, "efi_mean.tif"), efi_mean, prof)
        gio.write_tif(os.path.join(out, "efi_ci_low.tif"), efi_lo, prof)
        gio.write_tif(os.path.join(out, "efi_ci_high.tif"), efi_hi, prof)
        gio.write_tif(os.path.join(out, "efi_lowconf.tif"),
                      lowconf.astype("uint8"), prof, "uint8", 255)
        for t, yr in enumerate(yrs):
            gio.write_tif(os.path.join(out, f"efi_{yr}.tif"), efi_stack[t], prof)
        print(f"[EFI] GeoTIFF 导出完成 (efi_mean.tif, efi_ci_*.tif, efi_{yrs[0]}..{yrs[-1]}.tif)")
    else:
        print("[EFI] Demo 模式: 仅输出 .npy 数据")

    # 输出基准值统计表
    with open(os.path.join(out, "efi_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "param", "ref", "ref_ci_low", "ref_ci_high", "n_ref"])
        for name in ("fvc", "lai", "gpp"):
            for u in sorted(refs[name]):
                w.writerow([u, name, f"{refs[name][u]:.4f}",
                            f"{refs_lo[name][u]:.4f}", f"{refs_hi[name][u]:.4f}", n_ref[name][u]])

    print(f"[EFI] 全区均值 EFI = {np.nanmean(efi_mean):.3f} "
          f"[{np.nanmean(efi_lo):.3f}, {np.nanmean(efi_hi):.3f}]")
    vn = int(np.isfinite(efi_mean).sum())
    print(f"[EFI] 低置信度像元数: {int(lowconf.sum())} "
          f"({100*lowconf.sum()/max(vn,1):.1f}% 的有效区域)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.imshow(efi_mean, cmap="YlGn", vmin=0, vmax=1)
        plt.colorbar(label=f"EFI ({yrs[0]}-{yrs[-1]} mean)")
        plt.title("Ecological Function Index (All Natural Ecosystems)")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "01_efi_preview.png"), dpi=120)
        print(f"[EFI] 空间分布预览图已生成 -> {out}/01_efi_preview.png")
    except Exception as e:
        print(f"[EFI] (预览图跳过: {e})")
    print("[EFI] 全类型生态质量评估运行完成。")


if __name__ == "__main__":
    main()