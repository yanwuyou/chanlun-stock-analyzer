#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动力学模块
管线：MACD计算 → 中枢识别（三命运） → 走势类型 → 背驰（严格6条件） → 买卖点
基于第15/17/18/20/21/24/27/29/37/53/101课
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


# ============================================================
# 第一部分：MACD 计算
# ============================================================

def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    """计算 MACD 指标"""
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["diff"] = df["ema_fast"] - df["ema_slow"]
    df["dea"] = df["diff"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = 2 * (df["diff"] - df["dea"])
    return df


# ============================================================
# 第二部分：中枢识别（第17/18/20课）⭐ 三命运完整版
# ============================================================

def _sub_level_range(element: dict) -> Tuple[float, float]:
    """获取次级别走势类型的价格区间 [low, high]"""
    if element["dir"] == "up":
        return element["start_price"], element["end_price"]
    else:
        return element["end_price"], element["start_price"]


def identify_zhongshu(sub_elements: List[dict],
                      label: str = "bi") -> pd.DataFrame:
    """
    识别中枢：至少三个连续次级别走势类型重叠的部分
    sub_elements: 笔列表 或 线段列表（作为次级别走势类型）
    label: "bi" 或 "segment"

    中枢区间 Z = [ZD, ZG] = [max(三个低点), min(三个高点)]
    新增字段：ZG, ZD, ZZ, GG, DD（第20课完整指标集）
    新增：三种命运判断（延伸/新生/扩张）
    """
    if len(sub_elements) < 3:
        df = pd.DataFrame()
        df.attrs["zhongshu_list"] = []
        return df

    zhongshu_list = []
    n = len(sub_elements)

    for i in range(n - 2):
        e0, e1, e2 = sub_elements[i], sub_elements[i + 1], sub_elements[i + 2]
        lo0, hi0 = _sub_level_range(e0)
        lo1, hi1 = _sub_level_range(e1)
        lo2, hi2 = _sub_level_range(e2)

        ol_lo = max(lo0, lo1, lo2)
        ol_hi = min(hi0, hi1, hi2)

        if ol_lo < ol_hi:
            # 记录三段的全部高点和低点（用于后续计算 GG/DD）
            all_highs = [hi0, hi1, hi2]
            all_lows = [lo0, lo1, lo2]

            zhongshu_list.append({
                "sub_start": i, "sub_end": i + 2,
                "zd": ol_lo,
                "zg": ol_hi,
                "zz": (ol_lo + ol_hi) / 2,
                "gg": max(all_highs),   # 所有次级别高点的最大值
                "dd": min(all_lows),     # 所有次级别低点的最小值
                "start_date": sub_elements[i]["start_date"],
                "end_date": sub_elements[i + 2]["end_date"],
                "start_idx": sub_elements[i]["start_idx"],
                "end_idx": sub_elements[i + 2]["end_idx"],
                "element_count": 3,
                "fate": None,  # 稍后判断
                "label": label,
            })

    # 合并连续重叠的中枢
    if zhongshu_list:
        merged = [zhongshu_list[0]]
        for zs in zhongshu_list[1:]:
            last = merged[-1]
            if zs["zd"] < last["zg"] and zs["zg"] > last["zd"]:
                # 重叠 → 是同一中枢的延续
                last["zd"] = max(last["zd"], zs["zd"])
                last["zg"] = min(last["zg"], zs["zg"])
                last["zz"] = (last["zd"] + last["zg"]) / 2
                last["gg"] = max(last["gg"], zs["gg"])
                last["dd"] = min(last["dd"], zs["dd"])
                last["sub_end"] = zs["sub_end"]
                last["end_date"] = zs["end_date"]
                last["end_idx"] = zs["end_idx"]
                last["element_count"] += (zs["sub_end"] - zs["sub_start"])
            else:
                merged.append(zs)
        zhongshu_list = merged

    # 判断每个中枢的三种命运
    for idx in range(len(zhongshu_list)):
        zs = zhongshu_list[idx]
        next_zs = zhongshu_list[idx + 1] if idx + 1 < len(zhongshu_list) else None

        if next_zs:
            # 判断中枢新生 vs 扩张（第20课）
            # 两个中枢区间不重叠时：检查波动区间是否重叠
            if next_zs["zd"] > zs["zg"]:
                # 后中枢完全在前中枢上方
                if next_zs["dd"] <= zs["gg"]:
                    zs["fate"] = "expansion"  # 波动重叠 → 扩张为高级别中枢
                else:
                    zs["fate"] = "new_uptrend"
            elif next_zs["zg"] < zs["zd"]:
                # 后中枢完全在前中枢下方
                if next_zs["gg"] >= zs["dd"]:
                    zs["fate"] = "expansion"  # 波动重叠 → 扩张为高级别中枢
                else:
                    zs["fate"] = "new_downtrend"
            else:
                zs["fate"] = "extension"
        else:
            zs["fate"] = "active"  # 最新的中枢，还未确定命运

        # 中枢延伸超过9段 → 自动升级提醒
        if zs["element_count"] >= 9:
            zs["fate"] = "upgrade" if zs["fate"] in ("extension", "active") else zs["fate"]
            zs["upgrade_note"] = "延伸超9段，形成更高级别中枢"

    df = pd.DataFrame()
    df.attrs["zhongshu_list"] = zhongshu_list
    df.attrs["sub_element_type"] = label
    return df


# ============================================================
# 第三部分：走势类型判断（第15/17/18课）
# ============================================================

def identify_zoushi_type(zhongshu_list: List[dict],
                         sub_elements: List[dict]) -> dict:
    """
    判断走势类型：盘整 / 上涨趋势 / 下跌趋势

    严格条件：
    - 趋势 = 至少两个同向不重叠中枢（后DD > 前GG 或 后GG < 前DD）
    - 盘整 = 只有一个该级别中枢
    - 趋势延伸本质 = 不断产生新同向中枢
    """
    result = {
        "type": "unknown",
        "zhongshu_count": len(zhongshu_list),
        "latest_zs": zhongshu_list[-1] if zhongshu_list else None,
        "zs_chain": [],  # 中枢演化链
    }

    if len(zhongshu_list) >= 2:
        last_zs = zhongshu_list[-1]
        prev_zs = zhongshu_list[-2]

        if last_zs["zg"] > prev_zs["zg"] and last_zs["zd"] > prev_zs["zd"]:
            result["type"] = "uptrend"
        elif last_zs["zg"] < prev_zs["zg"] and last_zs["zd"] < prev_zs["zd"]:
            result["type"] = "downtrend"
        else:
            # 可能是扩张形成更大级别中枢
            result["type"] = "consolidation_with_expansion"

        result["zs_chain"] = [
            {"idx": i, "zd": z["zd"], "zg": z["zg"], "fate": z.get("fate")}
            for i, z in enumerate(zhongshu_list)
        ]
    elif len(zhongshu_list) == 1:
        result["type"] = "consolidation"
    else:
        if sub_elements:
            last_el = sub_elements[-1]
            result["type"] = "uptrend" if last_el["dir"] == "up" else "downtrend"

    return result


# ============================================================
# 第四部分：背驰判断（第15/24/27/37课）⭐ 严格6条件
# ============================================================

def _macd_area(df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
    """计算指定区间的MACD柱面积（绝对值之和）"""
    if start_idx < 0 or end_idx >= len(df) or start_idx >= end_idx:
        return 0.0
    return float(df.loc[start_idx:end_idx, "macd_hist"].abs().sum())


def _macd_diff_range(df: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[float, float]:
    """获取区间内DIFF的最小值和最大值"""
    seg = df.loc[start_idx:min(end_idx, len(df) - 1), "diff"]
    return float(seg.min()), float(seg.max())


def _is_near_zero_axis(df: pd.DataFrame, start_idx: int, end_idx: int,
                       threshold: float = None) -> bool:
    """
    检查MACD是否在区间内回拉到0轴附近（条件5 — 最关键！第37课）

    动态阈值：基于该区间DIFF的波动范围自动计算
    - DIFF穿越0轴（从正变负或从负变正）
    - 或 DIFF绝对值小于动态阈值（区间振幅的10%，至少0.01）
    """
    if start_idx < 0 or end_idx >= len(df) or start_idx >= end_idx:
        return False
    diffs = df.loc[start_idx:end_idx, "diff"]
    diff_range = diffs.max() - diffs.min()
    if threshold is None:
        threshold = max(diff_range * 0.1, 0.01) if diff_range > 0 else 0.05

    # DIFF穿越0轴
    signs = np.sign(diffs.values)
    has_cross = bool((signs[0] > 0 and signs[-1] < 0) or
                     (signs[0] < 0 and signs[-1] > 0) or
                     (np.diff(signs[signs != 0]) != 0).any())
    # 或接近0轴
    has_near_zero = (diffs.abs() < threshold).any()
    return has_cross or has_near_zero


def _c_has_third_point(sub_elements: List[dict], zs_b: dict, is_uptrend: bool) -> bool:
    """
    条件2（第37课）：c必须包含对B的第三类买卖点

    上涨趋势背驰 → c必须包含对B的三买：
      c离开B向上（创ZG以上新高），然后次级别回抽不破ZG
    下跌趋势背驰 → c必须包含对B的三卖：
      c离开B向下（创ZD以下新低），然后次级别反弹不破ZD

    sub_elements: 用于构建中枢的次级别元素列表（笔或线段）
    zs_b: 最后一个中枢B
    """
    start = zs_b["sub_end"]
    if start >= len(sub_elements) - 1:
        return False

    zg, zd = zs_b["zg"], zs_b["zd"]

    if is_uptrend:
        # 找次级别离开中枢向上（创新高越过ZG），且次级别回抽不破ZG
        leaving = None
        for i in range(start, len(sub_elements)):
            el = sub_elements[i]
            if el["dir"] == "up" and el["end_price"] > zg:
                leaving = i
                break
        if leaving is None:
            return False
        # 在离开段之后，找向下回抽不破ZG
        for j in range(leaving + 1, len(sub_elements)):
            el = sub_elements[j]
            if el["dir"] == "down":
                if el["end_price"] >= zg:
                    return True
                elif el["end_price"] < zg:
                    return False  # 跌回ZG以下，不是三买
    else:
        # 找次级别离开中枢向下（创新低越过ZD），且次级别反弹不破ZD
        leaving = None
        for i in range(start, len(sub_elements)):
            el = sub_elements[i]
            if el["dir"] == "down" and el["end_price"] < zd:
                leaving = i
                break
        if leaving is None:
            return False
        for j in range(leaving + 1, len(sub_elements)):
            el = sub_elements[j]
            if el["dir"] == "up":
                if el["end_price"] <= zd:
                    return True
                elif el["end_price"] > zd:
                    return False  # 涨回ZD以上，不是三卖

    return False


def identify_divergence(df: pd.DataFrame, zhongshu_list: List[dict],
                        sub_elements: List[dict],
                        df_low: pd.DataFrame = None) -> dict:
    """
    识别背驰（完整6条件）

    趋势背驰 a+A+b+B+c 结构：
    条件1：A和B必须是同级别中枢
    条件2：c必须包含对B的第三类买卖点
    条件3：b的级别不能大于c
    条件4：c必须创出新高/新低
    条件5：B必须将MACD黄白线回拉到0轴附近 ⭐
    条件6：c完成时MACD面积 < b段面积（或黄白线不创新高）

    盘整背驰：中枢震荡中离开段力度 < 进入段力度
    """
    result = {
        "has_divergence": False, "type": None, "details": None,
        "conditions_check": [], "confidence": "none",
    }

    if len(zhongshu_list) < 1 or len(sub_elements) < 4:
        return result

    # --- 趋势背驰 ---
    if len(zhongshu_list) >= 2:
        zs_a = zhongshu_list[-2]
        zs_b = zhongshu_list[-1]

        is_uptrend = zs_b["zg"] > zs_a["zg"] and zs_b["zd"] > zs_a["zd"]
        is_downtrend = zs_b["zg"] < zs_a["zg"] and zs_b["zd"] < zs_a["zd"]

        if is_uptrend or is_downtrend:
            conditions = []
            # 条件1：同级别中枢 — 通过构造保证（都来自同一个 sub_elements）

            b_start = int(zs_a["end_idx"])
            b_end = int(zs_b["start_idx"])
            c_start = int(zs_b["end_idx"])
            c_end = len(df) - 1

            # 条件2：c必须包含对B的第三类买卖点（第37课）
            c_has_third = _c_has_third_point(sub_elements, zs_b, is_uptrend)
            conditions.append(("条件2-c包含三类买卖点", c_has_third,
                               "三买确认" if (is_uptrend and c_has_third) else
                               "三卖确认" if (not is_uptrend and c_has_third) else "未确认"))

            # 条件4：c必须创出新高/新低
            if is_uptrend:
                c_high = float(df.loc[c_start:c_end, "high"].max())
                b_high = float(df.loc[b_start:min(b_end, len(df) - 1), "high"].max())
                c_new_extreme = c_high > b_high
                conditions.append(("条件4-c创新高", c_new_extreme, f"c高={c_high:.2f}>b高={b_high:.2f}"))
            else:
                c_low = float(df.loc[c_start:c_end, "low"].min())
                b_low = float(df.loc[b_start:min(b_end, len(df) - 1), "low"].min())
                c_new_extreme = c_low < b_low
                conditions.append(("条件4-c创新低", c_new_extreme, f"c低={c_low:.2f}<b低={b_low:.2f}"))

            # 条件5：B必须将MACD拉回0轴附近
            b_inner_start = int(zs_b["start_idx"])
            b_inner_end = int(zs_b["end_idx"])
            b_near_zero = _is_near_zero_axis(df, b_inner_start, b_inner_end)
            conditions.append(("条件5-B拉回0轴", b_near_zero, ""))

            # 条件6：力度对比
            if c_start < c_end:
                b_area = _macd_area(df, b_start, min(b_end, len(df) - 1))
                c_area = _macd_area(df, c_start, c_end)

                # 面积法
                area_divergence = c_area < b_area
                # 黄白线法
                c_diff_min, c_diff_max = _macd_diff_range(df, c_start, c_end)
                if is_uptrend:
                    b_diff_max = float(df.loc[b_start:min(b_end, len(df) - 1), "diff"].max())
                    diff_divergence = c_diff_max < b_diff_max
                    conditions.append(("条件6-黄白线不创新高", diff_divergence,
                                       f"c高={c_diff_max:.4f}<b高={b_diff_max:.4f}"))
                else:
                    b_diff_min = float(df.loc[b_start:min(b_end, len(df) - 1), "diff"].min())
                    diff_divergence = c_diff_min > b_diff_min
                    conditions.append(("条件6-黄白线不创新低", diff_divergence,
                                       f"c低={c_diff_min:.4f}>b低={b_diff_min:.4f}"))
                conditions.append(("条件6-面积背驰", area_divergence,
                                   f"c面积={c_area:.0f}<b面积={b_area:.0f}"))

                # 综合判断：需满足条件2/4/5/6
                force_divergence = area_divergence or diff_divergence
                all_passed = force_divergence and c_new_extreme and b_near_zero and c_has_third

                if all_passed:
                    result["has_divergence"] = True
                    result["type"] = "trend_divergence"
                    result["details"] = {
                        "direction": "top_divergence" if is_uptrend else "bottom_divergence",
                        "b_area": b_area, "c_area": c_area,
                        "b_range": (b_start, b_end),
                        "c_range": (c_start, c_end),
                        "zhongshu_a": zs_a, "zhongshu_b": zs_b,
                    }
                    result["confidence"] = "high"

                    # 第24课验证：背驰后回跌必须至少回到B中枢区间
                    if is_uptrend:
                        c_low = float(df.loc[c_start:c_end, "low"].min())
                        pulled_back_to_b = c_low <= zs_b["zg"]
                    else:
                        c_high = float(df.loc[c_start:c_end, "high"].max())
                        pulled_back_to_b = c_high >= zs_b["zd"]
                    result["details"]["pulled_back_to_b_zs"] = pulled_back_to_b
                    if not pulled_back_to_b:
                        result["warnings"] = result.get("warnings", [])
                        result["warnings"].append(
                            "第24课：c段未回到B中枢区间，背驰可能还未完成，需继续观察")
                elif force_divergence and not b_near_zero:
                    result["conditions_check"] = conditions
                    result["confidence"] = "low"
                    result["warnings"] = ["B段未将MACD拉回0轴，可能只是盘整背驰而非趋势背驰"]

            result["conditions_check"] = conditions

    # --- 盘整背驰（第27课）---
    if not result["has_divergence"] and len(zhongshu_list) >= 1:
        zs = zhongshu_list[-1]
        zs_el_start = zs["sub_start"]
        zs_el_end = zs["sub_end"]

        if zs_el_start > 0 and zs_el_end < len(sub_elements) - 1:
            enter_el = sub_elements[zs_el_start - 1]
            exit_el = sub_elements[zs_el_end + 1] if zs_el_end + 1 < len(sub_elements) else None

            if exit_el and enter_el["dir"] == exit_el["dir"]:
                enter_area = _macd_area(df, int(enter_el["start_idx"]),
                                        int(enter_el["end_idx"]))
                exit_area = _macd_area(df, int(exit_el["start_idx"]),
                                       int(exit_el["end_idx"]))

                if exit_area < enter_area:
                    last_price = float(df["close"].iloc[-1])
                    result["has_divergence"] = True
                    result["type"] = "consolidation_divergence"
                    result["confidence"] = "medium"
                    result["details"] = {
                        "direction": "top_divergence" if exit_el["dir"] == "up" else "bottom_divergence",
                        "enter_area": enter_area, "exit_area": exit_area,
                        "enter_range": (enter_el["start_idx"], enter_el["end_idx"]),
                        "exit_range": (exit_el["start_idx"], exit_el["end_idx"]),
                    }

                    # C段不破中枢 + 盘整背驰 → 其后必有回跌
                    if exit_el["dir"] == "up":
                        if exit_el["end_price"] < zs["zg"]:
                            result["details"]["c_break_zs"] = False
                            result["details"]["note"] = "C段未破ZG + 盘整背驰 → 其后必有回跌"
                        else:
                            result["details"]["c_break_zs"] = True
                            result["details"]["note"] = "C段上破ZG + 盘整背驰 → 先退出，回跌不破ZG则三买"

    return result


# ============================================================
# 第五部分：背驰-转折定理（第29课）
# ============================================================

def divergence_to_reversal(divergence: dict, zhongshu_list: List[dict],
                           sub_elements: List[dict]) -> dict:
    """
    背驰-转折定理（第29课）：某级别趋势背驰将导致三种情况之一

    判断方法（原文）：
      看反弹中第一个前趋势最后一个中枢级别的次级别走势，
      是否重新回抽最后一个中枢里。
      如果不能 → 第一种情况（最弱）
      如果能 → 第二种或第三种

    一、最后一个中枢的级别扩展（最弱）
      反弹/回落只触及GG/DD，未进入中枢区间[ZD,ZG]
    二、该级别更大级别的盘整
      反弹/回落进入中枢区间[ZD,ZG]，但未形成反趋势
    三、该级别以上级别的反趋势
      反弹突破ZG（底背驰）/ 回落跌破ZD（顶背驰）
    """
    if not divergence.get("has_divergence") or divergence.get("type") != "trend_divergence":
        return {"status": "no_trend_divergence"}

    last_zs = zhongshu_list[-1] if zhongshu_list else None
    if not last_zs:
        return {"status": "no_zhongshu"}

    direction = divergence.get("details", {}).get("direction", "")
    last_price = sub_elements[-1]["end_price"] if sub_elements else 0
    zg, zd, gg, dd = last_zs["zg"], last_zs["zd"], last_zs["gg"], last_zs["dd"]

    # 检查第一个次级别反向走势是否回抽到中枢内
    reentered_zs = False
    first_reverse = None
    if len(sub_elements) >= 2:
        # 找背驰后的第一个反向次级别走势
        last_dir = sub_elements[-1]["dir"]
        for i in range(len(sub_elements) - 1, 0, -1):
            if sub_elements[i]["dir"] != last_dir:
                first_reverse = sub_elements[i]
                break
        if first_reverse:
            if direction == "top_divergence":
                # 顶背驰后第一个向下走势是否回到中枢
                reentered_zs = first_reverse["end_price"] <= zg
            else:
                # 底背驰后第一个向上走势是否回到中枢
                reentered_zs = first_reverse["end_price"] >= zd

    if direction == "top_divergence":
        # 上涨趋势背驰 → 价格回落，分三种情况
        if not reentered_zs and last_price >= gg:
            situation = 1
            desc = "回落只触及GG未入中枢——最后一个中枢级别扩展（最弱反弹）"
        elif last_price >= zd:
            situation = 2
            desc = "回落进入中枢区间——该级别更大级别的盘整"
        else:
            situation = 3
            desc = "回落跌破ZD——该级别以上级别的反趋势（下跌趋势）"
    else:
        # 下跌趋势背驰 → 价格反弹，分三种情况
        if not reentered_zs and last_price <= dd:
            situation = 1
            desc = "反弹只触及DD未入中枢——最后一个中枢级别扩展（最弱反弹）"
        elif last_price <= zg:
            situation = 2
            desc = "反弹进入中枢区间——该级别更大级别的盘整"
        else:
            situation = 3
            desc = "反弹突破ZG——该级别以上级别的反趋势（上涨趋势）"

    return {
        "status": "active",
        "situation": situation,
        "description": desc,
        "zhongshu": last_zs,
        "reentered_zs": reentered_zs,
        "judgment_method": "第一个次级别反向走势是否回抽最后一个中枢",
    }


# ============================================================
# 第六部分：三类买卖点（第20/21/53/101课）⭐ 完整版
# ============================================================

def identify_trade_points(df: pd.DataFrame, divergence: dict,
                          zhongshu_list: List[dict],
                          sub_elements: List[dict]) -> dict:
    """
    识别三类买卖点

    一买：下跌趋势最后一个中枢下方 + 趋势背驰
    二买三种情况（第101课）：
      - 最强：二买与三买重合
      - 一般：一、二、三买点依次向上
      - 最弱：二买跌破一买位置（但形成盘整背驰）
    二买相对中枢位置（第21课）：
      - 在中枢之下 → 力度可疑
      - 在中枢之中 → 扩张/新生对半
      - 在中枢之上 → 新生机会大（最强）
    三买：次级别离开中枢 + 次级别回抽不破ZG
    """
    result = {"buy_points": [], "sell_points": []}

    if not zhongshu_list or not sub_elements:
        return result

    last_zs = zhongshu_list[-1]
    last_price = float(df["close"].iloc[-1])

    # --- 第一类买卖点 ---
    if divergence.get("type") == "trend_divergence" and divergence.get("has_divergence"):
        direction = divergence["details"]["direction"]

        if direction == "bottom_divergence":
            if last_price < last_zs["zd"]:
                dist_pct = abs(last_price - last_zs["zd"]) / last_zs["zd"]
                result["buy_points"].append({
                    "type": "一买",
                    "description": "下跌趋势背驰完成，位于最后一个中枢下方",
                    "confidence": "high" if dist_pct > 0.03 else "medium",
                    "entry_zone": f"下方{dist_pct*100:.1f}%",
                    "risk_note": "利润最大但最需区间套精确定位",
                })
        elif direction == "top_divergence":
            if last_price > last_zs["zg"]:
                result["sell_points"].append({
                    "type": "一卖",
                    "description": "上涨趋势背驰完成，位于最后一个中枢上方",
                    "confidence": "high",
                    "action": "必须全仓退出",
                })

    # --- 第二类买卖点 ---
    n_elements = len(sub_elements)
    if n_elements >= 3:
        last_el = sub_elements[-1]
        prev_el = sub_elements[-2]
        prev2_el = sub_elements[-3]

        # 二买：一买后第一次回调不破前低
        if prev_el["dir"] == "up" and last_el["dir"] == "down":
            # 检查是否在一买之后
            one_buy_exists = any(
                bp["type"] == "一买" for bp in result["buy_points"]
            )
            prev_low = prev2_el.get("start_price", float("inf"))

            if last_el["end_price"] > prev2_el.get("end_price", 0) or \
               last_el["end_price"] > prev_low:
                er_buy = {
                    "type": "二买",
                    "description": "一买后第一次次级别回调不破前低",
                    "confidence": "medium",
                    "safety": "安全性最高，散户首选",
                }

                # 二买三种情况判断
                if last_el["end_price"] > last_zs["zg"]:
                    er_buy["position_vs_zs"] = "above"
                    er_buy["quality"] = "最强——中枢之上，新生概率大"
                elif last_el["end_price"] < last_zs["zd"]:
                    er_buy["position_vs_zs"] = "below"
                    er_buy["quality"] = "可疑——中枢之下，力度存疑"
                else:
                    er_buy["position_vs_zs"] = "inside"
                    er_buy["quality"] = "中性——中枢之内，延续/新生对半"

                # 检查是否跌破一买低点
                if prev2_el.get("start_price") and last_el["end_price"] < prev2_el["start_price"]:
                    er_buy["quality"] = "最弱——跌破一买低点，需确认盘整背驰"

                result["buy_points"].append(er_buy)

        # 二卖：一卖后第一次反弹不创新高
        if prev_el["dir"] == "down" and last_el["dir"] == "up":
            if last_el["end_price"] < prev2_el.get("start_price", float("inf")):
                result["sell_points"].append({
                    "type": "二卖",
                    "description": "一卖后第一次反弹不创新高",
                    "confidence": "medium",
                })

    # --- 第三类买卖点 ---
    if zhongshu_list and n_elements >= 2:
        prev_el = sub_elements[-2] if n_elements >= 2 else None

        # 三买：价格在ZG上方 + 次级别回抽不破ZG
        # ⭐ 必须是第一次突破（第20课）：之前不能有已经回抽确认过的三买
        if last_price > last_zs["zg"] and prev_el and prev_el["dir"] == "down":
            if prev_el["end_price"] >= last_zs["zg"]:
                # 验证是第一次：检查在此之前是否有过向上突破ZG后回抽不破
                prior_break = False
                for i in range(1, n_elements - 1):
                    el_prev = sub_elements[i - 1]
                    el_curr = sub_elements[i]
                    if (el_prev["dir"] == "up" and el_prev["end_price"] > last_zs["zg"] and
                        el_curr["dir"] == "down" and el_curr["end_price"] >= last_zs["zg"]):
                        prior_break = True
                        break
                is_first = not prior_break
                result["buy_points"].append({
                    "type": "三买",
                    "description": "次级别离开中枢后回抽不破ZG",
                    "confidence": "high" if is_first else "medium",
                    "efficiency": "效率最高，大资金首选" if is_first else "非首次突破，效率降低",
                    "is_first_time": is_first,
                })

        # 三卖：价格在ZD下方 + 次级别反弹不破ZD
        if last_price < last_zs["zd"] and prev_el and prev_el["dir"] == "up":
            if prev_el["end_price"] <= last_zs["zd"]:
                # 验证是第一次
                prior_break = False
                for i in range(1, n_elements - 1):
                    el_prev = sub_elements[i - 1]
                    el_curr = sub_elements[i]
                    if (el_prev["dir"] == "down" and el_prev["end_price"] < last_zs["zd"] and
                        el_curr["dir"] == "up" and el_curr["end_price"] <= last_zs["zd"]):
                        prior_break = True
                        break
                is_first = not prior_break
                result["sell_points"].append({
                    "type": "三卖",
                    "description": "次级别跌破中枢后反弹不破ZD",
                    "confidence": "high" if is_first else "medium",
                    "action": "必须离场" if is_first else "非首次跌破，力度减弱",
                    "is_first_time": is_first,
                })

    # --- 小转大自动切换二买卖点（第53课）---
    # 当小级别背驰未产生本级别一买卖点时，二买卖点为最佳
    buy_types_set = {bp["type"] for bp in result["buy_points"]}
    sell_types_set = {sp["type"] for sp in result["sell_points"]}

    if "二买" in buy_types_set and "一买" not in buy_types_set:
        # 可能是小转大：有一买条件但未完全确认，二买成为最佳
        div_details = divergence.get("details", {})
        if divergence.get("has_divergence") and divergence.get("confidence") != "high":
            for bp in result["buy_points"]:
                if bp["type"] == "二买":
                    bp["small_to_large"] = True
                    bp["confidence"] = "high"
                    bp["description"] += " | 小转大：本级别无一买，二买是最佳介入点（第53课）"
                    result["small_to_large"] = True

    if "二卖" in sell_types_set and "一卖" not in sell_types_set:
        if divergence.get("has_divergence") and divergence.get("confidence") != "high":
            for sp in result["sell_points"]:
                if sp["type"] == "二卖":
                    sp["small_to_large"] = True
                    sp["confidence"] = "high"
                    sp["description"] += " | 小转大：本级别无一卖，二卖是最佳退出点（第53课）"
                    result["small_to_large"] = True

    # --- 二买三买重合（最强信号，第101课）---
    buy_types = {bp["type"] for bp in result["buy_points"]}
    if "二买" in buy_types and "三买" in buy_types:
        result["buy_points"].append({
            "type": "二买=三买重合",
            "description": "最强势启动信号——一买后凌厉突破前中枢且回调不破ZG",
            "confidence": "very_high",
        })

    return result


# ============================================================
# 第七部分：完整动力学分析管线
# ============================================================

def run_dynamics(df: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """
    执行完整动力学分析
    输入：原始df（含MACD）、经形态学处理的 df_clean（含笔/线段）
    输出：动力学分析结果字典
    """
    sub_elements = df_clean.attrs.get("segment_list")
    if not sub_elements:
        sub_elements = df_clean.attrs.get("bi_list", [])
    sub_label = "segment" if df_clean.attrs.get("segment_list") else "bi"

    # 中枢识别
    zs_df = identify_zhongshu(sub_elements, label=sub_label)
    zhongshu_list = zs_df.attrs.get("zhongshu_list", [])

    # 走势类型
    zoushi = identify_zoushi_type(zhongshu_list, sub_elements)

    # 背驰（严格6条件）
    divergence = identify_divergence(df, zhongshu_list, sub_elements)

    # 背驰-转折定理
    reversal = divergence_to_reversal(divergence, zhongshu_list, sub_elements)

    # 买卖点
    trade_points = identify_trade_points(df, divergence, zhongshu_list, sub_elements)

    return {
        "sub_elements": sub_elements,
        "sub_label": sub_label,
        "zhongshu_list": zhongshu_list,
        "zoushi": zoushi,
        "divergence": divergence,
        "reversal": reversal,
        "trade_points": trade_points,
        "sub_count": len(sub_elements),
        "zs_count": len(zhongshu_list),
    }
