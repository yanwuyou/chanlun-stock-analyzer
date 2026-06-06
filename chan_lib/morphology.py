#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
形态学模块（纯几何，无需 MACD）
管线：包含关系处理 → 分型识别 → 笔的划分 → 线段
基于第62/65/67/71/77/78/82课
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


# ============================================================
# 第一部分：包含关系处理（第62、65课）
# ============================================================

def process_inclusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理K线包含关系（顺序处理，不满足传递律）
    向上时：高高为高，低高为低
    向下时：低低为低，高低为高
    """
    df = df.copy()
    n = len(df)
    df["_merged_from"] = [0] * n

    i = 1
    while i < n:
        prev_idx = i - 1
        while prev_idx >= 0 and df.loc[prev_idx, "_merged_from"] < 0:
            prev_idx -= 1
        if prev_idx < 0:
            i += 1
            continue

        curr_h, curr_l = df.loc[i, "high"], df.loc[i, "low"]
        prev_h, prev_l = df.loc[prev_idx, "high"], df.loc[prev_idx, "low"]

        is_inclusion = (curr_h <= prev_h and curr_l >= prev_l) or \
                       (curr_h >= prev_h and curr_l <= prev_l)

        if is_inclusion:
            if df.loc[prev_idx, "_merged_from"] < 0:
                i += 1
                continue

            prev2_idx = prev_idx - 1
            while prev2_idx >= 0 and df.loc[prev2_idx, "_merged_from"] < 0:
                prev2_idx -= 1

            direction = "up" if prev2_idx < 0 else \
                ("up" if df.loc[prev_idx, "high"] >= df.loc[prev2_idx, "high"] else "down")

            if direction == "up":
                df.loc[prev_idx, "high"] = max(curr_h, prev_h)
                df.loc[prev_idx, "low"] = max(curr_l, prev_l)
            else:
                df.loc[prev_idx, "high"] = min(curr_h, prev_h)
                df.loc[prev_idx, "low"] = min(curr_l, prev_l)

            df.loc[prev_idx, "_merged_from"] += 1
            df.loc[i, "_merged_from"] = -1
        i += 1

    result = df[df["_merged_from"] >= 0].copy()
    result.reset_index(drop=True, inplace=True)
    return result


# ============================================================
# 第二部分：分型识别（第62、82课）
# ============================================================

def identify_fractals(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别顶分型和底分型
    顶分型：三根相邻K线，中间高点最高、低点最高
    底分型：三根相邻K线，中间低点最低、高点最低
    """
    df = df.copy()
    df["fractal"] = None
    df["fractal_high"] = np.nan
    df["fractal_low"] = np.nan
    df["fractal_strength"] = None  # 分型强弱：strong / weak / neutral

    n = len(df)
    for i in range(1, n - 1):
        h0, h1, h2 = df.loc[i - 1, "high"], df.loc[i, "high"], df.loc[i + 1, "high"]
        l0, l1, l2 = df.loc[i - 1, "low"], df.loc[i, "low"], df.loc[i + 1, "low"]

        if h1 > h0 and h1 > h2 and l1 >= l0 and l1 >= l2:
            df.loc[i, "fractal"] = "top"
            df.loc[i, "fractal_high"] = h1
            df.loc[i, "fractal_low"] = l1
            df.loc[i, "fractal_strength"] = _fractal_strength(df, i, "top")

        if l1 < l0 and l1 < l2 and h1 <= h0 and h1 <= h2:
            df.loc[i, "fractal"] = "bottom"
            df.loc[i, "fractal_high"] = h1
            df.loc[i, "fractal_low"] = l1
            df.loc[i, "fractal_strength"] = _fractal_strength(df, i, "bottom")

    return df


def _fractal_strength(df: pd.DataFrame, idx: int, ftype: str) -> str:
    """
    判断分型强弱（第82课完整版）

    顶分型力度判断：
    1. 强（延续成笔概率大）：
       a) 第2根长上影/长阴 + 第3根收不到第2根区间一半以上
       b) 第3根跌破第1根底 + 收不到第1根区间一半以上 → 最弱分型，杀伤力最强
    2. 弱（中继概率大）：
       第1根长阳 + 第2、3根小阴小阳
    3. 包含关系如果是长阴吞阳线 → 最坏的一种

    底分型：对称反向
    """
    if idx < 1 or idx >= len(df) - 1:
        return "neutral"

    c0, o0, h0, l0 = df.loc[idx - 1, "close"], df.loc[idx - 1, "open"], df.loc[idx - 1, "high"], df.loc[idx - 1, "low"]
    c1, o1, h1, l1 = df.loc[idx, "close"], df.loc[idx, "open"], df.loc[idx, "high"], df.loc[idx, "low"]
    c2, o2, h2, l2 = df.loc[idx + 1, "close"], df.loc[idx + 1, "open"], df.loc[idx + 1, "high"], df.loc[idx + 1, "low"]

    bar0_range = abs(c0 - o0)
    bar1_range = abs(c1 - o1)
    bar2_range = abs(c2 - o2)

    # 第2根K线区间 [low, high]
    bar1_low = min(o1, c1)
    bar1_high = max(o1, c1)
    bar1_mid = (bar1_low + bar1_high) / 2  # 第2根区间的半分位

    # 第1根K线区间
    bar0_low = min(o0, c0)
    bar0_high = max(o0, c0)
    bar0_mid = (bar0_low + bar0_high) / 2

    if ftype == "top":
        upper_shadow = h1 - max(o1, c1)
        real_body = abs(c1 - o1)

        # 条件A：长上影/长阴 + 第3根收不到第2根一半以上
        is_long_upper = upper_shadow > real_body * 1.2  # 长上影
        is_long_bearish = c1 < o1 and bar1_range > bar0_range * 1.5  # 长阴线
        third_not_above_half = c2 < bar1_mid  # 第3根收不到第2根一半

        if (is_long_upper or is_long_bearish) and third_not_above_half:
            return "strong"

        # 条件B：第3根跌破第1根底 + 收不到第1根一半以上（最弱分型，杀伤力最强）
        third_broke_bar0_low = l2 < l0  # 或 c2 < bar0_low
        third_not_above_bar0_half = c2 < bar0_mid
        if third_broke_bar0_low and third_not_above_bar0_half:
            return "strongest"  # 最强杀伤力——第82课所说"最弱的一种"

        # 条件C：中继分型——第1根长阳 + 第2、3根小阴小阳
        is_bar0_long_yang = c0 > o0 and bar0_range > bar1_range * 1.5
        bar2_small = bar2_range < bar1_range * 0.5
        if is_bar0_long_yang and bar2_small:
            return "weak"

        # 条件D：第3根收上第2根一半以上 → 分型力度不足
        if c2 >= bar1_mid and c2 > o2:
            return "weak"

    else:  # bottom fractal
        lower_shadow = min(o1, c1) - l1
        real_body = abs(c1 - o1)

        # 条件A：长下影/长阳 + 第3根收不到第2根一半以下
        is_long_lower = lower_shadow > real_body * 1.2
        is_long_bullish = c1 > o1 and bar1_range > bar0_range * 1.5
        third_not_below_half = c2 > bar1_mid

        if (is_long_lower or is_long_bullish) and third_not_below_half:
            return "strong"

        # 条件B：第3根突破第1根顶 + 收不到第1根一半以下（最强底分型）
        third_broke_bar0_high = h2 > h0
        third_not_below_bar0_half = c2 > bar0_mid
        if third_broke_bar0_high and third_not_below_bar0_half:
            return "strongest"

        # 条件C：中继分型——第1根长阴 + 第2、3根小阴小阳
        is_bar0_long_yin = c0 < o0 and bar0_range > bar1_range * 1.5
        bar2_small = bar2_range < bar1_range * 0.5
        if is_bar0_long_yin and bar2_small:
            return "weak"

        # 条件D：第3根收到第2根一半以下 → 分型力度不足
        if c2 <= bar1_mid and c2 < o2:
            return "weak"

    return "neutral"


# ============================================================
# 第三部分：笔的划分（第62、65、77课）
# ============================================================

def identify_bi(df: pd.DataFrame) -> pd.DataFrame:
    """
    从分型构建笔（三步法 - 第77课）
    Step 1: 同向分型去重（顶取高，底取低）
    Step 2: 确保一顶一底交替
    Step 3: 构建笔（顶底间至少一根独立K线）
    """
    df = df.copy()
    df["bi"] = np.nan
    df["bi_dir"] = None
    df["bi_start"] = False
    df["bi_end"] = False

    fractal_list = []
    for idx in df.index:
        if df.loc[idx, "fractal"] in ("top", "bottom"):
            fractal_list.append({
                "idx": int(idx),
                "type": df.loc[idx, "fractal"],
                "high": df.loc[idx, "fractal_high"],
                "low": df.loc[idx, "fractal_low"],
                "date": df.loc[idx, "date"],
                "strength": df.loc[idx, "fractal_strength"],
            })

    if len(fractal_list) < 2:
        return df

    # Step 1: 同向分型去重
    filtered = [fractal_list[0]]
    for curr in fractal_list[1:]:
        last = filtered[-1]
        if curr["type"] == last["type"]:
            if curr["type"] == "top":
                if curr["high"] >= last["high"]:
                    filtered[-1] = curr
            else:
                if curr["low"] <= last["low"]:
                    filtered[-1] = curr
        else:
            filtered.append(curr)

    # Step 2: 确保一顶一底交替
    while len(filtered) >= 2 and filtered[0]["type"] == filtered[1]["type"]:
        if filtered[0]["type"] == "top":
            if filtered[0]["high"] < filtered[1]["high"]:
                filtered.pop(0)
            else:
                filtered.pop(1)
        else:
            if filtered[0]["low"] > filtered[1]["low"]:
                filtered.pop(0)
            else:
                filtered.pop(1)

    # Step 3: 构建笔
    bi_list = []
    i = 0
    while i < len(filtered) - 1:
        a, b = filtered[i], filtered[i + 1]
        if a["type"] == b["type"]:
            i += 1
            continue
        if abs(b["idx"] - a["idx"]) < 2:
            i += 1
            continue

        if a["type"] == "bottom" and b["type"] == "top":
            if b["high"] > a["low"]:
                pct = (b["high"] - a["low"]) / a["low"]
                bi_list.append({
                    "start_idx": a["idx"], "end_idx": b["idx"],
                    "dir": "up", "start_date": a["date"], "end_date": b["date"],
                    "start_price": a["low"], "end_price": b["high"],
                    "amplitude": pct, "duration": (b["date"] - a["date"]).days,
                    "start_fractal_type": "bottom", "end_fractal_type": "top",
                    "start_fractal_strength": a.get("strength", "neutral"),
                })
                i += 2
                continue
        elif a["type"] == "top" and b["type"] == "bottom":
            if b["low"] < a["high"]:
                pct = (a["high"] - b["low"]) / a["high"]
                bi_list.append({
                    "start_idx": a["idx"], "end_idx": b["idx"],
                    "dir": "down", "start_date": a["date"], "end_date": b["date"],
                    "start_price": a["high"], "end_price": b["low"],
                    "amplitude": pct, "duration": (b["date"] - a["date"]).days,
                    "start_fractal_type": "top", "end_fractal_type": "bottom",
                    "start_fractal_strength": a.get("strength", "neutral"),
                })
                i += 2
                continue
        i += 1

    for j, bi in enumerate(bi_list):
        start, end = bi["start_idx"], bi["end_idx"]
        for k in range(start, min(end + 1, len(df))):
            df.loc[k, "bi"] = j
            df.loc[k, "bi_dir"] = bi["dir"]
        df.loc[start, "bi_start"] = True
        df.loc[min(end, len(df) - 1), "bi_end"] = True

    df.attrs["bi_list"] = bi_list
    return df


# ============================================================
# 第四部分：线段（第62/65/67/71/78课）⭐ 完整实现
# ============================================================

def _bi_price_range(bi: dict) -> Tuple[float, float]:
    """获取笔的价格区间 [low, high]"""
    if bi["dir"] == "up":
        return bi["start_price"], bi["end_price"]
    else:
        return bi["end_price"], bi["start_price"]


def _has_overlap(r1: Tuple[float, float], r2: Tuple[float, float]) -> bool:
    """两个区间是否有重叠"""
    return max(r1[0], r2[0]) < min(r1[1], r2[1])


def _feature_element(bi: dict, seg_dir: str, bi_list_pos: int) -> dict:
    """
    将笔转换为特征序列元素
    向上线段 → 特征序列是向下笔（特征元素的高=start_price, 低=end_price）
    向下线段 → 特征序列是向上笔（特征元素的高=end_price, 低=start_price）

    每个特征元素表示一个价格区间 [low, high]：
    - 向下笔（向上线段的特征元素）：high=start_price, low=end_price
    - 向上笔（向下线段的特征元素）：high=end_price, low=start_price

    bi_list_pos: 这笔在 bi_list 中的位置索引（用于线段起止点计算）
    """
    if bi["dir"] == "down":
        return {"high": bi["start_price"], "low": bi["end_price"],
                "bi_pos": [bi_list_pos],
                "start_date": bi["start_date"], "end_date": bi["end_date"],
                "dir": "down"}
    else:
        return {"high": bi["end_price"], "low": bi["start_price"],
                "bi_pos": [bi_list_pos],
                "start_date": bi["start_date"], "end_date": bi["end_date"],
                "dir": "up"}


def _process_feature_inclusion(features: List[dict], seg_dir: str) -> List[dict]:
    """
    对特征序列做包含处理
    向上线段的特征序列（向下笔）：方向判断用的参照系是线段方向
      - 向上(seg_dir=up) → 高高为高，低高为低
      - 向下(seg_dir=down) → 低低为低，高低为高

    判断"向上/向下"的标准：
      比较当前元素和前一个元素的高点
      - 如果当前高点 >= 前高点 → 向上
      - 如果当前低点 <= 前低点 → 向下
    """
    if len(features) < 2:
        return features

    result = [features[0]]
    i = 1
    while i < len(features):
        curr = features[i]
        prev = result[-1]

        is_inclusion = (curr["high"] <= prev["high"] and curr["low"] >= prev["low"]) or \
                       (curr["high"] >= prev["high"] and curr["low"] <= prev["low"])

        if is_inclusion:
            # 判断方向：用线段方向作为参照
            # 向上线段: 比较高点
            # 向下线段: 比较低点
            if seg_dir == "up":
                direction = "up" if curr["high"] >= prev["high"] else "down"
            else:
                direction = "down" if curr["low"] <= prev["low"] else "up"

            if direction == "up":
                result[-1] = {
                    "high": max(curr["high"], prev["high"]),
                    "low": max(curr["low"], prev["low"]),
                    "bi_pos": prev["bi_pos"] + curr["bi_pos"],
                    "start_date": prev["start_date"],
                    "end_date": curr["end_date"],
                    "dir": "up",
                }
            else:
                result[-1] = {
                    "high": min(curr["high"], prev["high"]),
                    "low": min(curr["low"], prev["low"]),
                    "bi_pos": prev["bi_pos"] + curr["bi_pos"],
                    "start_date": prev["start_date"],
                    "end_date": curr["end_date"],
                    "dir": "down",
                }
        else:
            result.append(curr)
        i += 1

    return result


def _find_feature_fractal(features: List[dict], ftype: str) -> Optional[int]:
    """
    在标准特征序列中找分型
    ftype="top": 找顶分型（中间高点最高、低点最高）
    ftype="bottom": 找底分型（中间低点最低、高点最低）
    返回分型的中间元素索引，找不到返回 None
    """
    if len(features) < 3:
        return None

    for i in range(1, len(features) - 1):
        h0, h1, h2 = features[i - 1]["high"], features[i]["high"], features[i + 1]["high"]
        l0, l1, l2 = features[i - 1]["low"], features[i]["low"], features[i + 1]["low"]

        if ftype == "top":
            if h1 > h0 and h1 > h2 and l1 >= l0 and l1 >= l2:
                return i
        else:
            if l1 < l0 and l1 < l2 and h1 <= h0 and h1 <= h2:
                return i

    return None


def identify_segments(bi_list: List[dict]) -> List[dict]:
    """
    从笔构建线段（第67/71/78课特征序列法）

    严格按原文：
      假设某转折点是两线段的分界点，对此用两种情况进行考察。
      特征序列分型中：
        第一元素 = 假设转折点前原线段的最后一个特征元素
        第二元素 = 从转折点开始的第一笔（与特征序列同向，即与线段反向）
        两元素间有缺口 → 第二种情况；无缺口 → 第一种情况

      第一种情况（无缺口）：线段在该分型的高/低点结束，无需等待
      第二种情况（有缺口）：必须从分型高/低点开始的序列中，
                           其反向特征序列出现分型，线段才在该高/低点结束
    """
    if len(bi_list) < 3:
        return []

    segments = []
    seg_start = 0

    while seg_start < len(bi_list) - 2:
        b0, b1, b2 = bi_list[seg_start], bi_list[seg_start + 1], bi_list[seg_start + 2]
        r0, r1, r2 = _bi_price_range(b0), _bi_price_range(b1), _bi_price_range(b2)

        if max(r0[0], r1[0], r2[0]) >= min(r0[1], r1[1], r2[1]):
            seg_start += 1
            continue

        seg_dir = b0["dir"]
        actual_end = _find_segment_end(bi_list, seg_start, seg_dir)
        actual_end = max(actual_end, seg_start + 2)
        actual_end = min(actual_end, len(bi_list) - 1)

        segments.append({
            "start_bi": seg_start,
            "end_bi": actual_end,
            "start_idx": bi_list[seg_start]["start_idx"],
            "end_idx": bi_list[actual_end]["end_idx"],
            "dir": seg_dir,
            "start_price": bi_list[seg_start]["start_price"],
            "end_price": bi_list[actual_end]["end_price"],
            "start_date": bi_list[seg_start]["start_date"],
            "end_date": bi_list[actual_end]["end_date"],
            "bi_count": actual_end - seg_start + 1,
        })

        seg_start = actual_end + 1

    return segments


def _find_segment_end(bi_list: List[dict], seg_start: int, seg_dir: str) -> int:
    """
    按第67/71课标准确定线段终点

    逐个假设同向笔创新高/低的位置为转折点，
    检查特征序列是否在该处形成分型，判断有无缺口，
    按第一种或第二种情况确定线段终止位置。
    """
    n = len(bi_list)
    # 跟踪当前线段到哪个同向笔为止
    current_end = seg_start + 2
    # 特征序列（与线段方向相反的笔），记录其在 bi_list 中的位置
    feat_positions = []
    for k in range(seg_start, seg_start + 3):
        if bi_list[k]["dir"] != seg_dir:
            feat_positions.append(k)

    j = seg_start + 3
    while j < n:
        bi = bi_list[j]

        if bi["dir"] == seg_dir:
            # 同向笔：检查是否创新高/低 → 这可能是转折点
            is_new_extreme = (
                (seg_dir == "up" and bi["end_price"] > bi_list[current_end]["end_price"]) or
                (seg_dir == "down" and bi["end_price"] < bi_list[current_end]["end_price"])
            )
            if is_new_extreme:
                # 假设 j 是转折点 → 构造/更新到此的特征序列
                feat_positions = [p for p in feat_positions if p < j]
                for k in range(max(feat_positions[-1] + 1 if feat_positions else seg_start, seg_start), j):
                    if bi_list[k]["dir"] != seg_dir and k not in feat_positions:
                        feat_positions.append(k)

                # 用特征序列检查是否形成分型，以及有无缺口
                feat_elements = [_feature_element(bi_list[p], seg_dir, p)
                                 for p in feat_positions]
                standard = _process_feature_inclusion(feat_elements, seg_dir)

                ftype = "top" if seg_dir == "up" else "bottom"
                fractal_idx = _find_feature_fractal(standard, ftype)

                if fractal_idx is not None and fractal_idx >= 1:
                    f0, f1 = standard[fractal_idx - 1], standard[fractal_idx]
                    if seg_dir == "up":
                        has_gap = f0["high"] < f1["low"]
                    else:
                        has_gap = f1["high"] < f0["low"]

                    if not has_gap:
                        # 第一种情况：线段在此结束
                        end_candidate = max(f["bi_pos"][-1]
                                            for f in standard[:fractal_idx + 1])
                        return end_candidate
                    else:
                        # 第二种情况：需等反向特征序列出现分型
                        confirmed = _wait_reverse_confirmation(
                            bi_list, j, seg_dir, feat_positions, fractal_idx, standard
                        )
                        if confirmed is not None:
                            return confirmed

                current_end = j
            j += 1
        else:
            # 反向笔 → 加入特征序列
            if j not in feat_positions:
                feat_positions.append(j)
            j += 1

    return current_end


def _wait_reverse_confirmation(bi_list: List[dict], turning_bi: int,
                               seg_dir: str, feat_positions: List[int],
                               fractal_idx: int, standard: List[dict]) -> Optional[int]:
    """
    第二种情况（有缺口）：从转折点开始的序列中，
    其反向特征序列出现分型后，线段在分型高/低点结束。

    turning_bi: 假设的转折点笔位置
    返回确认后的线段终点位置，或 None（尚未确认）
    """
    n = len(bi_list)
    # 分型对应的高/低点位置
    fractal_end = max(f["bi_pos"][-1] for f in standard[:fractal_idx + 1])

    # 从转折点向后扫描，提取反向特征序列
    # 向上线段 → 原特征序列是向下笔 → 反向特征序列是向上笔
    # 向下线段 → 原特征序列是向上笔 → 反向特征序列是向下笔
    reverse_dir = "down" if seg_dir == "up" else "up"

    # 收集从转折点开始的笔中，与反向特征序列同向的笔（即与原线段同向）
    reverse_feat = []
    for j in range(turning_bi, n):
        bi = bi_list[j]
        if bi["dir"] == seg_dir:  # 与原线段同向的笔 = 反向特征序列的元素
            reverse_feat.append(_feature_element(bi, seg_dir, j))

    if len(reverse_feat) < 3:
        return None  # 尚未形成足够元素

    # 包含处理后找反向分型
    reverse_standard = _process_feature_inclusion(reverse_feat, reverse_dir)
    target_fractal = "bottom" if seg_dir == "up" else "top"
    rev_idx = _find_feature_fractal(reverse_standard, target_fractal)

    if rev_idx is not None:
        # 反向特征序列出现分型 → 原线段在分型高/低点结束
        return fractal_end

    return None  # 尚未确认，线段延续


# ============================================================
# 线段标准化（第78课）
# ============================================================

def standardize_segments(segment_list: List[dict],
                         bi_list: List[dict]) -> List[dict]:
    """
    线段标准化（第78课）

    如果线段中最高/最低点不是线段的端点，标准化为端点都在高低点处。
    - 向上线段：最低点开始，最高点结束
    - 向下线段：最高点开始，最低点结束

    标准化后，线段成为首尾相连的折线，为中枢分析提供标准基础部件。
    """
    if not segment_list or not bi_list:
        return segment_list

    standardized = []
    for seg in segment_list:
        s = dict(seg)  # 复制
        start_bi = seg["start_bi"]
        end_bi = seg["end_bi"]

        # 计算线段内部的实际最高/最低点
        seg_high = seg.get("start_price", 0)
        seg_low = seg.get("start_price", 0)
        seg_high_idx = start_bi
        seg_low_idx = start_bi

        for k in range(start_bi, min(end_bi + 1, len(bi_list))):
            bi = bi_list[k]
            if bi["dir"] == "up":
                if bi["end_price"] > seg_high:
                    seg_high = bi["end_price"]
                    seg_high_idx = k
                if bi["start_price"] < seg_low:
                    seg_low = bi["start_price"]
                    seg_low_idx = k
            else:
                if bi["start_price"] > seg_high:
                    seg_high = bi["start_price"]
                    seg_high_idx = k
                if bi["end_price"] < seg_low:
                    seg_low = bi["end_price"]
                    seg_low_idx = k

        s["actual_high"] = seg_high
        s["actual_low"] = seg_low
        s["high_at_endpoint"] = (seg_high_idx == start_bi or seg_high_idx == end_bi)
        s["low_at_endpoint"] = (seg_low_idx == start_bi or seg_low_idx == end_bi)
        s["needs_standardization"] = not (s["high_at_endpoint"] and s["low_at_endpoint"])

        if s["needs_standardization"]:
            if seg["dir"] == "up":
                s["start_price"] = seg_low
                s["end_price"] = seg_high
            else:
                s["start_price"] = seg_high
                s["end_price"] = seg_low

        standardized.append(s)

    return standardized


# ============================================================
# 缺口分类体系（第77课）
# ============================================================

def classify_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    缺口分类（第77课）

    缺口定义：相邻两K线间无成交的区间
    分类：
    - 突破性缺口：不回补 → 强势，极少回补
    - 中继性缺口：回补后继续新高/新低 → 平势，回补几率对半
    - 衰竭性缺口：回补后不能新高/新低 → 弱势，至少造成大一级别调整

    注意：级别与K线图周期无关，只与走势类型级别有关。
    """
    df = df.copy()
    n = len(df)
    df["gap"] = None
    df["gap_type"] = None
    df["gap_range"] = None
    df["gap_filled"] = False

    for i in range(1, n):
        prev_high = df.loc[i - 1, "high"]
        prev_low = df.loc[i - 1, "low"]
        curr_high = df.loc[i, "high"]
        curr_low = df.loc[i, "low"]

        # 向上缺口：当前最低 > 前最高
        if curr_low > prev_high:
            gap_lo = prev_high
            gap_hi = curr_low
            df.loc[i, "gap"] = "up"
            df.at[i, "gap_range_lo"] = gap_lo
            df.at[i, "gap_range_hi"] = gap_hi

            # 检查是否回补：后续K线是否有价格进入缺口区间
            filled = False
            new_high_after_fill = False
            for j in range(i + 1, min(i + 50, n)):
                if df.loc[j, "low"] <= gap_hi:
                    filled = True
                    # 回补后是否继续创新高
                    future_highs = df.loc[j:, "high"].max()
                    new_high_after_fill = future_highs > df.loc[i - 1, "high"]
                    break

            if not filled:
                df.loc[i, "gap_type"] = "breakaway"  # 突破性缺口
            elif new_high_after_fill:
                df.loc[i, "gap_type"] = "continuation"  # 中继性缺口
            else:
                df.loc[i, "gap_type"] = "exhaustion"  # 衰竭性缺口
            df.loc[i, "gap_filled"] = filled

        # 向下缺口：当前最高 < 前最低
        elif curr_high < prev_low:
            gap_lo = curr_high
            gap_hi = prev_low
            df.loc[i, "gap"] = "down"
            df.at[i, "gap_range_lo"] = gap_lo
            df.at[i, "gap_range_hi"] = gap_hi

            filled = False
            new_low_after_fill = False
            for j in range(i + 1, min(i + 50, n)):
                if df.loc[j, "high"] >= gap_lo:
                    filled = True
                    future_lows = df.loc[j:, "low"].min()
                    new_low_after_fill = future_lows < df.loc[i - 1, "low"]
                    break

            if not filled:
                df.loc[i, "gap_type"] = "breakaway"
            elif new_low_after_fill:
                df.loc[i, "gap_type"] = "continuation"
            else:
                df.loc[i, "gap_type"] = "exhaustion"
            df.loc[i, "gap_filled"] = filled

    return df


# ============================================================
# 第五部分：完整形态学分析管线
# ============================================================

def run_morphology(df: pd.DataFrame) -> pd.DataFrame:
    """
    执行完整的形态学分析管线
    包含处理 → 分型 → 笔 → 线段 → 标准化 → 缺口分类
    """
    df_clean = process_inclusion(df)
    df_clean = identify_fractals(df_clean)
    df_clean = identify_bi(df_clean)
    df_clean = classify_gaps(df_clean)

    bi_list = df_clean.attrs.get("bi_list", [])
    if len(bi_list) >= 3:
        segment_list = identify_segments(bi_list)
        segment_list = standardize_segments(segment_list, bi_list)
        df_clean.attrs["segment_list"] = segment_list
    else:
        df_clean.attrs["segment_list"] = []

    return df_clean
