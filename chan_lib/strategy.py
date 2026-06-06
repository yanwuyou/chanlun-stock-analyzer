#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略层模块
管线：防狼术 → 两重表里关系（三水平）→ 中枢震荡监视器（Zn体系）→
      中阴阶段 → 区间套 → 同级别分解 → 板块强弱指标 → 多义性分解 → 三买机械化
基于第27/38-40/44/49/61/88-93/99/103/106课
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from .data import fetch_kline
from .dynamics import compute_macd


# ============================================================
# 布林通道（第90课）— 被多个策略函数引用
# ============================================================

def compute_boll(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    """
    计算布林带
    middle = MA(period)
    upper = middle + k * std
    lower = middle - k * std
    返回新增列 boll_mid, boll_upper, boll_lower, boll_width, boll_pct_b
    """
    df = df.copy()
    df["boll_mid"] = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["boll_upper"] = df["boll_mid"] + k * std
    df["boll_lower"] = df["boll_mid"] - k * std
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]
    df["boll_pct_b"] = (df["close"] - df["boll_lower"]) / (df["boll_upper"] - df["boll_lower"])
    return df


def _boll_narrowing(df: pd.DataFrame, lookback: int = 20) -> dict:
    """检测布林带是否在收口（第90课——变盘信号）"""
    if "boll_width" not in df.columns or len(df) < lookback + 20:
        return {"narrowing": False}
    recent_width = df["boll_width"].iloc[-lookback:]
    if recent_width.isna().any():
        return {"narrowing": False}
    hist_width = df["boll_width"].dropna().iloc[-100:]
    if len(hist_width) < 30:
        return {"narrowing": False}
    current = float(recent_width.iloc[-1])
    min_hist = float(hist_width.min())
    pct_rank = (hist_width < current).sum() / len(hist_width)
    narrowing = pct_rank < 0.2 or current < min_hist * 1.1
    return {
        "narrowing": narrowing,
        "current_width": current,
        "min_hist_width": min_hist,
        "pct_rank": pct_rank,
        "signal": "布林收口中——预示变盘在即" if narrowing else None,
    }


# ============================================================
# 第一部分：防狼术（第103课）
# ============================================================

def check_wolf_defense(df: pd.DataFrame) -> dict:
    """防狼术：MACD黄白线在0轴以下的品种，一律不碰"""
    last = df.iloc[-1]
    diff = float(last["diff"])
    dea = float(last["dea"])
    prev_diff = float(df["diff"].iloc[-2]) if len(df) > 1 else diff

    wolf_safe = diff > 0 and dea > 0
    wolf_danger = diff < 0 and dea < 0
    wolf_warning = not wolf_safe and not wolf_danger
    crossed_down = prev_diff > 0 and diff < 0
    crossed_up = prev_diff < 0 and diff > 0

    if wolf_safe:
        status, advice = "safe", "0轴之上，可以操作"
    elif wolf_danger:
        status, advice = "danger", "0轴之下！学屠龙前先学好防狼术——远离，直到重新站回0轴"
    else:
        status, advice = "warning", "临界区域，需要观察"

    return {
        "status": status, "safe": wolf_safe, "danger": wolf_danger,
        "warning": wolf_warning, "crossed_down": crossed_down,
        "crossed_up": crossed_up, "diff": diff, "dea": dea, "advice": advice,
    }


def multi_timeframe_wolf_defense(wolf_map: Dict[str, dict]) -> dict:
    """多周期防狼术联立"""
    daily = wolf_map.get("daily", {})
    weekly = wolf_map.get("weekly", {})
    result = {"daily": daily, "weekly": weekly}

    if weekly.get("danger"):
        result["verdict"] = "禁止操作"
        result["advice"] = "周线MACD在0轴之下——日线操作者应远离此品种"
    elif daily.get("danger"):
        result["verdict"] = "禁止操作"
        result["advice"] = "日线MACD在0轴之下——等站回0轴再说"
    elif daily.get("safe") and weekly.get("safe"):
        result["verdict"] = "可以操作"
        result["advice"] = "日线和周线都在0轴之上，安全边际最高"
    elif daily.get("safe"):
        result["verdict"] = "谨慎操作"
        result["advice"] = "日线0轴上但周线临界，控制仓位"
    else:
        result["verdict"] = "观望"
        result["advice"] = "等待日线MACD确认方向"
    return result


# ============================================================
# 第二部分：两重表里关系（第91-93/99课）
# ============================================================

def check_dual_table_relation(df: pd.DataFrame,
                              bi_list: List[dict] = None,
                              zhongshu_list: List[dict] = None) -> dict:
    """两重表里关系——四种状态编码 (1,1)/(1,0)/(-1,0)/(-1,1)"""
    fractal_df = df[df["fractal"].notna()] if "fractal" in df.columns else pd.DataFrame()

    if bi_list is not None and len(bi_list) > 0:
        last_bi = bi_list.iloc[-1] if hasattr(bi_list, "iloc") else bi_list[-1]
        if last_bi["dir"] == "up":
            if len(fractal_df) > 0 and fractal_df.iloc[-1]["fractal"] == "top":
                days_since = (df["date"].iloc[-1] - fractal_df.iloc[-1]["date"]).days
                if days_since <= 5:
                    return {
                        "state": "(1,0)", "meaning": "向上笔顶分型构造中",
                        "action": "准备卖出", "state_type": "top_forming",
                        "days_in_state": days_since,
                    }
            return {
                "state": "(1,1)", "meaning": "向上笔延伸中",
                "action": "持有", "state_type": "uptrend_active",
            }
        else:
            if len(fractal_df) > 0 and fractal_df.iloc[-1]["fractal"] == "bottom":
                days_since = (df["date"].iloc[-1] - fractal_df.iloc[-1]["date"]).days
                if days_since <= 5:
                    return {
                        "state": "(-1,0)", "meaning": "向下笔底分型构造中",
                        "action": "准备买入", "state_type": "bottom_forming",
                        "days_in_state": days_since,
                    }
            return {
                "state": "(-1,1)", "meaning": "向下笔延伸中",
                "action": "持币等待", "state_type": "downtrend_active",
            }
    return {"state": "unknown", "meaning": "无法判断", "action": "观望"}


def dual_table_level_strategy(df: pd.DataFrame,
                              bi_list: List[dict],
                              divergence: dict,
                              dual_state: dict) -> dict:
    """两重表里关系三水平操作策略（第93课）"""
    result = {"level": 1, "advice": "", "details": {}}

    if "close" in df.columns and len(df) >= 5:
        df["ma5"] = df["close"].rolling(5).mean()
        last_price = float(df["close"].iloc[-1])
        ma5 = float(df["ma5"].iloc[-1])
        result["details"]["ma5"] = ma5
        result["details"]["price_vs_ma5"] = "above" if last_price > ma5 else "below"

    state = dual_state.get("state", "")

    if state == "(1,0)":
        if "ma5" in df.columns:
            last_price = float(df["close"].iloc[-1])
            if last_price < float(df["ma5"].iloc[-1]):
                result["level"] = 1
                result["advice"] = "水平一：跌破5日均线，简单获利了结"
            else:
                result["advice"] = "水平一：(1,0)已出现，关注5日均线止盈"
        else:
            result["advice"] = "水平一：(1,0)出现，注意顶分型确认后的移动止盈"

    if state == "(1,0)" and len(bi_list) >= 4:
        up_bis = [b for b in bi_list if b["dir"] == "up"]
        if len(up_bis) >= 2:
            last_up = up_bis[-1]
            prev_up = up_bis[-2]
            last_pct = last_up.get("amplitude", 0)
            prev_pct = prev_up.get("amplitude", 0)
            result["details"]["last_up_amplitude"] = last_pct
            result["details"]["prev_up_amplitude"] = prev_pct
            result["level"] = 2
            if last_pct < prev_pct * 0.7:
                result["advice"] += " | 水平二：后段力度明显弱于前段，可利用震荡做短差"
            else:
                result["advice"] += " | 水平二：力度尚可，震荡中观望"

    if state == "(1,0)" and len(bi_list) >= 4 and "macd_hist" in df.columns:
        up_bis = [b for b in bi_list if b["dir"] == "up"]
        if len(up_bis) >= 2:
            last_up = up_bis[-1]
            prev_up = up_bis[-2]
            last_area = _macd_area_fast(df, last_up["start_idx"], last_up["end_idx"])
            prev_area = _macd_area_fast(df, prev_up["start_idx"], prev_up["end_idx"])
            result["details"]["last_macd_area"] = last_area
            result["details"]["prev_macd_area"] = prev_area
            result["level"] = 3
            if last_area < prev_area:
                result["advice"] += " | 水平三：MACD力度背驰确认，见顶概率高——用区间套定位精确卖点"
            else:
                result["advice"] += " | 水平三：MACD力度未背驰，可能只是中继分型"
    return result


def _macd_area_fast(df, start_idx, end_idx):
    if start_idx >= end_idx or start_idx < 0 or end_idx >= len(df):
        return 0.0
    return float(df.loc[start_idx:end_idx, "macd_hist"].abs().sum())


def multi_timeframe_dual_table(dual_map: Dict[str, dict]) -> dict:
    """多级别联立规则（第91-93课）"""
    daily = dual_map.get("daily", {})
    high_state = dual_map.get("weekly", dual_map.get("60min", {}))
    d_state = daily.get("state", "")
    h_state = high_state.get("state", "")
    result = {"daily_state": daily, "high_state": high_state}

    if h_state == "(1,1)":
        if d_state in ("(1,0)", "(-1,1)", "(-1,0)"):
            result["advice"] = "高级别向上笔延伸中，日线波动只是噪音——继续持有"
            result["action"] = "hold"
        else:
            result["advice"] = "多级别共振向上——最佳持股状态"
            result["action"] = "hold"
    elif h_state == "(-1,1)":
        result["advice"] = "高级别向下笔延伸中，日线反弹不参与——继续持币"
        result["action"] = "wait"
    elif h_state == "(1,0)":
        result["advice"] = "高级别顶分型构造中——用日线级别确认是否真正见顶"
        result["action"] = "prepare_sell"
    elif h_state == "(-1,0)":
        result["advice"] = "高级别底分型构造中——用日线级别确认是否真正见底"
        result["action"] = "prepare_buy"
    else:
        result["advice"] = "以日线状态为准"
        result["action"] = daily.get("action", "观望")
    return result


# ============================================================
# 第三部分：中枢震荡监视器（第92课）⭐ 5条完整变盘预警
# ============================================================

def zhongshu_monitor(df: pd.DataFrame, zhongshu_list: List[dict],
                     sub_elements: List[dict],
                     sub_zhongshu_list: List[dict] = None) -> dict:
    """中枢震荡监控（完整Zn体系 — 第92课5条变盘预警）"""
    if not zhongshu_list:
        return {"active": False}

    last_zs = zhongshu_list[-1]
    last_price = float(df["close"].iloc[-1])
    z = last_zs["zz"]
    zg = last_zs["zg"]
    zd = last_zs["zd"]

    if last_price > zg:
        position = "above_zs"
    elif last_price < zd:
        position = "below_zs"
    else:
        position = "inside_zs"

    strength = "偏强" if last_price > z else "偏弱"

    zn_values = []
    if sub_elements:
        for el in sub_elements[-6:]:
            el_z = (el["start_price"] + el["end_price"]) / 2
            zn_values.append(el_z)

    monitor = {
        "active": True, "zg": zg, "zd": zd, "zz": z,
        "position": position, "strength": strength,
        "last_price": last_price, "zs_range": f"[{zd:.2f}, {zg:.2f}]",
        "zn_values": zn_values[-6:],
        "alerts": [],
    }

    # 预警1：次级别是趋势类型 + Zn配合
    if len(sub_elements) >= 3 and sub_zhongshu_list:
        last_sub_zs = sub_zhongshu_list[-1]
        if last_sub_zs["zg"] < zd or last_sub_zs["zd"] > zg:
            if zn_values:
                recent_zn_trend = zn_values[-3:] if len(zn_values) >= 3 else zn_values
                zn_rising = all(recent_zn_trend[i] <= recent_zn_trend[i + 1]
                                for i in range(len(recent_zn_trend) - 1))
                zn_falling = all(recent_zn_trend[i] >= recent_zn_trend[i + 1]
                                 for i in range(len(recent_zn_trend) - 1))
                if zn_rising or zn_falling:
                    monitor["alerts"].append({
                        "type": "trend_zn_alignment",
                        "desc": "次级别趋势类型离开中枢 + Zn一致走向 → 变盘概率极高",
                        "severity": "danger",
                    })

    # 预警2：最后一个次级别中枢在ZS外
    if sub_zhongshu_list:
        last_sub_zs = sub_zhongshu_list[-1]
        if last_sub_zs["zg"] < zd:
            monitor["alerts"].append({
                "type": "sub_zs_below",
                "desc": "次级别中枢在ZS下方完成 → 向下变盘概率大",
                "severity": "warning",
            })
        elif last_sub_zs["zd"] > zg:
            monitor["alerts"].append({
                "type": "sub_zs_above",
                "desc": "次级别中枢在ZS上方完成 → 向上变盘概率大",
                "severity": "warning",
            })

    # 预警3：Zn缓慢提高但无力突破ZG → 上升楔型诱多
    if len(zn_values) >= 3:
        recent_zn = zn_values[-3:]
        if all(recent_zn[i] <= recent_zn[i + 1] for i in range(len(recent_zn) - 1)):
            if recent_zn[-1] < zg and recent_zn[-1] > z:
                monitor["alerts"].append({
                    "type": "rising_wedge",
                    "desc": "Zn缓慢提高但无力突破ZG——上升楔型诱多",
                    "severity": "warning",
                })
        if all(recent_zn[i] >= recent_zn[i + 1] for i in range(len(recent_zn) - 1)):
            if recent_zn[-1] > zd and recent_zn[-1] < z:
                monitor["alerts"].append({
                    "type": "falling_wedge",
                    "desc": "Zn缓慢降低但无力跌破ZD——下降楔型诱空",
                    "severity": "warning",
                })

    # 预警4：Zn反复穿越Z
    if len(zn_values) >= 4:
        crossings = sum(
            1 for i in range(1, len(zn_values))
            if (zn_values[i] - z) * (zn_values[i - 1] - z) < 0
        )
        if crossings >= 2:
            monitor["alerts"].append({
                "type": "zn_crossing_z",
                "desc": "Zn反复穿越中轴Z——方向不明，观望",
                "severity": "info",
            })

    # 预警5：布林收口
    boll = _boll_narrowing(df)
    if boll.get("narrowing"):
        monitor["alerts"].append({
            "type": "boll_narrowing",
            "desc": f"布林带宽收口至{boll.get('current_width', 0):.3f}（历史低位）——变盘时机临近",
            "severity": "info",
            "boll_detail": boll,
        })

    if position == "above_zs":
        monitor["advice"] = "中枢上方，等待三买确认或背驰卖出"
    elif position == "below_zs":
        monitor["advice"] = "中枢下方，关注盘整背驰买点或三卖风险"
    else:
        monitor["advice"] = f"中枢震荡中（{strength}），上抛下吸降低筹码成本"

    return monitor


# ============================================================
# 第四部分：中阴阶段（第88-90/99课）⭐ 布林增强
# ============================================================

def identify_bardo(df: pd.DataFrame, divergence: dict,
                   zhongshu_list: List[dict],
                   bi_list: List[dict]) -> dict:
    """识别中阴阶段（第88-90/99课，布林通道辅助）"""
    result = {
        "active": False, "stage": "none",
        "description": "", "action": "",
        "expected_zs_level": None,
        "boll_signals": [],
        "health": "unknown",
    }

    if not divergence.get("has_divergence"):
        return result
    if divergence.get("type") != "trend_divergence":
        return result
    if not zhongshu_list or not bi_list:
        return result

    last_zs = zhongshu_list[-1]
    last_price = bi_list[-1]["end_price"] if bi_list else 0

    # BOLL辅助判断
    if "boll_upper" in df.columns and "boll_mid" in df.columns:
        boll_upper = float(df["boll_upper"].iloc[-1])
        boll_mid = float(df["boll_mid"].iloc[-1])
        boll_lower = float(df["boll_lower"].iloc[-1])
        boll_pct_b = float(df["boll_pct_b"].iloc[-1]) if "boll_pct_b" in df.columns else None

        if len(df) >= 3:
            prev_pct_b = float(df["boll_pct_b"].iloc[-3]) if "boll_pct_b" in df.columns else None
            if prev_pct_b and prev_pct_b > 1.0 and boll_pct_b and boll_pct_b < 0.8:
                result["boll_signals"].append({
                    "type": "fell_from_upper",
                    "desc": "从上轨超强区域跌回——若再创新高而回不到超强区域，则为一卖",
                })

        if len(df) >= 5:
            upper_trend = df["boll_upper"].iloc[-5:].values
            if len(upper_trend) >= 3 and upper_trend[-1] < upper_trend[-3]:
                result["boll_signals"].append({
                    "type": "upper_turning",
                    "desc": "布林上轨开始转向——可能成为二卖的最大阻力位",
                })

        boll = _boll_narrowing(df)
        if boll.get("narrowing"):
            result["boll_signals"].append({
                "type": "boll_narrowing",
                "desc": "布林收口 + 中阴阶段 → 中阴结束的最佳时机，变盘在即",
                "severity": "important",
            })

        result["boll_status"] = {
            "upper": boll_upper, "mid": boll_mid, "lower": boll_lower,
            "pct_b": boll_pct_b,
            "price_vs_boll": "above_upper" if last_price > boll_upper else
                            "below_lower" if last_price < boll_lower else "in_band",
        }

    # 基本中阴阶段判断
    if last_zs["zd"] <= last_price <= last_zs["zg"]:
        result["active"] = True
        result["stage"] = "early"
        result["description"] = "走势因背驰死亡，当前处于中阴阶段——围绕最后一个中枢震荡"
        result["action"] = "按中枢震荡操作：上抛下吸，等待第三类买卖点确认方向"
        result["last_zs"] = last_zs
    elif last_price > last_zs["gg"]:
        result["stage"] = "bull_break"
        result["description"] = "价格突破GG，可能向上脱离中阴"
        result["action"] = "等待三买确认——如果次级别回抽不破ZG，中阴结束→上涨确立"
    elif last_price < last_zs["dd"]:
        result["stage"] = "bear_break"
        result["description"] = "价格跌破DD，可能向下脱离中阴"
        result["action"] = "等待三卖确认——如果次级别反弹不破ZD，中阴结束→下跌确立"

    # 健康vs危险中阴（第99课）
    if len(zhongshu_list) >= 2:
        prev_zs = zhongshu_list[-2]
        last_zg = last_zs.get("zg", 0)
        prev_zg = prev_zs.get("zg", 0)
        last_zd = last_zs.get("zd", 0)
        prev_zd = prev_zs.get("zd", 0)
        if last_zd > prev_zd and last_zg > prev_zg:
            result["health"] = "healthy"
            result["health_note"] = "中枢逐步抬高——健康中阴，后市大概率向上"
        elif last_zg < prev_zg and last_zd < prev_zd:
            result["health"] = "dangerous"
            result["health_note"] = "中枢逐步降低——危险中阴，后市大概率向下"
        else:
            result["health"] = "neutral"
            result["health_note"] = "中枢区间重叠——方向未决，观望为主"
    else:
        result["health"] = "neutral"

    result["expected_zs_level"] = "上一级别"
    result["note"] = "中阴阶段无一例外表现为不同级别的盘整"
    return result


# ============================================================
# 第五部分：区间套（第27/61/102课）⭐ 完整递归
# ============================================================

def interval_nesting_analysis(df_levels: Dict[str, pd.DataFrame],
                              primary_level: str = "daily") -> dict:
    """区间套精确定位（第102课完整递归算法）"""
    levels_order = ["weekly", "daily", "60min", "30min", "5min", "1min"]
    available = [k for k in levels_order if k in df_levels and len(df_levels[k]) > 0]

    result = {
        "active": False, "levels_checked": [],
        "precise_point": None, "resonance_depth": 0,
        "warning": None,
    }

    if len(available) < 2:
        result["warning"] = "需要至少两个周期的数据"
        return result

    primary_idx = available.index(primary_level) if primary_level in available else 0
    if primary_idx >= len(available) - 1:
        result["warning"] = "主级别已是最低级别，无更小级别可嵌套"
        return result

    nested_levels = available[primary_idx:]
    result["levels_checked"] = nested_levels

    resonance_depth = 0
    last_divergence_range = None

    for level_idx, level in enumerate(nested_levels):
        df = df_levels[level]
        if "macd_hist" not in df.columns or "diff" not in df.columns:
            break
        n = len(df)
        if n < 30:
            break

        if last_divergence_range is not None:
            # 从上一级别的背驰段映射到当前级别
            if level_idx > 0 and nested_levels[level_idx - 1] in df_levels:
                high_df = df_levels[nested_levels[level_idx - 1]]
                start_date = high_df["date"].iloc[last_divergence_range[0]]
                end_date = high_df["date"].iloc[min(last_divergence_range[1], len(high_df) - 1)]
                mask = (df["date"] >= start_date) & (df["date"] <= end_date)
                if mask.sum() < 10:
                    break
                check_start = int(df.index[mask][0])
                check_end = int(df.index[mask][-1])
            else:
                break
        else:
            check_start = max(0, n - n // 3)
            check_end = n - 1

        if check_end - check_start < 10:
            break

        mid = (check_start + check_end) // 2
        if mid - check_start >= 5 and check_end - mid >= 5:
            front_area = float(df.loc[check_start:mid, "macd_hist"].abs().sum())
            back_area = float(df.loc[mid:check_end, "macd_hist"].abs().sum())

            front_diff_range = float(df.loc[check_start:mid, "diff"].max() -
                                     df.loc[check_start:mid, "diff"].min())
            back_diff_range = float(df.loc[mid:check_end, "diff"].max() -
                                    df.loc[mid:check_end, "diff"].min())

            has_divergence = (
                (back_area < front_area * 0.8) or
                (back_diff_range < front_diff_range * 0.8)
            )

            if has_divergence:
                resonance_depth += 1
                last_divergence_range = (mid, check_end)

                macd_abs = df.loc[mid:check_end, "macd_hist"].abs()
                if len(macd_abs) > 0 and macd_abs.max() > 0:
                    peak_idx = mid + int(macd_abs.idxmax() - macd_abs.index[0])
                    precise_date = df["date"].iloc[min(peak_idx, n - 1)]
                    result["precise_point"] = {
                        "level": level,
                        "date": precise_date.strftime("%Y-%m-%d"),
                        "price": float(df["close"].iloc[min(peak_idx, n - 1)]),
                        "idx": min(peak_idx, n - 1),
                    }
            else:
                if resonance_depth > 0:
                    result["warning"] = (
                        f"区间套在{level}级别断裂——可能是小转大，"
                        f"区间套失效，须用二买/二卖确认（第53课）"
                    )
                break

    result["active"] = resonance_depth >= 2
    result["resonance_depth"] = resonance_depth

    if result["active"]:
        result["confidence"] = (
            "very_high" if resonance_depth >= 3 else
            "high" if resonance_depth >= 2 else "medium"
        )
        result["note"] = (
            f"区间套共振深度{resonance_depth}层——{result['confidence']}置信度。"
            f"所有级别走势必完美（第102课）。"
        )
    elif resonance_depth == 0:
        result["note"] = "主级别未发现背驰段——无需区间套定位"

    return result


# ============================================================
# 第六部分：同级别分解（第38-40课）
# ============================================================

def same_level_decomposition(bi_list: List[dict],
                             operate_level: str = "daily") -> dict:
    """同级别分解——机械化操作程式"""
    result = {
        "segments": [], "current_phase": None,
        "next_action": "", "auto_shift": False,
    }

    if len(bi_list) < 4:
        result["next_action"] = "数据不足，等待更多笔形成"
        return result

    phases = []
    i = 0
    while i < len(bi_list):
        phase_bi = [bi_list[i]]
        j = i + 1
        while j < len(bi_list):
            phase_bi.append(bi_list[j])
            if len(phase_bi) >= 3:
                last_three = phase_bi[-3:]
                if last_three[0]["dir"] != last_three[-1]["dir"]:
                    phases.append({
                        "start_bi": i, "end_bi": j - 1,
                        "dir": phase_bi[0]["dir"],
                        "bi_count": len(phase_bi) - 1,
                        "start_price": phase_bi[0]["start_price"],
                        "end_price": phase_bi[-2]["end_price"],
                    })
                    i = j
                    break
            j += 1
        else:
            break

    result["segments"] = phases

    if phases:
        last_phase = phases[-1]
        if last_phase["dir"] == "up":
            result["current_phase"] = "向上段"
            if len(phases) >= 2:
                prev_up = [p for p in phases[:-1] if p["dir"] == "up"]
                if prev_up:
                    prev_high = prev_up[-1]["end_price"]
                    if last_phase["end_price"] < prev_high:
                        result["next_action"] = "不创新高 → 卖出"
                    elif last_phase["end_price"] > prev_high:
                        result["next_action"] = "创新高 → 持有，等背驰信号"
            else:
                result["next_action"] = "第一段上涨 → 持有，等背驰卖出"
        else:
            result["current_phase"] = "向下段"
            if len(phases) >= 2:
                prev_down = [p for p in phases[:-1] if p["dir"] == "down"]
                if prev_down:
                    prev_low = prev_down[-1]["end_price"]
                    if last_phase["end_price"] > prev_low:
                        result["next_action"] = "不破前低 → 准备买入"
                    else:
                        result["next_action"] = "破前低 → 看是否形成盘整背驰，是则买入"

    return result


# ============================================================
# 第七部分：小背驰-大转折定理（第44课）
# ============================================================

def check_small_large_divergence(low_level_result: dict,
                                 high_level_result: dict) -> dict:
    """小背驰-大转折定理"""
    low_div = low_level_result.get("divergence", {})
    low_tp = low_level_result.get("trade_points", {})

    result = {"triggered": False, "direction": None, "condition_met": False}

    if not low_div.get("has_divergence"):
        return result

    direction = low_div.get("details", {}).get("direction", "")
    sell_points = low_tp.get("sell_points", [])
    buy_points = low_tp.get("buy_points", [])

    result["direction"] = direction

    if direction == "top_divergence":
        has_third_sell = any(sp["type"] == "三卖" for sp in sell_points)
        result["condition_met"] = has_third_sell
        if has_third_sell:
            result["triggered"] = True
            result["warning"] = "小级别顶背驰 + 三卖 → 可能引发大级别向下转折！"
    elif direction == "bottom_divergence":
        has_third_buy = any(bp["type"] == "三买" for bp in buy_points)
        result["condition_met"] = has_third_buy
        if has_third_buy:
            result["triggered"] = True
            result["warning"] = "小级别底背驰 + 三买 → 可能引发大级别向上转折！"

    if not result["condition_met"]:
        result["note"] = "必要条件未满足（无第三类买卖点），大转折概率低"

    return result


# ============================================================
# 第八部分：板块强弱指标（第106课）
# ============================================================

FIBONACCI_MA = [5, 13, 21, 34, 55, 89, 144, 233]


def compute_sector_strength_class(df: pd.DataFrame) -> dict:
    """菲波那契均线分类：价格在N条均线之上 → N类（0-8共9档）"""
    if "close" not in df.columns or len(df) < 233:
        return {"class": 0, "total_ma": len(FIBONACCI_MA), "detail": {}}

    last_price = float(df["close"].iloc[-1])
    ma_values = {}
    above_count = 0

    for period in FIBONACCI_MA:
        if len(df) >= period:
            ma_val = float(df["close"].rolling(period).mean().iloc[-1])
            ma_values[f"MA{period}"] = ma_val
            if last_price > ma_val:
                above_count += 1

    return {
        "class": above_count,
        "total_ma": len(FIBONACCI_MA),
        "detail": ma_values,
        "description": f"{above_count}类（攻克{above_count}条斐波那契均线）",
    }


def compute_sector_avg_strength(stocks_strength: List[dict]) -> float:
    """计算板块平均强弱"""
    if not stocks_strength:
        return 0.0
    return sum(s.get("class", 0) for s in stocks_strength) / len(stocks_strength)


# ============================================================
# 第九部分：中枢相对位置完全分类（第49课——操作总纲）
# ============================================================

def position_complete_classification(df: pd.DataFrame,
                                     zhongshu_list: List[dict],
                                     bi_list: List[dict]) -> dict:
    """中枢相对位置的完全分类（第49课）"""
    if not zhongshu_list:
        return {"category": "no_zhongshu", "sub_category": "无中枢", "advice": "无中枢，无法分类"}

    last_zs = zhongshu_list[-1]
    last_price = float(df["close"].iloc[-1])
    zg, zd = last_zs["zg"], last_zs["zd"]

    result = {"category": None, "sub_category": None, "advice": "", "zs": last_zs}

    if zd <= last_price <= zg:
        result["category"] = "inside_zs"
        result["sub_category"] = "中枢之中"
        result["advice"] = "不操作是最好的操作，等待演化。技术好可参与次级别二买"
    elif last_price < zd:
        result["category"] = "below_zs"
        result["sub_category"] = "中枢之下"
        result["advice"] = (
            "子类1（无三卖）：中枢震荡依旧 → 用区间套找向下离开的背驰买点\n"
            "子类2（有三卖）：中枢已结束 → 等三卖后次级别完成+背驰，或不参与等新中枢"
        )
    else:
        result["category"] = "above_zs"
        result["sub_category"] = "中枢之上"
        result["advice"] = (
            "子类1（无三买）：无合适买点，等待\n"
            "子类2（有三买）：刚形成时介入最佳；若已出现盘整顶背驰则等待"
        )

    return result


# ============================================================
# 第十部分：多义性分解（第54课）
# ============================================================

def multi_meaning_decomposition(sub_elements: List[dict]) -> dict:
    """
    走势多义性分解（第54课）

    核心原则：
    1. 两个同级别中枢之间必须有次级别走势连接
    2. 选择当下有明确操作意义的分解
    3. 多种分解可以相互印证

    结合律：用不同的方式组合三段的括弧位置
    例：g0g5 = g0d1+(d1g1+g1d2+d2g2)+g2d3+d3g3+g3d4
          = g0d1+d1g1+g1d2+(d2g2+g2d3+d3g3)+g3d4
    """
    n = len(sub_elements)
    if n < 5:
        return {"decompositions": [], "recommended": None}

    decompositions = []

    for start in range(0, n - 2):
        for mid_gap in range(1, min(4, n - start - 4)):
            zs1_end = start + 2
            zs2_start = zs1_end + mid_gap
            if zs2_start + 2 >= n:
                continue

            connector = sub_elements[zs1_end:zs2_start]
            if len(connector) < 1:
                continue

            seg1 = sub_elements[start:start + 3]
            seg2 = sub_elements[zs2_start:zs2_start + 3]

            def _calc_overlap(segs):
                lo = max(min(s["start_price"], s["end_price"]) for s in segs)
                hi = min(max(s["start_price"], s["end_price"]) for s in segs)
                return (lo, hi) if lo < hi else None

            ol1 = _calc_overlap(seg1)
            ol2 = _calc_overlap(seg2)
            if not ol1 or not ol2:
                continue

            last_price = sub_elements[-1]["end_price"] if sub_elements else 0
            zs2_lo, zs2_hi = ol2

            if last_price > zs2_hi:
                meaning = "第三类买点候选——中枢上方等待回调确认"
            elif last_price < zs2_lo:
                meaning = "第三类卖点候选——中枢下方等待反弹确认"
            else:
                meaning = "中枢震荡——上抛下吸"

            decompositions.append({
                "zs1_range": ol1, "zs2_range": ol2,
                "zs1_segments": (start, start + 2),
                "zs2_segments": (zs2_start, zs2_start + 2),
                "connector_segments": (zs1_end, zs2_start),
                "meaning": meaning,
                "has_third_point_signal": "第三类" in meaning,
            })

    recommended = None
    for d in decompositions:
        if d.get("has_third_point_signal"):
            recommended = d
            break
    if not recommended and decompositions:
        recommended = decompositions[0]

    return {
        "decompositions": decompositions,
        "count": len(decompositions),
        "recommended": recommended,
        "principle": "选择有明确操作意义的分解——第54课",
        "note": "多种分解可相互印证",
    }


# ============================================================
# 第十一部分：三买五步机械化操作程式（第79课）
# ============================================================

def third_buy_mechanized_operation(trade_points: dict,
                                   zhongshu_list: List[dict],
                                   bi_list: List[dict]) -> dict:
    """
    三买机械化操作五步法（第79课）

    步骤：
    1. 选定一个足够反应的级别（30分钟/5分钟/日线）
    2. 只介入在该级别出现第三类买点的股票
    3. 买入后，一旦新的次级别向上不能新高或出现盘整背驰 → 坚决卖掉
    4. 如果没出现3 → 持有到该上移走势背驰后至少卖掉一半
    5. 尽量只介入第一个中枢的第三类买点（第二中枢以后形成大级别中枢概率剧增）
    """
    result = {
        "operable": False, "step": 0, "action": "", "risk_level": "unknown",
    }

    buy_points = trade_points.get("buy_points", [])
    sell_points = trade_points.get("sell_points", [])

    third_buys = [bp for bp in buy_points if bp["type"] == "三买"]
    if not third_buys:
        result["action"] = "无三买信号，不介入。等待符合条件的股票"
        return result

    tb = third_buys[0]
    result["operable"] = True
    result["step"] = 2

    if zhongshu_list:
        zs_count = len(zhongshu_list)
        if zs_count == 1:
            result["zs_position"] = "第一个中枢"
            result["preferred"] = True
            result["action"] = "第一个中枢的三买——最优介入点，成功率最高"
            result["risk_level"] = "low"
        elif zs_count == 2:
            result["zs_position"] = "第二个中枢"
            result["preferred"] = False
            result["action"] = "第二个中枢的三买——注意大级别中枢形成风险，控制仓位"
            result["risk_level"] = "medium"
        else:
            result["zs_position"] = f"第{zs_count}个中枢"
            result["preferred"] = False
            result["action"] = "第三个或以后中枢的三买——大级别中枢形成概率剧增，不建议重仓"
            result["risk_level"] = "high"

    if tb.get("is_first_time") is False:
        result["step"] = 3
        result["exit_signal"] = "非首次突破——力度减弱，应考虑卖出"

    if result.get("preferred") and tb.get("is_first_time", True):
        result["step"] = 4
        result["hold_condition"] = (
            "持有到该级别中枢上移出现背驰后至少卖掉一半；"
            "然后次级别回试，不创新高或盘整背驰则全部出掉"
        )

    third_sells = [sp for sp in sell_points if sp["type"] == "三卖"]
    if third_sells:
        ts = third_sells[0]
        result["sell_operation"] = {
            "signal": "三卖出现",
            "action": "必须离场——次级别跌破中枢后反弹不破ZD",
            "urgency": "high",
            "is_first": ts.get("is_first_time", True),
        }

    return result


# ============================================================
# 第十二部分：8级递归记号体系（第88课）
# ============================================================

LEVEL_NOTATION = {
    "1min": "Y",
    "5min": "W",
    "30min": "S",
    "60min": "S",
    "daily": "R",
    "weekly": "Z",
    "monthly": "M",
    "quarterly": "J",
    "yearly": "N",
}

LEVEL_HIERARCHY = ["1min", "5min", "30min", "daily", "weekly", "monthly", "quarterly", "yearly"]


def get_level_notation(level_name: str, idx: int) -> str:
    """获取某级别某点位的记号，如 R191 (日线第191个标记点)"""
    prefix = LEVEL_NOTATION.get(level_name, "?")
    return f"{prefix}{idx}"


def analyze_multi_level_landmarks(results: Dict[str, dict]) -> dict:
    """
    多级别标记点分析（第88课）

    最牛的点：从线段一直到年，同时都有标号的那个点
    - 如果是顶 → 百年大顶
    - 如果是底 → 百年大底
    """
    if not results:
        return {"landmarks": [], "top_points": []}

    level_points = {}
    for level_key, r in results.items():
        notation = LEVEL_NOTATION.get(level_key, "?")
        zs_list = r.get("zhongshu_list", [])

        points = []
        for i, zs in enumerate(zs_list):
            points.append({
                "notation": f"{notation}{i}",
                "date": zs.get("end_date", ""),
                "price_range": (zs.get("zd", 0), zs.get("zg", 0)),
                "fate": zs.get("fate", ""),
            })
        level_points[level_key] = points

    notation_chain = []
    for level in LEVEL_HIERARCHY:
        if level in level_points and level_points[level]:
            notation_chain.append({
                "level": level,
                "notation": LEVEL_NOTATION.get(level, "?"),
                "latest": level_points[level][-1],
            })

    result = {
        "notation_chain": notation_chain,
        "chain_depth": len(notation_chain),
        "top_points": [],
    }

    if len(notation_chain) >= 4:
        result["significance"] = "极高——跨4级以上共振，可能为大级别转折"
    elif len(notation_chain) >= 3:
        result["significance"] = "高——跨3级共振，趋势转折可信度强"
    elif len(notation_chain) >= 2:
        result["significance"] = "中——跨2级共振，有一定操作意义"
    else:
        result["significance"] = "低——仅单级别，等待多级别验证"

    result["principle"] = (
        "走势必完美的递归记号体系（第88课）："
        "走势可以唯一地表示为 a1A1+a5A5+a30A30 的形式，"
        "任何高级别的改变必须先从低级别开始（第102课）"
    )

    return result


# ============================================================
# 第十三部分：资金管理（第31课）
# ============================================================

def capital_management(current_capital: float,
                       position_cost: float = 0,
                       current_price: float = 0,
                       shares: int = 0,
                       signal_confidence: float = 0.5,
                       max_position_pct: float = 0.3) -> dict:
    """
    资金管理模型（第31课）

    核心原则：
    1. 资金必须长期无压力（不借钱、不透支）
    2. 成本为0以前，要把成本变为0
    3. 成本为0以后，要挣股票（短差增加股数）
    4. 绝不往上加码，只往下买
    5. 每只股票留1/10机动资金做短差
    6. 短差绝不增加股票数量（只在成本归零后增加）
    """
    result = {
        "capital": current_capital,
        "position_value": position_cost * shares if shares > 0 else 0,
        "position_pct": 0.0,
        "mobile_capital": 0.0,
        "risk_level": "unknown",
        "phase": "unknown",
        "actions": [],
    }

    if shares > 0 and current_capital > 0:
        result["position_pct"] = (position_cost * shares) / current_capital
        result["mobile_capital"] = current_capital * 0.1

    max_position = current_capital * max_position_pct
    if result["position_value"] > max_position:
        result["risk_level"] = "danger"
        result["actions"].append({
            "action": "减仓",
            "reason": f"单只股票仓位{result['position_pct']:.1%}超上限{max_position_pct:.0%}",
            "target_value": max_position,
        })
    else:
        result["risk_level"] = "safe"

    if position_cost <= 0 or shares == 0:
        result["phase"] = "initial"
        suggested_position = current_capital * min(max_position_pct, signal_confidence)
        result["actions"].append({
            "action": "建仓",
            "reason": "无持仓，按信号置信度一次性建仓",
            "suggested_amount": suggested_position,
            "principle": "投入前充分准备，一次性买入，绝不往上加码",
        })
        result["mobile_capital"] = suggested_position * 0.1
        result["actions"].append({
            "action": "预留机动资金",
            "amount": result["mobile_capital"],
            "usage": "用于小级别短差降成本，每次短差不增加股票数量",
        })
    elif position_cost > 0:
        profit_pct = (current_price - position_cost) / position_cost if position_cost > 0 else 0
        if profit_pct < 1.0:
            result["phase"] = "cost_reduction"
            result["actions"].append({
                "action": "降成本",
                "reason": f"当前浮盈{profit_pct:.1%}，目标翻倍后出部分归零成本",
                "target": "100%涨幅附近找大级别卖点出掉部分",
                "mobile_usage": "用小级别短差降低成本，每次短差不增加股数",
            })
        else:
            result["phase"] = "share_accumulation"
            result["actions"].append({
                "action": "挣股票",
                "reason": "成本已可归零，每次短差抛了全部回补",
                "principle": "成本为0的股票越多越好，等待超大级别卖点一次性砸",
            })

    return result


def cost_zero_tracker(position_cost: float, current_price: float,
                      shares: int, profit_target_pct: float = 1.0) -> dict:
    """成本归零跟踪器（第31课）"""
    if position_cost <= 0 or shares <= 0:
        return {"cost_zero_possible": False, "reason": "无有效持仓"}

    profit_pct = (current_price - position_cost) / position_cost
    total_cost = position_cost * shares
    current_value = current_price * shares

    if current_price > 0:
        shares_to_sell = total_cost / current_price
    else:
        shares_to_sell = shares

    return {
        "cost_zero_possible": profit_pct >= profit_target_pct,
        "profit_pct": profit_pct,
        "total_cost": total_cost,
        "current_value": current_value,
        "shares_to_sell_for_cost_zero": int(shares_to_sell) + 1,
        "remaining_shares": shares - int(shares_to_sell) - 1,
        "phase": "share_accumulation" if profit_pct >= profit_target_pct else "cost_reduction",
        "next_milestone": (
            "找大级别卖点出掉部分收成本"
            if profit_pct >= profit_target_pct * 0.8
            else f"等待涨幅达到{profit_target_pct:.0%}"
        ),
    }


# ============================================================
# 第十四部分：每日走势分类（第46课）
# ============================================================

def daily_trend_classification(df_30min: pd.DataFrame) -> dict:
    """
    每日走势分类（第46课）

    一天8根30分钟K线，3根重叠=一个中枢。
    三类：一个中枢（平衡市）、两个中枢、无中枢（最强单边）。
    """
    if len(df_30min) < 8:
        return {"class": "insufficient_data", "reason": f"仅{len(df_30min)}根30分K线"}

    day_bars = df_30min.iloc[-8:].copy().reset_index(drop=True)

    pivots = []
    for i in range(6):
        bars = [day_bars.iloc[i + j] for j in range(3)]
        zg = min(b["high"] for b in bars)
        zd = max(b["low"] for b in bars)
        if zg > zd:
            pivots.append({"start_bar": i, "end_bar": i + 2, "zg": zg, "zd": zd})

    first3_high = max(day_bars.iloc[j]["high"] for j in range(3))
    first3_low = min(day_bars.iloc[j]["low"] for j in range(3))
    day_high = day_bars["high"].max()
    day_low = day_bars["low"].min()
    day_close = day_bars["close"].iloc[-1]

    result = {"pivot_count": len(pivots), "pivots": pivots}

    if len(pivots) == 0:
        result["class"] = "no_pivot"
        result["description"] = "最强单边——8根K线无相邻3根重叠"
        result["significance"] = "日K线具重要方向意义；大中枢中出现可能是骗线"
        result["strength"] = "strongest"

    elif len(pivots) == 1:
        result["class"] = "balance"
        p = pivots[0]
        if first3_high >= day_high:
            result["subtype"] = "weak_balance"
            result["description"] = "弱平衡市——前三根现全天高点"
            if day_close >= p["zg"]:
                result["strength"] = "average"
            elif day_close >= p["zd"]:
                result["strength"] = "weak"
            else:
                result["strength"] = "weakest"
        elif first3_low <= day_low:
            result["subtype"] = "strong_balance"
            result["description"] = "强平衡市——前三根现全天低点"
            if day_close >= p["zg"]:
                result["strength"] = "strongest"
            elif day_close >= p["zd"]:
                result["strength"] = "average"
            else:
                result["strength"] = "weak"
        else:
            result["subtype"] = "turning_balance"
            result["description"] = "转折平衡市——前三根未现全天高低点"

    elif len(pivots) >= 2:
        result["class"] = "trend_with_pivots"
        p1, p2 = pivots[0], pivots[-1]
        result["direction"] = "up" if p2["zg"] > p1["zg"] else "down"

        unilateral = [i for i in range(8)
                      if not any(p["start_bar"] <= i <= p["end_bar"] for p in pivots)]
        result["unilateral_bars"] = unilateral
        if 3 in unilateral or 4 in unilateral:
            result["unilateral_position"] = "第4/5根K线——变盘在午盘前后30分钟"

        if result["direction"] == "up":
            result["strength"] = "strongest" if day_close >= p2["zg"] else (
                "strong" if day_close >= p1["zg"] else "weak")
        else:
            result["strength"] = "strongest" if day_close <= p2["zd"] else (
                "strong" if day_close <= p1["zd"] else "weak")

    return result


# ============================================================
# 第十五部分：一夜情行情分析（第47课）
# ============================================================

def overnight_rebound_analysis(df_30min: pd.DataFrame,
                               df_5min: pd.DataFrame = None) -> dict:
    """
    一夜情反弹/回调分析（第47课）

    用第46课每日分类+中枢震荡力度比较+分笔背驰精确定位。
    中枢震荡有对称性，分笔背驰足以引发盘中大幅回拉。
    """
    result = {"analysis_time": "", "daily_class": None, "signals": []}

    if len(df_30min) < 8:
        result["error"] = "30分钟K线不足"
        return result

    daily = daily_trend_classification(df_30min)
    result["daily_class"] = daily["class"]

    if daily["pivot_count"] >= 1:
        day_pivot = daily["pivots"][0]
        result["day_pivot"] = {
            "zg": day_pivot["zg"], "zd": day_pivot["zd"],
            "zz": (day_pivot["zg"] + day_pivot["zd"]) / 2,
        }

        pre_bars = df_30min.iloc[:day_pivot["start_bar"]]
        post_bars = df_30min.iloc[day_pivot["end_bar"] + 1:]

        if len(pre_bars) >= 2 and len(post_bars) >= 2:
            pre_drop = float(pre_bars["high"].iloc[0] - pre_bars["low"].min())
            post_drop = float(post_bars["high"].iloc[0] - post_bars["low"].min())

            result["force_comparison"] = {
                "pre_pivot_drop": pre_drop,
                "post_pivot_drop": post_drop,
                "weakening": post_drop < pre_drop,
            }

            if post_drop < pre_drop:
                result["signals"].append({
                    "signal": "力度衰减——后段下跌<前段",
                    "implication": "将有强力回拉，目标为下跌最后一个反弹处",
                    "action": "准备回补",
                    "confidence": "high",
                })

        last_close = float(df_30min["close"].iloc[-1])
        if last_close < day_pivot["zd"]:
            result["signals"].append({
                "signal": "跌破当日中枢",
                "implication": "先有小三卖→两波下跌→随时完美",
                "action": "等待两波下跌后分笔背驰回补",
                "target": "下跌最后一个反弹处",
            })

    if daily["class"] == "no_pivot":
        result["signals"].append({
            "signal": "无中枢单边走势",
            "implication": "当日趋势极强，不可逆势",
            "action": "顺势持有，不抄底不摸顶",
            "confidence": "very_high",
        })

    if df_5min is not None and len(df_5min) >= 26:
        df5 = compute_macd(df_5min)
        recent_macd = df5["macd"].iloc[-20:]
        if len(recent_macd) >= 10:
            first = recent_macd.iloc[:10].abs().sum()
            second = recent_macd.iloc[10:].abs().sum()
            if second < first * 0.7:
                result["signals"].append({
                    "signal": "5分钟MACD分笔背驰",
                    "implication": "可能引发盘中大幅回拉",
                    "action": "关注反弹力度",
                })

    result["principle"] = (
        "一夜情分析范本（第47课）：用当日走势分类确定格局，"
        "用中枢震荡比较力度，用分笔背驰定位买卖点，全程当下分析。"
    )
    return result


# ============================================================
# 第十六部分：几何能量统一（第104课）
# ============================================================

def geometry_energy_correlation(zhongshu_list: List[dict],
                                divergence: dict,
                                bi_list: List[dict]) -> dict:
    """
    几何结构与能量动力统一（第104课）

    几何得分：中枢结构质量（宽度/延伸段数/数量）
    能量得分：MACD背驰质量（多重共振/置信度/0轴位置）
    统一指数：两者加权综合，共振时信号最可靠
    """
    result = {
        "geometry_score": 0.0, "energy_score": 0.0,
        "unity_index": 0.0, "alignment": "unknown", "details": {},
    }

    # 几何得分
    geo_score = 50.0
    if zhongshu_list:
        for zs in zhongshu_list:
            zg, zd = zs.get("zg", 0), zs.get("zd", 0)
            if zg > 0 and zd > 0:
                width_pct = (zg - zd) / ((zg + zd) / 2) * 100
                if width_pct < 3:
                    geo_score += 15
                elif width_pct < 8:
                    geo_score += 5
                else:
                    geo_score -= 5
        for zs in zhongshu_list:
            ec = zs.get("element_count", 0)
            if ec >= 9:
                geo_score += 20
            elif ec >= 5:
                geo_score += 8
        if len(zhongshu_list) >= 2:
            geo_score += 10

    if bi_list and len(bi_list) >= 3:
        amps = [abs(b.get("amplitude", 0)) for b in bi_list[-5:]]
        avg_amp = sum(amps) / len(amps) if amps else 0
        if avg_amp > 0.05:
            geo_score += 10
        result["details"]["avg_bi_amplitude"] = avg_amp

    geo_score = max(0, min(100, geo_score))
    result["geometry_score"] = geo_score

    # 能量得分
    energy_score = 50.0
    if divergence:
        div_list = divergence if isinstance(divergence, list) else divergence.get("divergences", [])
        div_count = len(div_list)
        if div_count >= 2:
            energy_score += 15
        elif div_count == 1:
            energy_score += 5
        conf = divergence.get("confidence", "") if isinstance(divergence, dict) else ""
        if conf in ("high", "very_high"):
            energy_score += 15
        if isinstance(divergence, dict) and divergence.get("near_zero_axis"):
            energy_score += 10

    energy_score = max(0, min(100, energy_score))
    result["energy_score"] = energy_score

    # 统一指数
    result["unity_index"] = geo_score * 0.5 + energy_score * 0.5
    u = result["unity_index"]
    if u >= 70:
        result["alignment"] = "几何与能量高度统一——信号极可靠"
    elif u >= 50:
        result["alignment"] = "几何与能量基本一致——信号可用"
    elif u >= 30:
        result["alignment"] = "几何与能量背离——需等待确认"
    else:
        result["alignment"] = "几何与能量严重分歧——远离观望"

    result["principle"] = "几何与能量的统一（第104课）：能量动力形态对应特殊几何结构，共振时转折最可靠。"
    return result


# ============================================================
# 第十七部分：机械化操作信号链（第105课）
# ============================================================

def mechanized_signal_chain(df: pd.DataFrame,
                            bi_list: List[dict] = None) -> dict:
    """
    机械化操作信号链（第105课）

    "用一个最简单的分型以及能否延伸为笔的最基本标准进行分类"
    信号链：分型出现→确认→笔延伸→等待反向分型→循环
    """
    if bi_list is None:
        bi_list = df.attrs.get("bi_list", [])

    result = {
        "current_signal": "unknown", "action": "观望",
        "position": "空仓", "signal_chain": [], "fractal_status": {},
    }

    if "fractal" not in df.columns:
        result["current_signal"] = "no_fractal_data"
        result["action"] = "请先运行形态学分析"
        return result

    recent_fx = df[df["fractal"] != ""].tail(3)
    if recent_fx.empty:
        result["current_signal"] = "no_recent_fractal"
        result["action"] = "等待分型出现"
        return result

    last_fx = recent_fx.iloc[-1]
    fx_type = str(last_fx["fractal"])

    if bi_list is not None and len(bi_list) > 0:
        last_bi = bi_list.iloc[-1] if hasattr(bi_list, "iloc") else bi_list[-1]
        bi_dir = last_bi.get("dir", "") if hasattr(last_bi, "get") else last_bi["dir"]

        if bi_dir == "down":
            result["current_signal"] = "down_stroke_extending"
            result["action"] = "持币等待"
            result["position"] = "空仓"
            result["signal_chain"].append({
                "stage": "向下笔延伸中",
                "status": f"自{last_bi.get('start_date','?')}向下笔运行中",
                "next_trigger": "等待底分型出现并确认",
            })
            if fx_type == "bottom":
                result["signal_chain"].append({
                    "stage": "底分型出现",
                    "status": "确认条件：第三根K线升破第一根高点",
                    "action_if_confirmed": "向上一笔开始→买入",
                    "action_if_failed": "向下笔继续→持币",
                })
                result["fractal_status"] = {
                    "type": "bottom",
                    "confirmed": _fx_confirmed(df, int(last_fx.name), "bottom"),
                    "idx": int(last_fx.name),
                }

        elif bi_dir == "up":
            result["current_signal"] = "up_stroke_extending"
            result["action"] = "持股待涨"
            result["position"] = "持仓"
            result["signal_chain"].append({
                "stage": "向上一笔延伸中",
                "status": f"自{last_bi.get('start_date','?')}向上笔运行中",
                "next_trigger": "等待顶分型出现并确认",
            })
            if fx_type == "top":
                result["signal_chain"].append({
                    "stage": "顶分型出现",
                    "status": "确认条件：第三根K线跌破第一根低点",
                    "action_if_confirmed": "向下一笔开始→卖出",
                    "action_if_failed": "向上笔继续→持股",
                })
                result["fractal_status"] = {
                    "type": "top",
                    "confirmed": _fx_confirmed(df, int(last_fx.name), "top"),
                    "idx": int(last_fx.name),
                }
    else:
        if fx_type == "bottom":
            result["current_signal"] = "bottom_fractal_forming"
            result["action"] = "关注——底分型可能形成买点"
        elif fx_type == "top":
            result["current_signal"] = "top_fractal_forming"
            result["action"] = "关注——顶分型可能形成卖点"

    result["principle"] = (
        "机械化操作（第105课）：用最简单的分型及能否延伸为笔进行分类，"
        "市场不过是一堆关节，机械化就是合于其节奏。"
    )
    return result


def _fx_confirmed(df: pd.DataFrame, fx_idx: int, fx_type: str) -> bool:
    """判断分型是否已被确认"""
    if fx_idx >= len(df) - 1 or fx_idx < 1:
        return False
    bar0 = df.iloc[fx_idx]
    bar1 = df.iloc[fx_idx + 1]
    if fx_type == "top":
        return float(bar1["low"]) < float(bar0["low"])
    elif fx_type == "bottom":
        return float(bar1["high"]) > float(bar0["high"])
    return False


# ============================================================
# 第十八部分：底部构造跟踪器（第108课）
# ============================================================

def bottom_construction_tracker(df: pd.DataFrame,
                                zhongshu_list: List[dict],
                                trade_points: dict,
                                bi_list: List[dict] = None) -> dict:
    """
    底部构造完整跟踪（第108课）

    走势类型底部：一买后到该中枢第一次出三买卖点前=底部构造过程
      三卖先出现→失败；三买先出现→完成
    分型底部（粗糙版）：
      跌破分型最低点→失败；站住分型上边沿→成功，至少展开向上一笔
    """
    result = {
        "bottom_in_construction": False, "bottom_type": None,
        "pivot_bottom": {}, "fractal_bottom": {},
        "construction_phase": "unknown", "action": "观望",
    }

    # 走势类型底部
    tp_list = trade_points if isinstance(trade_points, list) else trade_points.get("trade_points", [])
    first_buy_points = [tp for tp in tp_list if tp.get("type") == "first_buy"]

    if first_buy_points:
        last_1b = first_buy_points[-1]
        result["pivot_bottom"] = {
            "first_buy_date": last_1b.get("date", ""),
            "first_buy_price": last_1b.get("price", 0),
            "triggered": True,
        }

        relevant_zs = None
        for zs in zhongshu_list:
            if zs.get("start_date", "") >= last_1b.get("date", ""):
                relevant_zs = zs
                break
        if not relevant_zs and zhongshu_list:
            relevant_zs = zhongshu_list[-1]

        if relevant_zs:
            zg, zd = relevant_zs.get("zg", 0), relevant_zs.get("zd", 0)
            result["pivot_bottom"]["construction_zs"] = {
                "zg": zg, "zd": zd, "zz": (zg + zd) / 2,
                "fate": relevant_zs.get("fate", "unknown"),
            }

            third_buy = [tp for tp in tp_list if tp.get("type") == "third_buy"]
            third_sell = [tp for tp in tp_list if tp.get("type") == "third_sell"]
            recent_3b = third_buy[-1] if third_buy else None
            recent_3s = third_sell[-1] if third_sell else None

            if recent_3b:
                result["pivot_bottom"]["status"] = "complete"
                result["pivot_bottom"]["completion_signal"] = "三买确认"
                result["construction_phase"] = "底部完成——新行情展开"
                result["action"] = "买入/加仓"
            elif recent_3s:
                result["pivot_bottom"]["status"] = "failed"
                result["pivot_bottom"]["failure_signal"] = "三卖先出现"
                result["construction_phase"] = "底部构造失败"
                result["action"] = "止损/观望"
            else:
                result["pivot_bottom"]["status"] = "constructing"
                result["construction_phase"] = "底部构造中——等待三买或三卖"
                result["action"] = "中枢震荡操作：下探失败买，上冲无力卖"
                result["bottom_in_construction"] = True
    else:
        result["pivot_bottom"]["triggered"] = False

    # 分型底部
    if "fractal" in df.columns:
        bottom_fx = df[df["fractal"] == "bottom"].tail(3)
        if not bottom_fx.empty:
            bf = bottom_fx.iloc[-1]
            fx_high = float(bf["high"])
            fx_low = float(bf["low"])
            last_close = float(df["close"].iloc[-1])

            result["fractal_bottom"] = {
                "date": str(bf.get("date", "")),
                "range_low": fx_low, "range_high": fx_high,
                "broken": last_close < fx_low,
                "confirmed": last_close > fx_high,
            }

            if last_close < fx_low:
                result["fractal_bottom"]["status"] = "failed"
                if result["construction_phase"] == "unknown":
                    result["construction_phase"] = "分型底部失败"
            elif last_close > fx_high:
                result["fractal_bottom"]["status"] = "complete"
                if result["construction_phase"] == "unknown":
                    result["construction_phase"] = "分型底部完成——至少向上一笔"
                    result["action"] = "回踩不破为二买介入点"
            else:
                result["fractal_bottom"]["status"] = "constructing"
                if result["construction_phase"] == "unknown":
                    result["construction_phase"] = "分型底部构造中"
                    result["action"] = "区间下探失败时买，不追高"
                    result["bottom_in_construction"] = True

    if result["bottom_in_construction"] and bi_list:
        result["interval_nesting_hint"] = (
            "用区间套在低级别找精确买点：次级别找背驰段→次次级别找背驰点→共振即精确买点"
        )

    result["principle"] = (
        "底部精确定义（第108课）：一买后到该中枢第一次出三买卖点前为底部构造过程。"
        "三卖先出=失败，三买先出=完成。分型底部为粗糙替代版。"
    )
    return result


# ============================================================
# 第十九部分：综合交易决策引擎
# ============================================================

def trading_decision(single_tf_result: dict,
                     multi_tf_context: dict = None) -> dict:
    """
    综合交易决策引擎

    把所有分析模块的信号综合为一个明确的交易指令。
    决策层级（缠论原文优先级）：

    第一层——防狼术（第103课）：DIFF在0轴之下 → 一票否决，所有买点无效
    第二层——机械化信号链（第105课）：分型→笔延伸是核心操作信号
    第三层——背驰+买卖点（第15/24/37课）：验证机械化信号的可靠性
    第四层——中阴阶段（第88-90课）：中阴中降低仓位/置信度
    第五层——底部构造（第108课）：底部是否完成影响入场时机
    第六层——多级别联立（第91-93课）：高级别确认提升置信度
    第七层——几何能量统一（第104课）：共振质量微调置信度

    Returns:
        verdict: 买入 / 卖出 / 持股 / 持币 / 观望
        confidence: very_high / high / medium / low
        reasoning: 决策链条说明
        actions: 具体操作步骤
    """
    result = {
        "verdict": "观望",
        "confidence": "low",
        "reasoning": [],
        "actions": [],
        "risk_flags": [],
        "supporting_signals": [],
        "opposing_signals": [],
    }

    # ================================================================
    # 第一层：防狼术（一票否决权）
    # ================================================================
    wolf = single_tf_result.get("wolf_defense", {})
    if wolf.get("danger") or not wolf.get("safe", True):
        result["risk_flags"].append("防狼术警告：MACD黄白线在0轴之下，远离不做")
        result["verdict"] = "持币"
        result["confidence"] = "very_high"
        result["reasoning"].append(
            "第103课铁律：日线DIFF在0轴之下，任何买点都不参与。"
            "在市场底部没有构造完成前，涨势只能看成反弹。"
        )
        result["actions"].append("保持空仓，等待DIFF上0轴再做考虑")
        return result

    # ================================================================
    # 第二层：机械化信号链（核心驱动）
    # ================================================================
    mech = single_tf_result.get("mechanized_signal", {})
    signal = mech.get("current_signal", "unknown")
    fx_status = mech.get("fractal_status", {})

    # ================================================================
    # 第三层：背驰 + 买卖点（验证信号）
    # ================================================================
    divergence = single_tf_result.get("divergence", {})
    has_div = divergence.get("has_divergence", False)
    div_type = divergence.get("type", "")
    div_confidence = divergence.get("confidence", "")

    trade_points = single_tf_result.get("trade_points", {})
    buy_points = trade_points.get("buy_points", []) if isinstance(trade_points, dict) else []
    sell_points = trade_points.get("sell_points", []) if isinstance(trade_points, dict) else []

    has_first_buy = any(p.get("type") == "first_buy" for p in buy_points)
    has_second_buy = any(p.get("type") == "second_buy" for p in buy_points)
    has_third_buy = any(p.get("type") == "third_buy" for p in buy_points)
    has_first_sell = any(p.get("type") == "first_sell" for p in sell_points)
    has_second_sell = any(p.get("type") == "second_sell" for p in sell_points)
    has_third_sell = any(p.get("type") == "third_sell" for p in sell_points)

    # ================================================================
    # 第四层：中阴阶段检查
    # ================================================================
    bardo = single_tf_result.get("bardo", {})
    in_bardo = bardo.get("in_bardo", False)
    bardo_health = bardo.get("health", "unknown")

    # ================================================================
    # 第五层：底部构造状态
    # ================================================================
    bottom = single_tf_result.get("bottom_construction", {})
    bottom_phase = bottom.get("construction_phase", "unknown")

    # ================================================================
    # 第六层：多级别验证
    # ================================================================
    multi_confirm = False
    if multi_tf_context:
        multi_wolf = multi_tf_context.get("multi_wolf_defense", {})
        if multi_wolf.get("all_safe"):
            multi_confirm = True
            result["supporting_signals"].append("多周期防狼术全部安全")

    # ================================================================
    # 第七层：几何能量统一
    # ================================================================
    geo = single_tf_result.get("geometry_energy", {})
    geo_alignment = geo.get("alignment", "")
    geo_unity = geo.get("unity_index", 0)

    # ================================================================
    # 决策矩阵
    # ================================================================

    # --- 持股信号 ---

    if signal == "up_stroke_extending":
        # 向上一笔延伸中 → 默认持股
        if has_div and div_type == "top_divergence":
            # 顶背驰出现 → 准备卖出
            if fx_status.get("type") == "top" and fx_status.get("confirmed"):
                result["verdict"] = "卖出"
                result["confidence"] = "high" if div_confidence in ("high", "very_high") else "medium"
                result["reasoning"].append("顶分型已确认 + 顶背驰共振 → 卖出信号可靠")
                result["actions"].append("分批卖出，至少出掉一半")
                result["actions"].append("剩余仓位等二卖确认后全部清仓")
            else:
                result["verdict"] = "减仓"
                result["confidence"] = "medium"
                result["reasoning"].append("向上一笔延伸中但已出现顶背驰，顶分型尚未确认")
                result["actions"].append("减仓至半仓以下，等待顶分型确认后清仓")
        else:
            # 无背驰 → 坚定持股
            confidence_boost = ""
            if has_third_buy:
                result["supporting_signals"].append("三买已确认——上涨趋势确立")
                confidence_boost = "very_"
            if multi_confirm:
                result["supporting_signals"].append("多级别联立确认")
                confidence_boost = "very_"
            result["verdict"] = "持股"
            result["confidence"] = f"{confidence_boost}high"
            result["reasoning"].append("向上一笔延伸中，无顶背驰，继续持有")
            result["actions"].append("持股不动，等待顶分型出现或顶背驰信号再考虑卖出")

        if in_bardo:
            result["opposing_signals"].append(f"处于中阴阶段({bardo_health})，降低仓位")
            if result["verdict"] == "持股":
                result["actions"].append("中阴阶段建议保留半仓，上抛下吸做短差")

    # --- 持股 → 准备卖出过渡 ---

    elif signal == "up_stroke_extending" and fx_status.get("type") == "top":
        # 实际上这个分支不会独立触发（已在上面处理），但作为防御性代码保留
        pass

    # --- 持币信号 ---

    elif signal == "down_stroke_extending":
        # 向下一笔延伸中 → 默认持币
        if has_div and div_type == "bottom_divergence":
            # 底背驰出现 → 准备买入
            if fx_status.get("type") == "bottom" and fx_status.get("confirmed"):
                if has_second_buy:
                    result["verdict"] = "买入"
                    result["confidence"] = "very_high"
                    result["reasoning"].append("底分型确认 + 底背驰 + 二买共振 → 最佳买入时机")
                elif has_first_buy:
                    result["verdict"] = "买入"
                    result["confidence"] = "high"
                    result["reasoning"].append("底分型确认 + 底背驰 + 一买 → 可分批介入")
                else:
                    result["verdict"] = "买入"
                    result["confidence"] = "medium"
                    result["reasoning"].append("底分型确认 + 底背驰 → 可以轻仓试探")
                result["actions"].append("分批买入，第一批仓位不超过总资金20%")
                result["actions"].append("买入后如跌破底分型低点，无条件止损")
            else:
                result["verdict"] = "准备买入"
                result["confidence"] = "medium"
                result["reasoning"].append("向下一笔延伸中 + 底背驰出现，等待底分型确认")
                result["actions"].append("密切关注，底分型一旦确认即可介入")
        else:
            # 无背驰 → 继续持币
            result["verdict"] = "持币"
            result["confidence"] = "high"
            result["reasoning"].append("向下一笔延伸中，无底背驰，继续等待")
            result["actions"].append("耐心持币，等待底分型+底背驰共振信号")

        # 底部构造中 → 提示接近买点
        if bottom_phase in ("底部构造中——等待三买或三卖", "分型底部构造中"):
            result["reasoning"].append("底部正在构造中，密切关注三类买卖点的出现")
            if result["verdict"] == "持币":
                result["actions"].append("可在中枢下沿轻仓试探，跌破DD止损")

    # --- 买点信号（已确认） ---

    elif signal in ("bottom_fractal_forming",):
        if fx_status.get("confirmed"):
            if has_second_buy or has_third_buy:
                result["verdict"] = "买入"
                result["confidence"] = "very_high"
                result["reasoning"].append("底分型确认 + 二买/三买共振 —— 最可靠的买入信号")
            elif has_first_buy:
                result["verdict"] = "买入"
                result["confidence"] = "high"
                result["reasoning"].append("底分型确认 + 一买 —— 可分批建仓")
            else:
                result["verdict"] = "买入"
                result["confidence"] = "medium"
                result["reasoning"].append("底分型确认，但无标准买卖点配合 —— 轻仓试探")
            result["actions"].append("分批买入，首仓不超过20%")
            result["actions"].append("如回踩底分型区间不破可加仓（类二买）")
        else:
            result["verdict"] = "准备买入"
            result["confidence"] = "low"
            result["reasoning"].append("底分型出现但未确认，等待第三根K线升破第一根高点")
            result["actions"].append("不要提前买入，等确认信号")

    # --- 卖点信号（已确认） ---

    elif signal in ("top_fractal_forming",):
        if fx_status.get("confirmed"):
            result["verdict"] = "卖出"
            result["confidence"] = "high" if has_div else "medium"
            result["reasoning"].append("顶分型确认，卖出")
            if has_div:
                result["reasoning"].append("顶背驰共振，卖出信号加强")
            result["actions"].append("至少卖出半仓，剩余等二卖确认后清仓")
        else:
            result["verdict"] = "准备卖出"
            result["confidence"] = "low"
            result["reasoning"].append("顶分型出现但未确认，等待第三根K线跌破第一根低点")
            result["actions"].append("密切观察，顶分型确认立即减仓")

    # --- 独立背驰信号（无机械化信号配合） ---

    elif signal in ("unknown", "no_recent_fractal", "no_fractal_data"):
        if has_div:
            if div_type == "bottom_divergence":
                result["verdict"] = "准备买入"
                result["confidence"] = "medium"
                result["reasoning"].append("底背驰出现，但机械化信号未确认，等待底分型")
            elif div_type == "top_divergence":
                result["verdict"] = "准备卖出"
                result["confidence"] = "medium"
                result["reasoning"].append("顶背驰出现，但机械化信号未确认，等待顶分型")
        else:
            result["verdict"] = "观望"
            result["confidence"] = "low"
            result["reasoning"].append("无明确机械化信号，也无背驰，建议观望等待")

    # ================================================================
    # 中阴阶段强制降级
    # ================================================================
    if in_bardo and result["verdict"] in ("买入", "持股"):
        if result["confidence"] == "very_high":
            result["confidence"] = "high"
        elif result["confidence"] == "high":
            result["confidence"] = "medium"
        result["opposing_signals"].append(
            f"中阴阶段（{bardo_health}），操作应偏保守，按中枢震荡策略上抛下吸"
        )

    # ================================================================
    # 底部构造未完成 → 买入降级
    # ================================================================
    if bottom_phase in ("底部构造中——等待三买或三卖",) and result["verdict"] == "买入":
        result["confidence"] = "medium"
        result["reasoning"].append("底部尚未完全确认（三买未出），轻仓试探为主")

    # ================================================================
    # 几何能量微调
    # ================================================================
    if geo_unity >= 70 and result["verdict"] in ("买入", "卖出"):
        if result["confidence"] == "high":
            result["confidence"] = "very_high"
        result["supporting_signals"].append(f"几何能量高度统一(unity={geo_unity:.0f})，信号加强")
    elif geo_unity < 30 and result["verdict"] in ("买入", "卖出"):
        result["opposing_signals"].append(f"几何能量分歧(unity={geo_unity:.0f})，信号可靠性降低")

    # ================================================================
    # 组装最终输出
    # ================================================================
    result["summary"] = (
        f"【{result['verdict']}】(置信度:{result['confidence']})。"
        f"{' '.join(result['reasoning'])}"
    )

    return result


def multi_timeframe_decision(multi_tf_result: dict) -> dict:
    """
    多级别联立综合决策

    将同一只股票的多周期分析结果综合为一个跨级别决策。
    原则：大级别定方向，小级别定节奏。
    只有在大小级别方向一致时才重仓操作。
    """
    tf_results = multi_tf_result.get("timeframe_results", {})

    decisions = {}
    for level_key, r in tf_results.items():
        decisions[level_key] = trading_decision(r, multi_tf_result)

    result = {
        "decisions": decisions,
        "verdict": "观望",
        "confidence": "low",
        "reasoning": [],
        "actions": [],
    }

    # 大级别定方向
    daily_dec = decisions.get("daily", {})
    weekly_dec = decisions.get("weekly", {})
    min30_dec = decisions.get("30min", {})

    daily_verdict = daily_dec.get("verdict", "观望")
    weekly_verdict = weekly_dec.get("verdict", "观望")

    # 周线持股 + 日线持股 → 满仓持有
    if weekly_verdict in ("持股", "买入") and daily_verdict in ("持股",):
        result["verdict"] = "满仓持股"
        result["confidence"] = "very_high"
        result["reasoning"].append("周线级别向上 + 日线级别向上 → 大小级别共振，最佳持仓状态")
        result["actions"].append("坚定持有，日线级别震荡无须理会")
        result["actions"].append("只有周线出现顶分型+顶背驰才考虑减仓")

    # 周线持股 + 日线持币/准备买入 → 等日线买点
    elif weekly_verdict in ("持股", "买入") and daily_verdict in ("持币", "准备买入", "观望"):
        result["verdict"] = "等待日线买点"
        result["confidence"] = "high"
        result["reasoning"].append("周线向上趋势中，日线回调是买入机会")
        result["actions"].append("在日线底分型+底背驰处积极买入")
        result["actions"].append("这是'大级别向上中的小级别回调'——最安全的买点")

    # 周线持币 + 日线持股/买入 → 仅短线参与
    elif weekly_verdict in ("持币", "卖出", "准备卖出") and daily_verdict in ("持股", "买入"):
        result["verdict"] = "短线轻仓"
        result["confidence"] = "medium"
        result["reasoning"].append("周线向下但日线有反弹，只能短线轻仓参与")
        result["actions"].append("仓位不超过20%，日线卖点出现立即离场")
        result["actions"].append("不要贪恋，这是逆大势操作")

    # 周线持币 + 日线持币 → 完全空仓
    elif weekly_verdict in ("持币", "卖出") and daily_verdict in ("持币", "卖出"):
        result["verdict"] = "完全空仓"
        result["confidence"] = "very_high"
        result["reasoning"].append("周线日线双空——大小级别共振向下，持有现金是最佳策略")
        result["actions"].append("耐心等待周线级别底部分型出现")

    # 周线+日线都准备 → 密切关注
    elif "准备" in weekly_verdict and "准备" in daily_verdict:
        result["verdict"] = "密切关注"
        result["confidence"] = "medium"
        result["reasoning"].append("多级别都在等待确认信号，变盘在即")
        result["actions"].append("做好双向准备，信号确认后立即行动")

    else:
        result["verdict"] = "观望"
        result["confidence"] = "low"
        result["reasoning"].append("多级别信号不一致或不确定，观望为佳")

    # 加入小区间套提示
    interval = multi_tf_result.get("interval_nesting", {})
    if interval.get("resonance_depth", 0) >= 2:
        result["reasoning"].append(f"区间套{interval['resonance_depth']}级共振——精确买卖点可信")

    result["summary"] = (
        f"【{result['verdict']}】(置信度:{result['confidence']})。"
        f"{' '.join(result['reasoning'])}"
    )

    return result


def quick_weekly_check(symbol: str) -> dict:
    """
    轻量级周线交叉验证——专用于扫描模式

    扫描模式只跑日线，缺少大级别验证会导致"日线看多但周线空头"的误判。
    只做最低开销检查：防狼术（DIFF是否在0轴之上）+ 笔方向。

    Returns:
        warning: 是否需要降级（周线空头 = True）
    """
    try:
        df = fetch_kline(symbol, period="weekly")
        if len(df) < 20:
            return {"data_ok": False, "warning": False, "reason": "周线数据不足（<20条）"}

        df = compute_macd(df)
        last_diff = float(df["diff"].iloc[-1])
        diff_above_zero = last_diff > 0

        direction = "unknown"
        recent = df["diff"].tail(5)
        diffs = recent.diff().dropna()
        if len(diffs) >= 2:
            if all(d > 0 for d in diffs):
                direction = "up"
            elif all(d < 0 for d in diffs):
                direction = "down"

        reasons = []
        if not diff_above_zero:
            reasons.append("周线DIFF在0轴之下——大级别空头主导")
        if direction == "down":
            reasons.append("周线DIFF持续下行——周线级别调整未结束")

        return {
            "data_ok": True,
            "warning": not diff_above_zero or direction == "down",
            "diff_above_zero": diff_above_zero,
            "last_diff": round(last_diff, 4),
            "stroke_direction": direction,
            "reason": "；".join(reasons) if reasons else "",
        }
    except Exception as e:
        return {"data_ok": False, "warning": False, "reason": f"周线检查失败: {e}"}
