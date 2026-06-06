#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论完整分析引擎 —— 基于《教你炒股票108课》完整理论体系

分析管线（完整版）：
  多周期数据获取 → 各周期独立形态学分析（包含→分型→笔→线段）
  → f1递归（线段→中枢）→ f2递归（走势类型→高级别中枢）
  → 动力学分析（走势类型→严格背驰→买卖点）
  → 策略层综合研判（防狼术→表里关系→中阴→区间套→同级别分解→板块）
  → 多级别联立综合报告
"""

import sys
import os
import time
import warnings
import webbrowser
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# 确保 chan_lib 包在 sys.path 上（兼容 skill 解压后的非包环境）
_srcdir = os.path.dirname(os.path.abspath(__file__))
if _srcdir not in sys.path:
    sys.path.insert(0, _srcdir)

import numpy as np
import pandas as pd

from chan_lib.data import fetch_kline, fetch_multi_timeframe, fetch_csi300_stocks
from chan_lib.morphology import run_morphology
from chan_lib.dynamics import compute_macd, run_dynamics, identify_zhongshu
from chan_lib.strategy import (
    check_wolf_defense, multi_timeframe_wolf_defense,
    check_dual_table_relation, dual_table_level_strategy,
    multi_timeframe_dual_table, zhongshu_monitor,
    identify_bardo, interval_nesting_analysis,
    same_level_decomposition, check_small_large_divergence,
    compute_sector_strength_class, compute_boll,
    position_complete_classification,
    multi_meaning_decomposition,
    third_buy_mechanized_operation,
    analyze_multi_level_landmarks,
    capital_management, cost_zero_tracker,
    daily_trend_classification, overnight_rebound_analysis,
    geometry_energy_correlation, mechanized_signal_chain,
    bottom_construction_tracker,
    trading_decision, multi_timeframe_decision,
)

warnings.filterwarnings("ignore")


# 周期中文映射
PERIOD_CN = {
    "daily": "日线", "weekly": "周线", "monthly": "月线",
    "30min": "30分钟", "60min": "60分钟", "5min": "5分钟", "1min": "1分钟",
    "30": "30分钟", "60": "60分钟",  # 兼容 fetch_multi_timeframe 的短键名
}


def _period_cn(key: str) -> str:
    """将内部周期键转为中文显示名"""
    return PERIOD_CN.get(key, key)


# ============================================================
# 第一部分：单周期完整分析
# ============================================================

def analyze_single_timeframe(df: pd.DataFrame, name: str = "",
                             quiet: bool = False) -> dict:
    """
    对单一周期执行完整缠论分析管线
    """
    if len(df) < 30:
        return {}

    # 大级别容错：按周期分级最低K线数
    period = df.attrs.get("period", "daily") if hasattr(df, "attrs") else "daily"
    min_bars = {"monthly": 12, "weekly": 20}.get(period, 30)
    if len(df) < min_bars:
        return {}

    df = compute_macd(df)
    df = compute_boll(df)  # 第90课：布林通道
    df_clean = run_morphology(df)

    bi_list = df_clean.attrs.get("bi_list", [])
    segment_list = df_clean.attrs.get("segment_list", [])

    dynamics = run_dynamics(df, df_clean)

    # 计算次级别中枢列表（用于中枢震荡监控的预警1/2）
    sub_zs_list = None
    if segment_list and bi_list:
        sub_zs_df = identify_zhongshu(bi_list, label="bi")
        sub_zs_list = sub_zs_df.attrs.get("zhongshu_list", [])

    wolf = check_wolf_defense(df)
    dual_state = check_dual_table_relation(df_clean, bi_list,
                                           dynamics.get("zhongshu_list", []))
    dual_strategy = dual_table_level_strategy(df, bi_list,
                                              dynamics.get("divergence", {}),
                                              dual_state)
    zs_monitor = zhongshu_monitor(df,
                                  dynamics.get("zhongshu_list", []),
                                  dynamics.get("sub_elements", []),
                                  sub_zs_list)
    bardo = identify_bardo(df, dynamics.get("divergence", {}),
                           dynamics.get("zhongshu_list", []), bi_list)
    same_level = same_level_decomposition(bi_list)
    position_cls = position_complete_classification(
        df_clean, dynamics.get("zhongshu_list", []), bi_list
    )
    sector_strength = compute_sector_strength_class(df)
    multi_decomp = multi_meaning_decomposition(
        dynamics.get("sub_elements", [])
    )
    third_buy_ops = third_buy_mechanized_operation(
        dynamics.get("trade_points", {}),
        dynamics.get("zhongshu_list", []),
        bi_list,
    )

    # 第31课：资金管理
    cap_mgmt = capital_management(
        current_capital=100000,  # 默认参考值，实际使用时应传入真实数据
        signal_confidence=0.5,
    )

    # 第104课：几何能量统一
    geo_energy = geometry_energy_correlation(
        dynamics.get("zhongshu_list", []),
        dynamics.get("divergence", {}),
        bi_list,
    )

    # 第105课：机械化操作信号链
    mech_signal = mechanized_signal_chain(df_clean, bi_list)

    # 第108课：底部构造跟踪
    bottom_track = bottom_construction_tracker(
        df_clean, dynamics.get("zhongshu_list", []),
        dynamics.get("trade_points", {}), bi_list,
    )

    # 第46课+第47课：当日走势分类 + 一夜情分析（仅30分钟周期时执行）
    daily_class = {}
    overnight = {}
    if df_clean.attrs.get("period") == "30min" and len(df) >= 8:
        daily_class = daily_trend_classification(df_clean)
        overnight = overnight_rebound_analysis(df_clean)

    return {
        "symbol": name, "date": df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "last_price": float(df["close"].iloc[-1]),
        "kline_count": len(df),
        "_raw_df": df,  # 原始数据，供图表生成
        "bi_list": bi_list, "bi_count": len(bi_list),
        "segment_list": segment_list, "segment_count": len(segment_list),
        "zhongshu_list": dynamics.get("zhongshu_list", []),
        "zs_count": dynamics.get("zs_count", 0),
        "zoushi": dynamics.get("zoushi", {}),
        "divergence": dynamics.get("divergence", {}),
        "reversal": dynamics.get("reversal", {}),
        "trade_points": dynamics.get("trade_points", {}),
        "wolf_defense": wolf,
        "dual_state": dual_state,
        "dual_strategy": dual_strategy,
        "zhongshu_monitor": zs_monitor,
        "bardo": bardo,
        "same_level_decomp": same_level,
        "position_classification": position_cls,
        "sector_strength": sector_strength,
        "multi_meaning_decomp": multi_decomp,
        "third_buy_operation": third_buy_ops,
        "capital_management": cap_mgmt,
        "geometry_energy": geo_energy,
        "mechanized_signal": mech_signal,
        "bottom_construction": bottom_track,
        "daily_classification": daily_class,
        "overnight_analysis": overnight,
    }


# ============================================================
# 第二部分：多级别联立分析
# ============================================================

def analyze_multi_timeframe(symbol: str, name: str = "",
                            periods: List[str] = None,
                            quiet: bool = False,
                            start_date: str = "20200101",
                            end_date: str = None) -> dict:
    """
    多级别联立缠论分析
    默认获取30分钟/日线/周线/月线四个周期，逐级递归
    """
    if periods is None:
        periods = ["30", "daily", "weekly", "monthly"]

    if not quiet:
        print(f"\n{'='*70}")
        print(f"  《缠论》股析 - 多级别联立：{symbol} {name}")
        print(f"{'='*70}")
        print(f"  [1/4] 获取多周期数据 ({', '.join(periods)})...")

    tf_data = fetch_multi_timeframe(symbol, periods=periods, start_date=start_date,
                                    end_date=end_date)

    results = {}
    level_names = []

    for level_key, df in tf_data.items():
        if len(df) < 30:
            if not quiet:
                print(f"    {level_key}: 数据不足（{len(df)}条），跳过")
            continue

        level_names.append(level_key)
        if not quiet:
            print(f"  [2/4] {level_key} 周期分析... ({len(df)}条K线)")

        results[level_key] = analyze_single_timeframe(df, name=name, quiet=True)

    if not quiet:
        print(f"  [3/4] 多级别联立研判...")

    # 多级别联立
    multi_wolf = multi_timeframe_wolf_defense({
        k: v["wolf_defense"] for k, v in results.items()
    }) if results else {}

    multi_dual = None
    if "daily" in results and ("weekly" in results or "60min" in results):
        high_key = "weekly" if "weekly" in results else "60min"
        multi_dual = multi_timeframe_dual_table({
            "daily": results["daily"].get("dual_state", {}),
            high_key: results.get(high_key, {}).get("dual_state", {}),
        })

    # 区间套
    df_map = {k: tf_data[k] for k in level_names if k in tf_data}
    interval_result = interval_nesting_analysis(df_map, "daily")

    # 小背驰-大转折
    small_large = {}
    if "30min" in results and "daily" in results:
        small_large = check_small_large_divergence(
            {"divergence": results["30min"].get("divergence", {}),
             "trade_points": results["30min"].get("trade_points", {})},
            {"divergence": results["daily"].get("divergence", {}),
             "trade_points": results["daily"].get("trade_points", {})},
        )

    # 8级递归记号体系（第88课）
    landmarks = analyze_multi_level_landmarks(results)

    # === 综合交易决策 ===
    # 对每个周期单独做决策
    single_decisions = {}
    for level_key, r in results.items():
        single_decisions[level_key] = trading_decision(r, None)

    # 多级别联立决策
    multi_decision = None
    if len(results) >= 2:
        multi_decision = multi_timeframe_decision({
            "timeframe_results": results,
            "multi_wolf_defense": multi_wolf,
            "multi_dual_table": multi_dual,
            "interval_nesting": interval_result,
        })

    if not quiet:
        print(f"  [4/4] 生成综合报告...\n")

    return {
        "symbol": symbol, "name": name,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "levels_analyzed": level_names,
        "timeframe_results": results,
        "multi_wolf_defense": multi_wolf,
        "multi_dual_table": multi_dual,
        "interval_nesting": interval_result,
        "small_large_divergence": small_large,
        "landmarks": landmarks,
        "single_decisions": single_decisions,
        "multi_decision": multi_decision,
    }


# ============================================================
# 第三部分：综合报告
# ============================================================

def _translate_zoushi_type(t: str) -> str:
    """把走势类型翻译成大白话"""
    m = {
        "uptrend": "中长期处于上升通道，大方向向好",
        "downtrend": "中长期处于下降通道，大方向偏弱",
        "consolidation": "目前处于盘整格局，多空拉锯，等待选择方向",
        "consolidation_with_expansion": "中枢扩张中，多空双方争夺加剧，变盘在即",
        "unknown": "走势结构尚不完整，信号不明确",
    }
    return m.get(t, t)


def _zoushi_lesson(zt: str) -> str:
    """根据走势类型返回对应的课程序号"""
    m = {
        "uptrend": "第15/17课：上涨趋势由两个及以上同向不重叠中枢构成，趋势必完美",
        "downtrend": "第15/17课：下跌趋势由两个及以上同向不重叠中枢构成，趋势必完美",
        "consolidation": "第18课：盘整是只有一个该级别中枢的走势类型",
        "consolidation_with_expansion": "第20课：中枢扩张形成更高级别中枢",
        "unknown": "第15课：走势类型分为上涨趋势、下跌趋势、盘整三种",
    }
    return m.get(zt, m["unknown"])


def _translate_zs_fate(fate: str) -> str:
    m = {
        "延续": "价格还在中枢范围内震荡，像车还没开出停车场——方向没出来，耐心等",
        "三类买点": "价格向上突破中枢后回踩确认——多头确认，回调不破ZG即三买",
        "三类卖点": "价格向下跌破中枢后反弹确认——空头确认，反弹不破ZD即三卖",
        "扩张": "中枢区间在变大，多空双方越打越激烈，变盘在即",
        "新生": "旧中枢被打破，新的筹码密集区正在形成——趋势在转移",
        "active": "最新中枢仍在活动，走势围绕它展开",
        "upgrade": "中枢延伸超过9段，升级为更高级别中枢",
    }
    return m.get(fate, fate)


def _translate_divergence(div: dict) -> tuple:
    """背驰翻译，返回 (白话结论, 课程依据)"""
    if not div.get("has_divergence"):
        return ("下跌动能和上涨动能旗鼓相当，没有一方衰竭的信号，方向不明确",
                "第15/24课：背驰是趋势结束的标志，必须有力度衰减信号")
    d = div.get("details", {})
    direction = d.get("direction", "")
    conf = div.get("confidence", "")
    div_type = div.get("type", "")
    conditions = div.get("conditions_check", [])

    lesson = ""
    if div_type == "trend_divergence":
        lesson = "第24/37课：趋势背驰须满足6条件——同级别中枢、c含三类买卖点、b级别不大于c、c创新高/低、B拉回0轴、力度背驰"
    elif div_type == "consolidation_divergence":
        lesson = "第27课：盘整背驰——中枢震荡中离开段力度小于进入段力度"

    if direction == "bottom_divergence":
        base = "【下跌动能衰竭】卖的人没劲了——像皮球拍下去弹不起来了"
        if conf in ("high", "very_high"):
            base += "，反弹概率很大，值得积极关注"
        else:
            base += "，不过衰减力度还不够强，需要再观察确认一下"
        return (base, lesson)
    elif direction == "top_divergence":
        base = "【上涨动能衰竭】买的人没劲了——像人跑累了速度在下降"
        if conf in ("high", "very_high"):
            base += "，回调风险较大，应考虑减仓"
        else:
            base += "，不过衰减力度还不够强，需要再观察确认"
        return (base, lesson)
    return ("", lesson)


def _translate_dual_state(state: str) -> tuple:
    """表里关系翻译，返回 (白话结论, 课程依据)"""
    m = {
        "(1,1)": ("【持股待涨】处于向上攻击状态，像上楼梯一样步步高",
                  "第91-93课：(1,1)表示向上笔延伸中，是最健康的持股状态"),
        "(-1,1)": ("【持币等待】处于向下回落状态，像下楼梯还在往下走，别急着接飞刀",
                   "第91-93课：(-1,1)表示向下笔延伸中，操作级别应持币"),
        "(1,0)": ("【警惕变盘】上涨中出现停顿（顶分型），像车踩了刹车——可能歇口气，也可能掉头",
                  "第91-93课：(1,0)表示向上笔顶分型构造中，是卖出的预警信号"),
        "(-1,0)": ("【准备机会】下跌中出现停顿（底分型），像球落地——可能正在筑底，关注确认",
                   "第91-93课：(-1,0)表示向下笔底分型构造中，是买入的准备信号"),
    }
    return m.get(state, (state, ""))


def _translate_trade_point(tp_type: str) -> tuple:
    """买卖点翻译，返回 (白话解释, 课程依据)"""
    m = {
        "一买": ("第一类买点——下跌趋势末端背驰点，利润最大但风险最高，像抄底",
                 "第20/21课：一买是下跌趋势最后一个中枢下方，趋势背驰的终点"),
        "二买": ("第二类买点——一买后的回踩确认，安全性最高散户首选，像二次探底不破",
                 "第21/101课：二买是一买后第一次次级别回调不破前低的点"),
        "三买": ("第三类买点——突破主力成本区后回抽站稳，效率最高，像突破后确认",
                 "第20课：三买是次级别离开中枢后回抽不破ZG的点"),
        "一卖": ("第一类卖点——上涨趋势末端背驰点，利润该兑现了，像山顶该下山了",
                 "第20/21课：一卖是上涨趋势最后一个中枢上方，趋势背驰的终点"),
        "二卖": ("第二类卖点——一卖后的反弹确认，最后逃跑窗口",
                 "第21课：二卖是一卖后第一次反弹不创新高的点"),
        "三卖": ("第三类卖点——跌破主力成本区后反弹确认，必须止损的信号",
                 "第20课：三卖是次级别跌破中枢后反弹不破ZD的点"),
        "二买=三买重合": ("最强势启动信号——二买和三买在同一位置，力度超强",
                          "第101课：二买三买重合是最强的底部信号"),
    }
    return m.get(tp_type, (tp_type, ""))


def _format_lesson(num: str, title: str) -> str:
    """格式化课程序号"""
    return f"第{num}课《{title}》"


# ============================================================
# 第三部分：综合报告（新版--双层解释：缠论依据 + 白话解读）
# ============================================================

def print_full_report(result: dict):
    """打印完整分析报告——每项结论都包含缠论依据和白话解读"""
    if not result or not result.get("timeframe_results"):
        print("无分析结果")
        return

    tf_results = result["timeframe_results"]
    levels_str = ', '.join(_period_cn(k) for k in result['levels_analyzed'])

    # 获取数据区间信息
    daily_r = tf_results.get("daily", {})
    first_r = next(iter(tf_results.values()), {})
    raw_df = first_r.get("_raw_df")
    date_range = ""
    if raw_df is not None and len(raw_df) > 0:
        d0 = raw_df["date"].iloc[0].strftime("%Y-%m-%d") if hasattr(raw_df["date"].iloc[0], "strftime") else str(raw_df["date"].iloc[0])[:10]
        d1 = raw_df["date"].iloc[-1].strftime("%Y-%m-%d") if hasattr(raw_df["date"].iloc[-1], "strftime") else str(raw_df["date"].iloc[-1])[:10]
        date_range = f"数据区间：{d0} ~ {d1}"

    title_str = f"{result['symbol']} {result['name']}"

    print(f"\n{'═'*66}")
    print(f"  《缠论》股析 —— {title_str}")
    print(f"  分析时间：{result['analysis_date']}  |  分析级别：{levels_str}")
    if date_range:
        print(f"  {date_range}")
    print(f"{'═'*66}")

    # ================================================================
    # 先取关键数据
    # ================================================================
    multi_wolf = result.get("multi_wolf_defense", {})
    if multi_wolf:
        daily_wolf = multi_wolf.get("daily", {})
        weekly_wolf = multi_wolf.get("weekly", {})
    else:
        daily_wolf = daily_r.get("wolf_defense", {})
        weekly_wolf = {}

    multi_dec = result.get("multi_decision", {})
    single_decs = result.get("single_decisions", {})

    # ================================================================
    # 综合决策（最顶层）
    # ================================================================
    if multi_dec:
        verdict = multi_dec.get("verdict", "?")
        conf = multi_dec.get("confidence", "?")
        conf_cn = {"very_high": "很高", "high": "较高", "medium": "中等", "low": "偏低"}.get(conf, conf)
        print(f"\n  ┌{'─'*62}┐")
        print(f"  │  综合决策：【{verdict}】  可靠性：{conf_cn}")
        if multi_dec.get("reasoning"):
            for r_line in multi_dec["reasoning"]:
                print(f"  │  → {r_line}")
        print(f"  └{'─'*62}┘")
        if multi_dec.get("actions"):
            print(f"\n  【操作建议】")
            for i, act in enumerate(multi_dec.get("actions", []), 1):
                print(f"    {i}. {act}")
    else:
        for level_key, dec in single_decs.items():
            v = dec.get("verdict", "?")
            c = dec.get("confidence", "?")
            print(f"\n  ┌{'─'*62}┐")
            print(f"  │  {level_key}决策：【{v}】  可靠性：{c}")
            if dec.get("summary"):
                print(f"  │  {dec['summary']}")
            print(f"  └{'─'*62}┘")

    # ================================================================
    # 一、防狼术（第103课）
    # ================================================================
    print(f"\n{'─'*66}")
    print(f"  一、防狼术 —— 能不能做这只票的底线判断")
    print(f"  课程出处：《教你炒股票》第103课「防狼术」")
    print(f"  核心原理：MACD黄白线（DIFF/DEA）在0轴之下代表空头主导，任何买点都不参与")
    print(f"{'─'*66}")

    diff_val = daily_wolf.get("diff", 0)
    dea_val = daily_wolf.get("dea", 0)
    print(f"  DIFF = {diff_val:.4f}  |  DEA = {dea_val:.4f}")

    if daily_wolf.get("safe"):
        print(f"  【缠论判断】DIFF > 0 且 DEA > 0 → 在安全区，符合操作前提")
        print(f"  【白话结论】趋势指标在多方区域，这只票过了第一关，可以继续往下分析")
    elif daily_wolf.get("warning"):
        print(f"  【缠论判断】DIFF 和 DEA 在0轴附近分叉 → 临界区，方向待选择")
        print(f"  【白话结论】趋势指标在纠结，向上向下都可能，只能轻仓试探，不可重仓")
    elif daily_wolf.get("danger"):
        print(f"  【缠论判断】DIFF < 0 且 DEA < 0 → 在空头区，严令禁止操作")
        print(f"  【白话结论】趋势指标在空方区域——像有狼群出没，别进去！等MACD白线回到0轴之上再考虑")

    if weekly_wolf:
        print()
        if weekly_wolf.get("safe"):
            print(f"  周线交叉验证：周线DIFF也在0轴之上 → 大级别也是多头，安全边际加倍")
        elif weekly_wolf.get("danger"):
            print(f"  周线交叉验证：周线DIFF在0轴之下！→ 日线虽然在安全区，但大级别还是空头，需要降低预期")

    v = multi_wolf.get("verdict", "") if multi_wolf else ""
    is_danger = v == "禁止操作" or daily_wolf.get("danger", False)
    daily_wolf_safe = daily_wolf.get("safe", False)

    if is_danger:
        print(f"\n  {'!'*60}")
        print(f"  !! 防狼术警告：基础条件不满足，暂时不宜操作 !!")
        print(f"  一定要等MACD黄白线重新站上0轴再来看")
        print(f"  以下各级别分析仅供参考结构认知，不代表操作建议")
        print(f"  {'!'*60}")

    # ================================================================
    # 各周期详细分析
    # ================================================================
    section_num = 1
    for level_key in result["levels_analyzed"]:
        r = tf_results.get(level_key, {})
        if not r:
            continue

        section_num += 1
        section_titles = {2: "二", 3: "三", 4: "四", 5: "五", 6: "六"}
        sec_label = section_titles.get(section_num, str(section_num))

        print(f"\n{'─'*66}")
        print(f"  {sec_label}、{_period_cn(level_key)}级别详细分析")
        print(f"  最新收盘价：{r.get('last_price', 0):.2f}  |  K线数量：{r.get('kline_count', 0)} 根")

        # --- 走势类型 ---
        zoushi = r.get("zoushi", {})
        zt = zoushi.get("type", "unknown")
        print(f"\n  ▎走势类型判断")
        print(f"  【课程依据】{_zoushi_lesson(zt)}")
        print(f"  【技术判断】走势类型 = {zt}")
        print(f"  【白话解读】{_translate_zoushi_type(zt)}")

        if zoushi.get("latest_zs"):
            zs = zoushi["latest_zs"]
            print(f"\n  ▎中枢分析（主力成本区）")
            print(f"  【课程依据】第17/18/20课：中枢=至少三段次级别走势重叠区间，是多空博弈的核心地带")
            print(f"  【技术数据】ZG={zs['zg']:.2f}  ZD={zs['zd']:.2f}  ZZ={zs.get('zz', 0):.2f}")
            print(f"  【白话解读】近期的密集成交区在 [{zs['zd']:.2f} ~ {zs['zg']:.2f}]，是'大资金打架'的战场")
            fate = zs.get("fate", "")
            print(f"    中枢状态：{fate} → {_translate_zs_fate(fate)}")

        # --- 背驰 ---
        divergence = r.get("divergence", {})
        print(f"\n  ▎力量判断（有没有力竭）")
        div_text, div_lesson = _translate_divergence(divergence)
        if div_lesson:
            print(f"  【课程依据】{div_lesson}")
        if divergence.get("has_divergence"):
            d = divergence.get("details", {})
            print(f"  【技术数据】类型={divergence['type']}  方向={d.get('direction','')}  可靠性={divergence.get('confidence','')}")
            if divergence.get("conditions_check"):
                print(f"    6条件检查：")
                for cond_name, cond_result, cond_note in divergence["conditions_check"]:
                    mark = "✓" if cond_result else "✗"
                    print(f"      {mark} {cond_name}: {cond_note}")
        else:
            print(f"  【技术数据】未检测到背驰")
        print(f"  【白话解读】{div_text}")

        # --- 买卖点 ---
        tp = r.get("trade_points", {})
        print(f"\n  ▎买卖点（什么时候动手）")
        print(f"  【课程依据】第20/21/53/101课：三类买卖点分别是趋势背驰终点、回踩确认点、突破回抽点")
        if tp.get("buy_points"):
            for bp in tp["buy_points"]:
                conf = bp.get("confidence", "")
                star = "★★★" if conf in ("high", "very_high") else ("★★" if conf == "medium" else "★")
                plain, lesson = _translate_trade_point(bp["type"])
                print(f"    ★ {bp['type']} {star} 可靠性={conf}")
                print(f"      信号：{bp['description']}")
                print(f"      白话：{plain}")
                if bp.get("quality"):
                    print(f"      质量：{bp['quality']}")
        if tp.get("sell_points"):
            for sp in tp["sell_points"]:
                conf = sp.get("confidence", "")
                star = "★★★" if conf in ("high", "very_high") else ("★★" if conf == "medium" else "★")
                plain, lesson = _translate_trade_point(sp["type"])
                print(f"    ☆ {sp['type']} {star} 可靠性={conf}")
                print(f"      信号：{sp['description']}")
                print(f"      白话：{plain}")
        if not tp.get("buy_points") and not tp.get("sell_points"):
            print(f"    暂无标准买卖点——没信号的时候不动手，强行操作等于赌博")

        # --- 表里关系 ---
        dual = r.get("dual_state", {})
        state = dual.get("state", "?")
        plain, lesson = _translate_dual_state(state)
        print(f"\n  ▎当前状态（表里关系）")
        print(f"  【课程依据】{lesson}")
        print(f"  【技术数据】当前状态 = {state}")
        print(f"  【白话解读】{plain}")

        # --- 中枢监视器 ---
        zs_mon = r.get("zhongshu_monitor", {})
        if zs_mon.get("active"):
            pos = zs_mon.get("position", "")
            pos_cn = {
                "above_zs": "价格在主力成本区上方——多头占优，像站在楼上往下看",
                "below_zs": "价格在主力成本区下方——空头占优，像掉到楼下了",
                "inside_zs": "价格在主力成本区里面——多空拉锯战，方向不明确",
            }.get(pos, pos)
            print(f"\n  ▎中枢位置判断（第92课：中枢震荡监视器）")
            print(f"  【白话解读】{pos_cn}")
            if zs_mon.get("alerts"):
                for a in zs_mon["alerts"]:
                    print(f"    !! {a.get('desc', '')}")

        # --- 中阴 ---
        bardo = r.get("bardo", {})
        if bardo.get("active"):
            print(f"\n  ▎中阴阶段（第88-90课：走势类型的死亡与新生的过渡期）")
            print(f"  【白话解读】走势因背驰而'死亡'，当前处于方向选择的过渡期——像车到了十字路口")
            print(f"    阶段：{bardo.get('description', '')}")
            print(f"    策略：{bardo.get('action', '')}")
            if bardo.get("health_note"):
                print(f"    健康度：{bardo['health_note']}")

        # --- 底部构造 ---
        bottom = r.get("bottom_construction", {})
        if bottom.get("construction_phase") and bottom["construction_phase"] != "unknown":
            print(f"\n  ▎底部构造（第108课：底部结构的精确定义）")
            print(f"  【技术判断】{bottom['construction_phase']}")
            print(f"  【白话解读】{bottom.get('action', '')}")

    # ================================================================
    # 最终大白话总结
    # ================================================================
    print(f"\n{'═'*66}")
    print(f"  白话总结（说人话版）")
    print(f"{'═'*66}")

    if not daily_wolf_safe:
        print(f"\n  !! 防狼术未通过，以上分析仅用于学习走势结构，不代表操作建议 !!")
        print(f"{'═'*66}\n")
        return

    # 走势方向
    zt = daily_r.get("zoushi", {}).get("type", "")
    zoushi_plain = {
        "uptrend": "中长期处于上升通道，大方向由多头掌控",
        "downtrend": "中长期处于下降通道，空头占主导",
        "consolidation": "处于震荡格局，没有明确趋势方向",
    }.get(zt, "走势结构不清晰")
    print(f"\n  ① 大方向：{zoushi_plain}")

    # 表里关系
    state = daily_r.get("dual_state", {}).get("state", "")
    state_plain = {
        "(1,1)": "当前在上涨途中，持股的人继续拿，没买的等回调别追高",
        "(-1,1)": "当前在下跌途中，持币观望，等跌透了再说",
        "(1,0)": "上涨中出现了停顿信号（顶分型），要提高警惕，不是马上卖但要盯紧了",
        "(-1,0)": "下跌中出现了停顿信号（底分型），机会可能在酝酿，关注后续确认",
    }.get(state, "状态不明")
    print(f"  ② 当前姿态：{state_plain}")

    # 动能
    div = daily_r.get("divergence", {})
    if div.get("has_divergence"):
        dd = div.get("details", {})
        if dd.get("direction") == "bottom_divergence":
            print(f"  ③ 下跌动能：已经在衰减——卖的人没力气了，底部可能不远")
        elif dd.get("direction") == "top_divergence":
            print(f"  ③ 上涨动能：已经在衰减——买的人没力气了，顶部风险在增加")
    else:
        print(f"  ③ 多空力量：双方势均力敌，没有哪一方出现明显的力竭")

    # 买卖点
    tp = daily_r.get("trade_points", {})
    if tp.get("buy_points"):
        names = [bp["type"] for bp in tp["buy_points"]]
        print(f"  ④ 买点：出现了 {'、'.join(names)}——可以考虑按纪律入场")
    elif tp.get("sell_points"):
        names = [sp["type"] for sp in tp["sell_points"]]
        print(f"  ④ 卖点：出现了 {'、'.join(names)}——建议考虑兑现或减仓")
    else:
        print(f"  ④ 操作信号：暂无——这种时候不动手就是最好的操作")

    # 位置
    zs_pos = daily_r.get("zhongshu_monitor", {}).get("position", "")
    pos_plain = {
        "above_zs": "价格站在主力成本区上方，多头占优",
        "below_zs": "价格在主力成本区下方，上面有套牢盘压着",
        "inside_zs": "价格在主力成本区里面，多空还在争夺",
    }.get(zs_pos, "位置不明")
    print(f"  ⑤ 相对位置：{pos_plain}")

    # 一句话结论
    print(f"\n  ╔{'═'*62}╗")
    st = daily_r.get("dual_state", {}).get("state", "")
    if st == "(1,1)" and not tp.get("sell_points"):
        print(f"  ║  一句话：趋势向好，继续持有或等回调加仓")
    elif st == "(-1,0)" and div.get("has_divergence"):
        print(f"  ║  一句话：机会在酝酿，密切关注底分型确认信号")
    elif st == "(-1,1)":
        print(f"  ║  一句话：下跌还没完，持币观望别手痒")
    elif st == "(1,0)" and tp.get("sell_points"):
        print(f"  ║  一句话：上涨遇阻了，可以考虑分批兑现利润")
    elif tp.get("buy_points"):
        print(f"  ║  一句话：买点已现，按纪律入场，设好止损位")
    elif tp.get("sell_points"):
        print(f"  ║  一句话：卖点已现，按纪律离场，别贪最后一口")
    else:
        print(f"  ║  一句话：信号不明确的时候，不动就是赚")
    print(f"  ╚{'═'*62}╝")

    print(f"\n{'═'*66}\n")


# ============================================================
# HTML 报告生成器
# ============================================================

HTML_REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def generate_html_report(result: dict, chart_images: dict = None) -> str:
    """
    生成带K线图的HTML分析报告

    Args:
        result: print_full_report 同款 result 字典
        chart_images: {level_key: base64_png_string} 各周期的K线图

    Returns:
        HTML文件路径
    """
    if chart_images is None:
        chart_images = {}

    tf_results = result.get("timeframe_results", {})
    daily_r = tf_results.get("daily", {})
    first_r = next(iter(tf_results.values()), {})
    raw_df = first_r.get("_raw_df")

    date_range = ""
    if raw_df is not None and len(raw_df) > 0:
        d0 = raw_df["date"].iloc[0].strftime("%Y-%m-%d") if hasattr(raw_df["date"].iloc[0], "strftime") else str(raw_df["date"].iloc[0])[:10]
        d1 = raw_df["date"].iloc[-1].strftime("%Y-%m-%d") if hasattr(raw_df["date"].iloc[-1], "strftime") else str(raw_df["date"].iloc[-1])[:10]
        date_range = f"{d0} ~ {d1}"

    title_str = f"{result['symbol']} {result['name']}"
    levels_str = ", ".join(_period_cn(k) for k in result.get("levels_analyzed", ["daily"]))
    multi_wolf = result.get("multi_wolf_defense", {})
    if multi_wolf:
        daily_wolf = multi_wolf.get("daily", {})
        weekly_wolf = multi_wolf.get("weekly", {})
    else:
        daily_wolf = daily_r.get("wolf_defense", {})
        weekly_wolf = {}
    daily_wolf_safe = daily_wolf.get("safe", False)

    multi_dec = result.get("multi_decision", {})
    single_decs = result.get("single_decisions", {})

    def esc(text):
        """HTML转义"""
        if text is None:
            return ""
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 构建HTML
    html = []
    html.append('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>《缠论》股析 - ''' + esc(title_str) + '''</title>
<style>
  :root {
    --c-up: #e74c3c; --c-down: #27ae60;
    --c-bg: #f1f3f5; --c-card: #ffffff; --c-text: #2c3e50; --c-sub: #7f8c8d;
    --c-border: #dee2e6; --c-accent: #2471a3;
    --c-buy: #c0392b; --c-sell: #1e8449; --c-warn: #e67e22;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.10);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", "Helvetica Neue", sans-serif;
    background: linear-gradient(180deg, #eef0f4 0%, #f5f6f9 100%);
    color: var(--c-text); line-height: 1.85; padding: 0; min-height: 100vh;
  }
  .container { max-width: 1060px; margin: 0 auto; padding: 28px 20px 40px; }

  /* ---- 头部 ---- */
  .header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2d45 50%, #1a2744 100%);
    color: #fff; padding: 36px 44px; border-radius: 14px; margin-bottom: 28px;
    box-shadow: 0 6px 24px rgba(13,27,42,0.25);
  }
  .header h1 { font-size: 27px; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px; }
  .header .meta { color: #8899b4; font-size: 14px; }
  .header .meta span { color: #bccfe0; }

  /* ---- 卡片 ---- */
  .card {
    background: var(--c-card); border-radius: 12px; padding: 32px 36px;
    margin-bottom: 22px; box-shadow: var(--shadow-sm);
    border: 1px solid var(--c-border); transition: box-shadow 0.2s;
  }
  .card:hover { box-shadow: var(--shadow-md); }

  /* ---- 决策框 ---- */
  .verdict-box {
    background: linear-gradient(135deg, #fdf2f2 0%, #fff 40%, #fff 100%);
    border-left: 5px solid var(--c-buy);
  }
  .verdict-box h2 { border-bottom: none; margin-bottom: 8px; }
  .verdict-box.danger {
    background: linear-gradient(135deg, #f2fdf2 0%, #fff 40%, #fff 100%);
    border-left-color: var(--c-sell);
  }

  /* ---- 标题 ---- */
  h2 {
    font-size: 21px; color: #1a1a2e; margin-bottom: 14px;
    padding-bottom: 10px; border-bottom: 2px solid #e8ecf1;
    display: flex; align-items: center; gap: 10px;
  }
  h2 .section-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 34px; height: 34px; border-radius: 50%;
    background: var(--c-accent); color: #fff; font-size: 15px; font-weight: 700;
    flex-shrink: 0;
  }
  h3 {
    font-size: 17px; color: #34495e; margin: 22px 0 10px;
    padding-left: 12px; border-left: 3px solid var(--c-accent);
  }

  /* ---- 图表 ---- */
  .chart-wrapper { text-align: center; margin: 24px 0 10px; }
  .chart-wrapper img {
    max-width: 100%; height: auto; border-radius: 8px;
    box-shadow: 0 3px 20px rgba(0,0,0,0.12);
    border: 1px solid #e0e4e8;
  }
  .chart-caption {
    font-size: 13px; color: var(--c-sub); margin-top: 8px;
    text-align: center;
  }

  /* ---- 数据行 ---- */
  .kv-row { display: flex; gap: 28px; flex-wrap: wrap; margin: 10px 0 14px; }
  .kv-item { display: flex; gap: 6px; align-items: baseline; }
  .kv-label { color: var(--c-sub); font-size: 14px; }
  .kv-value { font-weight: 700; font-size: 15px; }
  .kv-value.up { color: var(--c-up); }
  .kv-value.down { color: var(--c-down); }

  /* ---- 三层解释体系 ---- */
  .layer-lesson {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 13px; color: var(--c-accent); background: #eaf2fa;
    padding: 5px 12px; border-radius: 20px; margin-bottom: 8px; font-weight: 600;
  }
  .layer-lesson::before { content: "📖"; font-size: 14px; }
  .layer-tech {
    font-size: 14px; color: #555; margin: 6px 0; padding: 6px 0;
    font-family: "Consolas", "Courier New", monospace;
    background: #f8f9fb; padding: 8px 14px; border-radius: 6px;
  }
  .layer-tech strong { color: #2c3e50; }
  .layer-plain {
    font-size: 15px; margin: 6px 0 16px; padding: 12px 16px;
    background: #fefffc; border-left: 4px solid #bdc3c7;
    border-radius: 0 8px 8px 0; line-height: 1.9;
  }
  .layer-plain::before {
    content: "💡 "; font-size: 13px;
  }

  /* ---- 信号标签 ---- */
  .buy-tag {
    display: inline-block; background: #fdecea; color: var(--c-buy);
    padding: 3px 12px; border-radius: 6px; font-weight: 700;
    font-size: 14px; margin: 3px;
  }
  .sell-tag {
    display: inline-block; background: #eafaf1; color: var(--c-sell);
    padding: 3px 12px; border-radius: 6px; font-weight: 700;
    font-size: 14px; margin: 3px;
  }
  .star { color: #e67e22; font-size: 13px; }

  .signal-item {
    padding: 10px 0; border-bottom: 1px dashed #e8ecf1;
  }
  .signal-item:last-child { border-bottom: none; }
  .signal-item p { margin: 4px 0; font-size: 14px; }

  /* ---- 警告 ---- */
  .alert {
    background: #fef9e7; border-left: 4px solid var(--c-warn);
    padding: 12px 16px; margin: 8px 0; border-radius: 0 8px 8px 0;
    font-size: 14px; line-height: 1.8;
  }
  .danger-banner {
    background: linear-gradient(135deg, #fdecea 0%, #fadbd8 100%);
    color: #922b21; text-align: center; padding: 18px 24px;
    border-radius: 10px; font-weight: 700; font-size: 16px;
    margin-bottom: 24px; border: 1px solid #f5b7b1;
  }

  /* ---- 总结 ---- */
  .summary-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px; margin: 12px 0;
  }
  .summary-card {
    background: #f8f9fb; border-radius: 8px; padding: 16px 18px;
    border: 1px solid #e8ecf1;
  }
  .summary-card .num {
    font-size: 24px; font-weight: 800; color: var(--c-accent);
    display: block; margin-bottom: 4px;
  }
  .summary-card .txt { font-size: 14px; color: #555; line-height: 1.7; }

  .one-liner {
    text-align: center; font-size: 20px; font-weight: 700;
    padding: 24px 20px; color: #1a1a2e;
    background: linear-gradient(135deg, #f8f9fb 0%, #fff 100%);
    border-radius: 10px; margin-top: 16px;
    border: 1px solid #d5dbdb;
  }

  /* ---- 页脚 ---- */
  .footer {
    text-align: center; color: #aab7c4; font-size: 12px;
    padding: 28px 20px 8px; line-height: 2;
  }

  /* ---- 徽章 ---- */
  .badge {
    display: inline-block; padding: 4px 12px; border-radius: 14px;
    font-size: 13px; font-weight: 700;
  }
  .badge-buy { background: #fdecea; color: var(--c-buy); }
  .badge-sell { background: #eafaf1; color: var(--c-sell); }
  .badge-warn { background: #fef9e7; color: var(--c-warn); }
  .badge-info { background: #eaf2fa; color: var(--c-accent); }

  /* ---- 响应式 ---- */
  @media (max-width: 768px) {
    .container { padding: 10px; }
    .header { padding: 24px 20px; }
    .header h1 { font-size: 20px; }
    .card { padding: 20px 16px; }
    h2 { font-size: 18px; }
  }
</style>
</head>
<body>
<div class="container">
''')

    # ---- 头部 ----
    html.append(f'<div class="header">')
    html.append(f'<h1>《缠论》股析：{esc(title_str)}</h1>')
    html.append(f'<div class="meta">')
    html.append(f'分析时间：{esc(result.get("analysis_date", ""))} &nbsp;|&nbsp; 分析级别：{esc(levels_str)}')
    if date_range:
        html.append(f'&nbsp;|&nbsp; 数据区间：{esc(date_range)}')
    html.append(f'</div></div>')

    # ---- 综合决策 ----
    if multi_dec:
        verdict = multi_dec.get("verdict", "?")
        conf = multi_dec.get("confidence", "?")
        conf_cn = {"very_high": "很高", "high": "较高", "medium": "中等", "low": "偏低"}.get(conf, conf)
        cls = "danger" if "空" in str(verdict) or "卖" in str(verdict) else ""
        html.append(f'<div class="card verdict-box {cls}">')
        html.append(f'<h2><span class="section-num">决</span>综合决策：{esc(verdict)} &nbsp; <span class="badge badge-info">可靠性：{conf_cn}</span></h2>')
        if multi_dec.get("reasoning"):
            for r_line in multi_dec["reasoning"]:
                html.append(f'<p>{esc(r_line)}</p>')
        html.append(f'</div>')
        if multi_dec.get("actions"):
            html.append(f'<div class="card"><h2><span class="section-num">策</span>操作建议</h2><ol>')
            for act in multi_dec.get("actions", []):
                html.append(f'<li>{esc(act)}</li>')
            html.append(f'</ol></div>')
    elif single_decs:
        for level_key, dec in single_decs.items():
            v = dec.get("verdict", "?")
            c = dec.get("confidence", "?")
            cls = "danger" if "空" in str(v) or "卖" in str(v) else ""
            html.append(f'<div class="card verdict-box {cls}">')
            html.append(f'<h2><span class="section-num">决</span>{esc(level_key)}决策：【{esc(v)}】 <span class="badge badge-info">可靠性：{esc(c)}</span></h2>')
            if dec.get("summary"):
                html.append(f'<p>{esc(dec["summary"])}</p>')
            html.append(f'</div>')

    # ---- 防狼术区域放概览图 ----
    overview_chart = chart_images.get("daily") or chart_images.get(list(chart_images.keys())[0] if chart_images else "")
    is_danger = (multi_wolf or {}).get("verdict", "") == "禁止操作" or daily_wolf.get("danger", False)

    # ---- 一、防狼术 ----
    html.append(f'<div class="card">')
    html.append(f'<h2><span class="section-num">一</span>防狼术 —— 能不能做这只票的底线判断</h2>')
    html.append(f'<span class="layer-lesson">《教你炒股票》第103课「防狼术」</span>')
    html.append(f'<p style="color:var(--sub);margin:8px 0">核心原理：MACD黄白线（DIFF/DEA）在0轴之下代表空头主导，任何买点都不参与</p>')

    diff_val = daily_wolf.get("diff", 0)
    dea_val = daily_wolf.get("dea", 0)
    html.append(f'<div class="kv-row">')
    dc = "down" if diff_val < 0 else "up"
    html.append(f'<div class="kv-item"><span class="kv-label">DIFF:</span><span class="kv-value {dc}">{diff_val:.4f}</span></div>')
    dc2 = "down" if dea_val < 0 else "up"
    html.append(f'<div class="kv-item"><span class="kv-label">DEA:</span><span class="kv-value {dc2}">{dea_val:.4f}</span></div>')
    html.append(f'</div>')

    if daily_wolf.get("safe"):
        html.append(f'<p class="layer-plain"><strong>判断：</strong>DIFF > 0 且 DEA > 0 → 在安全区，符合操作前提。趋势指标在多方区域，这只票过了第一关，可以继续往下分析。</p>')
    elif daily_wolf.get("warning"):
        html.append(f'<p class="layer-plain"><strong>判断：</strong>DIFF 和 DEA 在0轴附近分叉 → 临界区，方向待选择。趋势指标在纠结，只能轻仓试探。</p>')
    elif daily_wolf.get("danger"):
        html.append(f'<p class="layer-plain"><strong>判断：</strong>DIFF < 0 且 DEA < 0 → 在空头区，严令禁止操作。趋势指标在空方区域——像有狼群出没，别进去！</p>')

    if weekly_wolf:
        if weekly_wolf.get("safe"):
            html.append(f'<p class="layer-plain">周线交叉验证：周线DIFF也在0轴之上 → 大级别也是多头，安全边际加倍</p>')
        elif weekly_wolf.get("danger"):
            html.append(f'<p class="layer-plain">周线交叉验证：周线DIFF在0轴之下！大级别还是空头，需要降低预期</p>')
    html.append(f'</div>')

    if is_danger:
        html.append(f'<div class="danger-banner">防狼术结论：基础条件不满足，这只票暂时不能碰。等MACD黄白线重新站上0轴再来看。</div>')
        # 即使防狼术未过，仍展示K线图作为视觉凭证
        if overview_chart:
            html.append(f'<div class="card">')
            html.append(f'<h2><span class="section-num">图</span>K线全景图（当前处于防狼术空头区）</h2>')
            html.append(f'<div class="chart-wrapper">')
            html.append(f'<img src="data:image/png;base64,{overview_chart}" alt="缠论K线图">')
            html.append(f'</div></div>')

    # ---- 概览K线图（多级别时在防狼术后展示，单级别在分析区内展示） ----
    levels = result.get("levels_analyzed", ["daily"])
    if overview_chart and not is_danger and len(levels) > 1:
        html.append(f'<div class="card">')
        html.append(f'<h2><span class="section-num">图</span>K线全景图（缠论元素叠加）</h2>')
        html.append(f'<div class="chart-wrapper">')
        html.append(f'<img src="data:image/png;base64,{overview_chart}" alt="缠论K线图">')
        html.append(f'</div></div>')

    # ---- 各周期详细分析（即使防狼术未过也展示完整结构） ----
    if True:  # 始终展示各级别分析详情
        section_num = 1
        for level_key in result.get("levels_analyzed", ["daily"]):
            r = tf_results.get(level_key, {})
            if not r:
                continue

            section_num += 1
            section_titles = {2: "二", 3: "三", 4: "四", 5: "五", 6: "六"}
            sec_label = section_titles.get(section_num, str(section_num))

            html.append(f'<div class="card">')
            html.append(f'<h2><span class="section-num">{sec_label}</span>{esc(_period_cn(level_key))}级别详细分析</h2>')
            html.append(f'<div class="kv-row">')
            html.append(f'<div class="kv-item"><span class="kv-label">最新收盘价：</span><span class="kv-value">{r.get("last_price", 0):.2f}</span></div>')
            html.append(f'<div class="kv-item"><span class="kv-label">K线数量：</span><span class="kv-value">{r.get("kline_count", 0)} 根</span></div>')
            html.append(f'</div>')

            # ---- 本级别K线图 ----
            level_chart = chart_images.get(level_key, "")
            if level_chart:
                html.append(f'<div class="chart-wrapper">')
                html.append(f'<img src="data:image/png;base64,{level_chart}" alt="{level_key}K线图">')
                html.append(f'</div>')

            # 走势类型
            zoushi = r.get("zoushi", {})
            zt = zoushi.get("type", "unknown")
            html.append(f'<h3>走势类型判断</h3>')
            html.append(f'<span class="layer-lesson">{esc(_zoushi_lesson(zt))}</span>')
            html.append(f'<p class="layer-tech">走势类型 = <strong>{esc(zt)}</strong></p>')
            html.append(f'<p class="layer-plain">{esc(_translate_zoushi_type(zt))}</p>')

            # 中枢
            if zoushi.get("latest_zs"):
                zs = zoushi["latest_zs"]
                html.append(f'<h3>中枢分析（主力成本区）</h3>')
                html.append(f'<span class="layer-lesson">第17/18/20课：中枢=至少三段次级别走势重叠区间，是多空博弈的核心地带</span>')
                html.append(f'<p class="layer-tech">ZG={zs["zg"]:.2f} &nbsp; ZD={zs["zd"]:.2f} &nbsp; ZZ={zs.get("zz", 0):.2f}</p>')
                fate = zs.get("fate", "")
                html.append(f'<p class="layer-plain">近期的密集成交区在 [{zs["zd"]:.2f} ~ {zs["zg"]:.2f}]，是"大资金打架"的战场。中枢状态：{esc(fate)} → {esc(_translate_zs_fate(fate))}</p>')

            # 背驰
            divergence = r.get("divergence", {})
            html.append(f'<h3>力量判断（有没有力竭）</h3>')
            div_text, div_lesson = _translate_divergence(divergence)
            if div_lesson:
                html.append(f'<span class="layer-lesson">{esc(div_lesson)}</span>')
            if divergence.get("has_divergence"):
                d = divergence.get("details", {})
                html.append(f'<p class="layer-tech">类型={esc(divergence["type"])} &nbsp; 方向={esc(d.get("direction",""))} &nbsp; 可靠性={esc(divergence.get("confidence",""))}</p>')
                if divergence.get("conditions_check"):
                    html.append(f'<p>6条件检查：</p><ul>')
                    for cond_name, cond_result, cond_note in divergence["conditions_check"]:
                        mark = "&#10003;" if cond_result else "&#10007;"
                        color = "green" if cond_result else "red"
                        html.append(f'<li style="color:{color}">{mark} {esc(cond_name)}: {esc(cond_note)}</li>')
                    html.append(f'</ul>')
            else:
                html.append(f'<p class="layer-tech">未检测到背驰</p>')
            html.append(f'<p class="layer-plain">{esc(div_text)}</p>')

            # 买卖点
            tp = r.get("trade_points", {})
            html.append(f'<h3>买卖点（什么时候动手）</h3>')
            html.append(f'<span class="layer-lesson">第20/21/53/101课：三类买卖点分别是趋势背驰终点、回踩确认点、突破回抽点</span>')
            if tp.get("buy_points"):
                for bp in tp["buy_points"]:
                    conf = bp.get("confidence", "")
                    star = "★★★" if conf in ("high", "very_high") else ("★★" if conf == "medium" else "★")
                    plain, lesson = _translate_trade_point(bp["type"])
                    html.append(f'<div class="signal-item">')
                    html.append(f'<span class="buy-tag">{esc(bp["type"])}</span> <span class="star">{star}</span> 可靠性={esc(conf)}')
                    html.append(f'<p style="margin:4px 0;font-size:14px">信号：{esc(bp.get("description",""))}</p>')
                    html.append(f'<p class="layer-plain">{esc(plain)}</p>')
                    if bp.get("quality"):
                        html.append(f'<p style="font-size:13px;color:var(--sub)">质量：{esc(bp["quality"])}</p>')
                    html.append(f'</div>')
            if tp.get("sell_points"):
                for sp in tp["sell_points"]:
                    conf = sp.get("confidence", "")
                    star = "★★★" if conf in ("high", "very_high") else ("★★" if conf == "medium" else "★")
                    plain, lesson = _translate_trade_point(sp["type"])
                    html.append(f'<div class="signal-item">')
                    html.append(f'<span class="sell-tag">{esc(sp["type"])}</span> <span class="star">{star}</span> 可靠性={esc(conf)}')
                    html.append(f'<p style="margin:4px 0;font-size:14px">信号：{esc(sp.get("description",""))}</p>')
                    html.append(f'<p class="layer-plain">{esc(plain)}</p>')
                    html.append(f'</div>')
            if not tp.get("buy_points") and not tp.get("sell_points"):
                html.append(f'<p class="layer-plain">暂无标准买卖点——没信号的时候不动手，强行操作等于赌博</p>')

            # 表里关系
            dual = r.get("dual_state", {})
            state = dual.get("state", "?")
            plain, lesson = _translate_dual_state(state)
            html.append(f'<h3>当前状态（表里关系）</h3>')
            html.append(f'<span class="layer-lesson">{esc(lesson)}</span>')
            html.append(f'<p class="layer-tech">当前状态 = <strong>{esc(state)}</strong></p>')
            html.append(f'<p class="layer-plain">{esc(plain)}</p>')

            # 中枢监视器
            zs_mon = r.get("zhongshu_monitor", {})
            if zs_mon.get("active"):
                pos = zs_mon.get("position", "")
                pos_cn = {
                    "above_zs": "价格在主力成本区上方——多头占优，像站在楼上往下看",
                    "below_zs": "价格在主力成本区下方——空头占优，像掉到楼下了",
                    "inside_zs": "价格在主力成本区里面——多空拉锯战，方向不明确",
                }.get(pos, pos)
                html.append(f'<h3>中枢位置判断（第92课：中枢震荡监视器）</h3>')
                html.append(f'<p class="layer-plain">{esc(pos_cn)}</p>')
                if zs_mon.get("alerts"):
                    for a in zs_mon["alerts"]:
                        html.append(f'<div class="alert">{esc(a.get("desc", ""))}</div>')

            # 中阴
            bardo = r.get("bardo", {})
            if bardo.get("active"):
                html.append(f'<h3>中阴阶段（第88-90课）</h3>')
                html.append(f'<p class="layer-plain">走势因背驰而"死亡"，当前处于方向选择的过渡期——像车到了十字路口</p>')
                html.append(f'<p>阶段：{esc(bardo.get("description",""))} &nbsp;|&nbsp; 策略：{esc(bardo.get("action",""))}</p>')
                if bardo.get("health_note"):
                    html.append(f'<p>健康度：{esc(bardo["health_note"])}</p>')

            # 底部构造
            bottom = r.get("bottom_construction", {})
            if bottom.get("construction_phase") and bottom["construction_phase"] != "unknown":
                html.append(f'<h3>底部构造（第108课）</h3>')
                html.append(f'<p class="layer-tech">{esc(bottom["construction_phase"])}</p>')
                html.append(f'<p class="layer-plain">{esc(bottom.get("action", ""))}</p>')

            html.append(f'</div>')  # end card

    # ---- 白话总结（即使防狼术未过也展示，但加注警示） ----
    if True:  # 始终展示总结
        html.append(f'<div class="card">')
        if is_danger:
            html.append(f'<div class="alert"><strong>!! 注意：以下总结仅用于了解走势结构，防狼术未通过，不宜操作 !!</strong></div>')
        html.append(f'<h2><span class="section-num">总</span>白话总结（说人话版）</h2>')

        html.append(f'<div class="summary-grid">')

        zt = daily_r.get("zoushi", {}).get("type", "")
        zoushi_plain = {
            "uptrend": "中长期处于上升通道，大方向由多头掌控",
            "downtrend": "中长期处于下降通道，空头占主导",
            "consolidation": "处于震荡格局，没有明确趋势方向",
        }.get(zt, "走势结构不清晰")
        html.append(f'<div class="summary-card"><span class="num">①</span><span class="txt">大方向：{esc(zoushi_plain)}</span></div>')

        state = daily_r.get("dual_state", {}).get("state", "")
        state_plain = {
            "(1,1)": "当前在上涨途中，持股的人继续拿，没买的等回调别追高",
            "(-1,1)": "当前在下跌途中，持币观望，等跌透了再说",
            "(1,0)": "上涨中出现了停顿信号（顶分型），要提高警惕",
            "(-1,0)": "下跌中出现了停顿信号（底分型），机会可能在酝酿",
        }.get(state, "状态不明")
        html.append(f'<div class="summary-card"><span class="num">②</span><span class="txt">当前姿态：{esc(state_plain)}</span></div>')

        div = daily_r.get("divergence", {})
        if div.get("has_divergence"):
            dd = div.get("details", {})
            if dd.get("direction") == "bottom_divergence":
                html.append(f'<div class="summary-card"><span class="num">③</span><span class="txt">下跌动能已经在衰减——卖的人没力气了，底部可能不远</span></div>')
            elif dd.get("direction") == "top_divergence":
                html.append(f'<div class="summary-card"><span class="num">③</span><span class="txt">上涨动能已经在衰减——买的人没力气了，顶部风险在增加</span></div>')
        else:
            html.append(f'<div class="summary-card"><span class="num">③</span><span class="txt">多空力量双方势均力敌，没有哪一方出现明显的力竭</span></div>')

        tp = daily_r.get("trade_points", {})
        if tp.get("buy_points"):
            names = [bp["type"] for bp in tp["buy_points"]]
            html.append(f'<div class="summary-card"><span class="num">④</span><span class="txt">买点：出现了 {esc("、".join(names))}——可以考虑按纪律入场</span></div>')
        elif tp.get("sell_points"):
            names = [sp["type"] for sp in tp["sell_points"]]
            html.append(f'<div class="summary-card"><span class="num">④</span><span class="txt">卖点：出现了 {esc("、".join(names))}——建议考虑兑现或减仓</span></div>')
        else:
            html.append(f'<div class="summary-card"><span class="num">④</span><span class="txt">操作信号：暂无——这种时候不动手就是最好的操作</span></div>')

        zs_pos = daily_r.get("zhongshu_monitor", {}).get("position", "")
        pos_plain = {
            "above_zs": "价格站在主力成本区上方，多头占优",
            "below_zs": "价格在主力成本区下方，上面有套牢盘压着",
            "inside_zs": "价格在主力成本区里面，多空还在争夺",
        }.get(zs_pos, "位置不明")
        html.append(f'<div class="summary-card"><span class="num">⑤</span><span class="txt">相对位置：{esc(pos_plain)}</span></div>')

        html.append(f'</div>')

        # 一句话
        st = daily_r.get("dual_state", {}).get("state", "")
        if st == "(1,1)" and not tp.get("sell_points"):
            oneliner = "趋势向好，继续持有或等回调加仓"
        elif st == "(-1,0)" and div.get("has_divergence"):
            oneliner = "机会在酝酿，密切关注底分型确认信号"
        elif st == "(-1,1)":
            oneliner = "下跌还没完，持币观望别手痒"
        elif st == "(1,0)" and tp.get("sell_points"):
            oneliner = "上涨遇阻了，可以考虑分批兑现利润"
        elif tp.get("buy_points"):
            oneliner = "买点已现，按纪律入场，设好止损位"
        elif tp.get("sell_points"):
            oneliner = "卖点已现，按纪律离场，别贪最后一口"
        else:
            oneliner = "信号不明确的时候，不动就是赚"
        html.append(f'<div class="one-liner">{esc(oneliner)}</div>')
        html.append(f'</div>')

    # ---- 页脚 ----
    html.append(f'<div class="footer">')
    html.append(f'本报告基于缠中说禅《教你炒股票108课》理论体系自动生成 | 仅供参考，不构成投资建议<br>')
    html.append(f'生成时间：{esc(result.get("analysis_date", ""))}')
    html.append(f'</div>')

    html.append('</div></body></html>')

    full_html = "\n".join(html)

    # 保存
    os.makedirs(HTML_REPORT_DIR, exist_ok=True)
    sd = date_range[:10].replace("-", "") if date_range else datetime.now().strftime("%Y%m%d")
    fname = f"{result['symbol']}_{result['name']}_{sd}.html"
    fpath = os.path.join(HTML_REPORT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(full_html)

    return fpath


# ============================================================
# 第四部分：批量扫描
# ============================================================

def analyze_stock(symbol: str, name: str = "", quiet: bool = False,
                  start_date: str = "20200101", end_date: str = None) -> dict:
    """
    对一只股票进行完整缠论分析（简化入口，主要用于批量扫描）
    由于批量扫描数据量大，仅用日线周期
    """
    if not quiet:
        print(f"\n  [分析] {symbol} {name}")

    try:
        df = fetch_kline(symbol, period="daily", start_date=start_date,
                         end_date=end_date)
        df.attrs["period"] = "daily"
        if len(df) < 60:
            if not quiet:
                print(f"    数据不足")
            return {}

        if not quiet:
            print(f"    获取 {len(df)} 条日线数据")

        result = analyze_single_timeframe(df, name=name, quiet=quiet)
        result["code"] = symbol  # 保留代码，供周线交叉验证使用
        result["_raw_df"] = df   # 保留原始数据，供图表生成使用
        return result

    except Exception as e:
        if not quiet:
            print(f"    分析失败: {e}")
        return {}


def scan_watchlist(stocks: List[Tuple[str, str]],
                   delay: float = 2.0, start_date: str = "20200101",
                   end_date: str = None) -> List[dict]:
    """批量扫描自选股池（日线级别 + 周线交叉验证）"""
    results = []
    for i, (symbol, name) in enumerate(stocks):
        if i > 0:
            time.sleep(delay)
        try:
            result = analyze_stock(symbol, name, quiet=True, start_date=start_date,
                                   end_date=end_date)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  {symbol} {name} 分析失败: {e}")

    # 周线交叉验证
    from chan_lib.strategy import quick_weekly_check
    for r in results:
        if r.get("wolf_defense", {}).get("safe"):
            symbol = r.get("code", "")
            if symbol:
                weekly = quick_weekly_check(symbol)
                r["weekly_check"] = weekly

    return results


def scan_csi300(max_stocks: int = None, delay: float = 1.5,
                start_date: str = "20200101", end_date: str = None) -> List[dict]:
    """扫描沪深300成分股（日线级别）"""
    stocks = fetch_csi300_stocks()
    total = len(stocks)
    if max_stocks:
        stocks = stocks[:max_stocks]
        total = len(stocks)

    all_csi300 = fetch_csi300_stocks()
    print(f"\n  沪深300成分股共 {len(all_csi300)} 只，本次扫描 {total} 只\n")

    results = []
    ok_count = 0
    fail_count = 0

    for i, (symbol, name) in enumerate(stocks):
        if i > 0:
            time.sleep(delay)
        pct = (i + 1) / total * 100
        print(f"\r  扫描进度: {i+1}/{total} ({pct:.0f}%)  "
              f"通过: {ok_count}  失败: {fail_count}", end="")

        try:
            result = analyze_stock(symbol, name, quiet=True, start_date=start_date,
                                   end_date=end_date)
            if result:
                results.append(result)
                if result.get("wolf_defense", {}).get("safe"):
                    ok_count += 1
        except Exception:
            fail_count += 1
            continue

    print(f"\n  扫描完成: {len(results)} 只成功, {fail_count} 只失败")

    # 排序
    def sort_key(r):
        score = 0
        tp = r.get("trade_points", {})
        if tp.get("buy_points"):
            for bp in tp["buy_points"]:
                if bp["type"] == "三买":
                    score += 30
                elif bp["type"] == "二买":
                    score += 20
                elif bp["type"] == "一买":
                    score += 10
                if bp.get("confidence") in ("high", "very_high"):
                    score += 5

        wd = r.get("wolf_defense", {})
        if wd.get("safe"):
            score += 50
        elif wd.get("warning"):
            score += 20

        zs_mon = r.get("zhongshu_monitor", {})
        pos = zs_mon.get("position", "")
        if pos == "above_zs":
            score += 5
        elif pos == "inside_zs":
            score += 3

        div = r.get("divergence", {})
        if div.get("has_divergence") and div.get("type") == "trend_divergence":
            direction = div.get("details", {}).get("direction", "")
            if direction == "bottom_divergence":
                score += 15

        return -score

    results.sort(key=sort_key)

    # === 周线交叉验证：对日线评分靠前的股票补查周线，过滤大级别空头 ===
    print(f"\n  周线交叉验证中...")
    from chan_lib.strategy import quick_weekly_check
    weekly_risk_flags = []
    for r in results:
        if r.get("wolf_defense", {}).get("safe"):
            symbol = r.get("code", "")
            if symbol:
                weekly = quick_weekly_check(symbol)
                r["weekly_check"] = weekly
                if weekly.get("warning"):
                    weekly_risk_flags.append(r.get("symbol", symbol))

    if weekly_risk_flags:
        print(f"  !! 周线空头警告: {', '.join(weekly_risk_flags)}")
        # 对周线警告的股票降权重排
        def sort_key_with_weekly(r2):
            base = -sort_key(r2)
            weekly = r2.get("weekly_check", {})
            if weekly.get("warning"):
                base -= 30  # 大幅降分，放到后面
            return -base
        results.sort(key=sort_key_with_weekly)
    # ===============================================================

    return results


# ============================================================
# 第五部分：报告打印（批量模式）
# ============================================================

def print_scan_summary(results: List[dict]):
    """打印自选池扫描总结"""
    print(f"\n{'='*70}")
    print(f"  自选池缠论扫描总览")
    print(f"{'='*70}")

    safe_stocks = []
    danger_stocks = []
    buy_stocks = []
    sell_stocks = []

    for r in results:
        if not r:
            continue
        wd = r.get("wolf_defense", {})
        tp = r.get("trade_points", {})

        if wd.get("safe"):
            safe_stocks.append(r)
        if wd.get("danger"):
            danger_stocks.append(r)
        if tp.get("buy_points"):
            buy_stocks.append(r)
        if tp.get("sell_points"):
            sell_stocks.append(r)

    print(f"\n  防狼术通过（0轴之上）: {len(safe_stocks)} 只")
    for r in safe_stocks:
        dual = r.get("dual_state", {})
        state_icon = {"(1,1)": "+", "(1,0)": "↓?", "(-1,0)": "↑?", "(-1,1)": "-"}.get(
            dual.get("state", ""), "?")
        print(f"     [{state_icon}] {r.get('symbol', '')} {r.get('name', '')}  "
              f"{r.get('last_price', 0):.2f}  {dual.get('state', '')}")

    if danger_stocks:
        print(f"\n  防狼术警告（0轴之下）: {len(danger_stocks)} 只")
        for r in danger_stocks:
            print(f"     [XX] {r.get('symbol', '')} {r.get('name', '')}  "
                  f"{r.get('last_price', 0):.2f}  -- 远离！")

    if buy_stocks:
        print(f"\n  有买点信号的: {len(buy_stocks)} 只")
        for r in buy_stocks:
            tp = r.get("trade_points", {})
            bp_types = ", ".join(
                f"{p['type']}({p.get('confidence', '')})"
                for p in tp.get("buy_points", [])
            )
            weekly_note = ""
            wc = r.get("weekly_check", {})
            if wc.get("warning"):
                weekly_note = f"  [!!周线空头: {wc.get('reason', '')}]"
            print(f"     [BUY] {r.get('symbol', '')} {r.get('name', '')} → {bp_types}{weekly_note}")

    if sell_stocks:
        print(f"\n  有卖点信号的: {len(sell_stocks)} 只")
        for r in sell_stocks:
            tp = r.get("trade_points", {})
            sp_types = ", ".join(
                f"{p['type']}({p.get('confidence', '')})"
                for p in tp.get("sell_points", [])
            )
            print(f"     [SELL] {r.get('symbol', '')} {r.get('name', '')} → {sp_types}")

    print(f"\n{'='*70}\n")


def print_csi300_summary(results: List[dict], top_n: int = 30):
    """打印沪深300筛选结果（分层排序）"""
    print(f"\n{'='*70}")
    print(f"  沪深300 缠论筛选结果")
    print(f"{'='*70}")

    tier1 = []  # 0轴之上 + 有买点
    tier2 = []  # 0轴之上 + 无买点
    tier3 = []  # 临界 + 有信号
    tier4 = []  # 其余0轴之上

    for r in results:
        wd = r.get("wolf_defense", {})
        tp = r.get("trade_points", {})
        has_buy = bool(tp.get("buy_points"))

        if wd.get("safe") and has_buy:
            tier1.append(r)
        elif wd.get("safe") and not has_buy:
            tier2.append(r)
        elif wd.get("warning") and has_buy:
            tier3.append(r)
        elif wd.get("safe"):
            tier4.append(r)

    def print_tier(title, stocks, top=30):
        if not stocks:
            return
        print(f"\n  [{title}] ({len(stocks)} 只)")
        print(f"  {'代码':<8} {'名称':<10} {'价格':>8} {'DIFF':>8}  "
              f"{'中枢位置':<8} {'信号':<18} {'状态':<6}")
        print(f"  {'-'*65}")
        for r in stocks[:top]:
            symbol = r.get("symbol", "")
            name = r.get("name", "")[:8]
            price = r.get("last_price", 0)
            diff = r.get("wolf_defense", {}).get("diff", 0)
            pos = r.get("zhongshu_monitor", {}).get("position", "—")
            tp = r.get("trade_points", {})

            signals = []
            for bp in tp.get("buy_points", []):
                conf = bp.get("confidence", "")
                star = "★" if conf in ("high", "very_high") else ""
                signals.append(f"{bp['type']}{star}")
            sig_str = ", ".join(signals) if signals else "—"

            dual = r.get("dual_state", {})
            state_short = {"(1,1)": "+↑", "(-1,1)": "-↓",
                           "(1,0)": "顶?", "(-1,0)": "底?"}.get(dual.get("state", ""), "")

            weekly_warn = ""
            wc = r.get("weekly_check", {})
            if wc.get("warning"):
                weekly_warn = " !!周线空头"

            print(f"  {symbol:<8} {name:<10} {price:>8.2f} {diff:>8.4f}  "
                  f"{pos:<8} {sig_str:<18} {state_short}{weekly_warn}")

    print_tier("第一梯队 -- 0轴之上 + 有买点信号 [可直接候选]", tier1, top_n)
    print_tier("第二梯队 — 0轴之上 + 等待信号", tier2, top_n)
    print_tier("第三梯队 — 临界0轴 + 有信号（观察名单）", tier3, top_n)

    total_safe = len([r for r in results if r.get("wolf_defense", {}).get("safe")])
    total_danger = len([r for r in results if r.get("wolf_defense", {}).get("danger")])
    total_buy = len([r for r in results if r.get("trade_points", {}).get("buy_points")])

    print(f"\n  {'─'*65}")
    print(f"  总计: {len(results)} 只分析成功")
    print(f"  防狼术通过（0轴之上）: {total_safe} 只 "
          f"({total_safe / max(len(results), 1) * 100:.0f}%)")
    print(f"  防狼术警告（0轴之下）: {total_danger} 只")
    print(f"  有买点信号: {total_buy} 只")
    if tier1:
        print(f"  可直接候选（第一梯队）: {len(tier1)} 只")
    weekly_warned = [r for r in results if r.get("weekly_check", {}).get("warning")]
    if weekly_warned:
        names = [r.get("symbol", "") for r in weekly_warned]
        print(f"  !! 周线空头警告（日线好看但周线还在向下）: {len(weekly_warned)} 只 — {', '.join(names)}")
    print(f"{'='*70}\n")


# ============================================================
# 第六部分：主入口
# ============================================================

DEFAULT_WATCHLIST = [
    # === 第一梯队：准备买入信号 ===
    ("601689", "拓普集团"),
    ("688082", "盛美上海"),
    # === 第二梯队：等待买入时机 ===
    ("600522", "中天科技"),
    ("605117", "德业股份"),
    ("688047", "龙芯中科"),
    # === 第三梯队：已持有继续拿 ===
    ("600584", "长电科技"),
    # === 原有自选 ===
    ("000001", "平安银行"),
    ("600519", "贵州茅台"),
    ("000858", "五粮液"),
    ("601318", "中国平安"),
    ("300750", "宁德时代"),
    ("600036", "招商银行"),
    ("000333", "美的集团"),
    ("600276", "恒瑞医药"),
]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 解析可选参数
        args = sys.argv[1:]
        start_date = "20200101"
        end_date = None
        use_monthly = False
        show_chart = False

        i = 0
        positional = []
        while i < len(args):
            a = args[i]
            if a == "--start" and i + 1 < len(args):
                start_date = args[i + 1]
                i += 2
            elif a == "--end" and i + 1 < len(args):
                end_date = args[i + 1]
                i += 2
            elif a == "--monthly":
                use_monthly = True
                i += 1
            elif a == "--chart":
                show_chart = True
                i += 1
            else:
                positional.append(a)
                i += 1

        if not positional:
            # 只有可选参数，无位置参数 → 扫自选池
            results = scan_watchlist(DEFAULT_WATCHLIST, start_date=start_date,
                                     end_date=end_date)
            print_scan_summary(results)
        else:
            arg = positional[0]

            if arg == "--csi300":
                max_n = int(positional[1]) if len(positional) > 1 else None
                results = scan_csi300(max_stocks=max_n, start_date=start_date,
                                      end_date=end_date)
                print_csi300_summary(results)

            elif arg == "--watchlist":
                results = scan_watchlist(DEFAULT_WATCHLIST, start_date=start_date,
                                         end_date=end_date)
                print_scan_summary(results)

            elif arg == "--full":
                # 多级别联立完整分析
                code = positional[1] if len(positional) > 1 else "600519"
                name = positional[2] if len(positional) > 2 else ""
                periods = ["30", "daily", "weekly", "monthly"]
                result = analyze_multi_timeframe(code, name, periods=periods,
                                                  start_date=start_date,
                                                  end_date=end_date)
                print_full_report(result)

                # 生成K线图 + HTML报告（始终生成）
                if result.get("timeframe_results"):
                    from chan_lib.visual import chart_to_base64
                    import chan_lib.data as cdata
                    chart_images = {}
                    for level_key, r in result["timeframe_results"].items():
                        df = r.get("_raw_df")
                        if df is None:
                            try:
                                period_map = {"30min": "30", "60min": "60", "daily": "daily",
                                             "weekly": "weekly", "monthly": "monthly"}
                                p = period_map.get(level_key, "daily")
                                df = cdata.fetch_kline(code, period=p, start_date=start_date,
                                                       end_date=end_date)
                            except Exception:
                                continue
                        try:
                            b64 = chart_to_base64(df,
                                bi_list=r.get("bi_list", []),
                                segment_list=r.get("segment_list", []),
                                zhongshu_list=r.get("zhongshu_list", []),
                                trade_points=r.get("trade_points", {}),
                                symbol=code, name=name, period=level_key,
                                start_date=start_date, end_date=end_date,
                                max_bars=300)
                            if b64:
                                chart_images[level_key] = b64
                        except Exception:
                            import traceback
                            print(f"  [图表] {level_key}: 生成异常 - {traceback.format_exc()}")
                            continue

                    if chart_images:
                        html_path = generate_html_report(result, chart_images)
                        print(f"\n  HTML报告已生成: {html_path}")
                        webbrowser.open(f"file:///{html_path}")

            else:
                # 单只股票（默认日线级别）
                code = arg
                name = positional[1] if len(positional) > 1 else ""
                result = analyze_stock(code, name, start_date=start_date,
                                       end_date=end_date)
                if result:
                    # 综合交易决策
                    from chan_lib.strategy import trading_decision
                    dec = trading_decision(result)
                    df = result.get("_raw_df")
                    # 包装成多级别格式以复用打印
                    wrapped = {
                        "symbol": code, "name": name,
                        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "levels_analyzed": ["daily"],
                        "timeframe_results": {"daily": result},
                        "single_decisions": {"daily": dec},
                    }
                    print_full_report(wrapped)

                    # 生成K线图 + HTML报告（始终生成）
                    if df is not None:
                        from chan_lib.visual import chart_to_base64
                        try:
                            b64 = chart_to_base64(df,
                                bi_list=result.get("bi_list", []),
                                segment_list=result.get("segment_list", []),
                                zhongshu_list=result.get("zhongshu_list", []),
                                trade_points=result.get("trade_points", {}),
                                symbol=code, name=name, period="daily",
                                start_date=start_date, end_date=end_date,
                                max_bars=300)
                            if b64:
                                html_path = generate_html_report(wrapped, {"daily": b64})
                                print(f"\n  HTML报告已生成: {html_path}")
                                webbrowser.open(f"file:///{html_path}")
                        except Exception:
                            import traceback
                            print(f"  [图表] daily: 生成异常 - {traceback.format_exc()}")
    else:
        # 默认：扫自选池
        results = scan_watchlist(DEFAULT_WATCHLIST)
        print_scan_summary(results)
