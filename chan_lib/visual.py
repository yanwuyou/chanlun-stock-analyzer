#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缠论K线图可视化模块
用 mplfinance 画K线，叠加缠论元素（笔→线段→中枢→买卖点）+ MACD副图
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无GUI后端，仅用于保存文件，避免PyInstaller打包后的后端冲突

# ---- 中文字体配置（必须在 import pyplot 之前设好 rcParams） ----
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "WenQuanYi Micro Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from typing import List, Dict, Optional

import mplfinance as mpf

# ---- 颜色方案（A股红涨绿跌）----
UP_COLOR = "#e74c3c"
DOWN_COLOR = "#27ae60"
BI_UP_COLOR = "#ff6b6b"
BI_DOWN_COLOR = "#51cf66"
SEG_UP_COLOR = "#c0392b"
SEG_DOWN_COLOR = "#1e8449"
ZS_COLORS = ["#3498db", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22", "#e74c3c"]
ZS_ALPHA = 0.22
BUY_COLOR = "#ff0000"
SELL_COLOR = "#00aa00"

CHART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "charts")

PERIOD_CN = {
    "daily": "日线", "weekly": "周线", "monthly": "月线",
    "30min": "30分钟", "60min": "60分钟", "5min": "5分钟", "1min": "1分钟",
    "30": "30分钟", "60": "60分钟",  # 兼容 fetch_multi_timeframe 的短键名
}


def _period_cn(key: str) -> str:
    return PERIOD_CN.get(key, key)


def plot_chan_chart(df: pd.DataFrame,
                    bi_list: List[dict] = None,
                    segment_list: List[dict] = None,
                    zhongshu_list: List[dict] = None,
                    trade_points: dict = None,
                    symbol: str = "",
                    name: str = "",
                    period: str = "daily",
                    start_date: str = None,
                    end_date: str = None,
                    max_bars: int = 300,
                    save_path: str = None) -> Optional[str]:
    """
    绘制缠论完整K线图（K线 + 笔 + 线段 + 中枢 + 买卖点 + MACD + 成交量）

    Returns:
        保存路径，失败返回 None
    """
    if bi_list is None:
        bi_list = []
    if segment_list is None:
        segment_list = []
    if zhongshu_list is None:
        zhongshu_list = []
    if trade_points is None:
        trade_points = {}

    # ---- 裁剪区间 ----
    plot_df = df.copy()
    if "date" not in plot_df.columns:
        return None
    if not isinstance(plot_df["date"].iloc[0], pd.Timestamp):
        plot_df["date"] = pd.to_datetime(plot_df["date"])

    if start_date:
        plot_df = plot_df[plot_df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        plot_df = plot_df[plot_df["date"] <= pd.Timestamp(end_date)]

    if len(plot_df) < 5:
        return None

    # ---- 准备 mplfinance 数据 ----
    mpf_df = plot_df.set_index("date")[["open", "high", "low", "close", "volume"]].copy()
    mpf_df.columns = ["Open", "High", "Low", "Close", "Volume"]

    # ---- 数据清洗：纯有效交易行过滤 + 聚焦最近 N 根K线 ----
    mpf_df = mpf_df[~mpf_df.index.duplicated()].sort_index()
    mpf_df = mpf_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if max_bars and len(mpf_df) > max_bars:
        mpf_df = mpf_df.tail(max_bars).copy()
        # 同步裁剪 plot_df，保证 MACD 副图数据对齐
        keep_dates = set(mpf_df.index)
        plot_df = plot_df[plot_df["date"].isin(keep_dates)]

    # MACD addplot
    apds = []
    if "diff" in plot_df.columns and "dea" in plot_df.columns:
        macd_df = plot_df.set_index("date")[["diff", "dea", "macd_hist"]].copy()
        macd_df.columns = ["DIFF", "DEA", "MACD"]
        # MACD柱：正值红色，负值绿色
        macd_pos = macd_df["MACD"].copy()
        macd_neg = macd_df["MACD"].copy()
        macd_pos[macd_pos < 0] = 0
        macd_neg[macd_neg > 0] = 0
        apds.append(mpf.make_addplot(macd_df["DIFF"], panel=2, color="#e67e22", width=1.0, ylabel="DIFF/DEA"))
        apds.append(mpf.make_addplot(macd_df["DEA"], panel=2, color="#3498db", width=1.0))
        apds.append(mpf.make_addplot(macd_pos, type="bar", panel=2, color=UP_COLOR, width=0.7))
        apds.append(mpf.make_addplot(macd_neg, type="bar", panel=2, color=DOWN_COLOR, width=0.7))

    # ---- 风格 ----
    mc = mpf.make_marketcolors(
        up=UP_COLOR, down=DOWN_COLOR,
        edge="inherit", wick="inherit",
        volume={"up": UP_COLOR, "down": DOWN_COLOR},
        alpha=0.85,
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridcolor="#e0e0e0",
        gridstyle="--",
        facecolor="#fafbfc",
        figcolor="white",
        y_on_right=False,
    )
    style["rc"] = {"font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun"],
                   "axes.unicode_minus": False}

    title_str = f"{symbol} {name}  {_period_cn(period)}"
    if start_date or end_date:
        sd = start_date or plot_df["date"].iloc[0].strftime("%Y%m%d")
        ed = end_date or plot_df["date"].iloc[-1].strftime("%Y%m%d")
        title_str += f"  [{sd}-{ed}]"

    # ---- 画图 ----
    n_candles = len(mpf_df)
    # 动态宽度：每根K线约0.07英寸，最少18英寸，最多44英寸
    fig_w = max(18, min(44, n_candles * 0.085))
    fig_h = max(10, fig_w * 0.42)

    fig, axlist = mpf.plot(
        mpf_df,
        type="candle",
        style=style,
        title=title_str,
        volume=True,
        addplot=apds,
        panel_ratios=(4, 1.2, 1.5) if apds else (4, 1.2),
        returnfig=True,
        figsize=(fig_w, fig_h),
        warn_too_much_data=len(mpf_df) + 500,
        show_nontrading=False,
        tight_layout=True,
    )

    ax_main = axlist[0]  # 主图（K线）

    # ---- 叠加缠论元素 ----
    _draw_segments(ax_main, segment_list, mpf_df)
    _draw_bi(ax_main, bi_list, mpf_df)
    _draw_zhongshu(ax_main, zhongshu_list, mpf_df)
    _draw_trade_points(ax_main, trade_points, mpf_df, bi_list)

    # 图例
    legend_elements = []
    if bi_list:
        legend_elements.append(Line2D([0], [0], color=BI_UP_COLOR, lw=1.2, ls="--", label="笔"))
    if segment_list:
        legend_elements.append(Line2D([0], [0], color=SEG_UP_COLOR, lw=1.8, label="线段"))
    if zhongshu_list:
        legend_elements.append(mpatches.Patch(color=ZS_COLORS[0], alpha=ZS_ALPHA, label="中枢"))
    buy_pts = trade_points.get("buy_points", [])
    sell_pts = trade_points.get("sell_points", [])
    if buy_pts:
        legend_elements.append(Line2D([0], [0], marker="^", color="w", markerfacecolor=BUY_COLOR,
                                      markersize=8, label="买点"))
    if sell_pts:
        legend_elements.append(Line2D([0], [0], marker="v", color="w", markerfacecolor=SELL_COLOR,
                                      markersize=8, label="卖点"))
    if legend_elements:
        ax_main.legend(handles=legend_elements, loc="upper left", fontsize=7,
                       framealpha=0.85, ncol=min(len(legend_elements), 4))

    # ---- 轻量轴美化（不破坏 mplfinance 布局） ----
    _finalize_axes(axlist, mpf_df, apds)

    # ---- 保存 ----
    os.makedirs(CHART_DIR, exist_ok=True)
    if save_path is None:
        sd = start_date or plot_df["date"].iloc[0].strftime("%Y%m%d")
        fname = f"{symbol}_{period}_{sd}.png"
        save_path = os.path.join(CHART_DIR, fname)

    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    return save_path


# ============================================================
# 内部绘图函数
# ============================================================

def _in_visible_range(d, mpf_df) -> bool:
    """检查日期是否在图表可见范围内"""
    try:
        d_ts = pd.Timestamp(d)
        return d_ts >= mpf_df.index[0] and d_ts <= mpf_df.index[-1]
    except Exception:
        return False


def _finalize_axes(axlist, mpf_df, apds):
    """轻量轴美化——Y轴价格格式化 + 成交量格式化 + X轴旋转 + MACD零线"""
    ax_main = axlist[0]

    # Y轴：价格格式化
    price_range = mpf_df["High"].max() - mpf_df["Low"].min()
    if price_range > 100:
        price_fmt = "%.0f"
    elif price_range > 10:
        price_fmt = "%.1f"
    else:
        price_fmt = "%.2f"
    ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: price_fmt % v))
    ax_main.tick_params(axis="y", labelsize=8)
    # X轴：旋转刻度避免文字重叠
    ax_main.tick_params(axis="x", rotation=45, labelsize=7)

    # 成交量面板：整数格式化 + 隐藏X轴刻度
    ax_vol = axlist[1]
    ax_vol.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/10000:.0f}万" if v >= 10000 else f"{v:.0f}"))
    ax_vol.tick_params(axis="y", labelsize=7)
    ax_vol.tick_params(axis="x", labelbottom=False)
    ax_vol.set_ylabel("VOL", fontsize=7, color="#888888")

    # MACD 零轴参考线 + 隐藏X轴刻度
    if apds and len(axlist) >= 3:
        ax_macd = axlist[2]
        ax_macd.axhline(y=0, color="#999999", linewidth=0.5, linestyle="-")
        ax_macd.tick_params(axis="y", labelsize=7)
        ax_macd.tick_params(axis="x", labelbottom=False)


def _draw_bi(ax, bi_list: List[dict], mpf_df):
    """画笔：虚线 + 端点三角标记 + 终点价格标注（zorder 在K线上方）"""
    if not bi_list:
        return
    price_range = mpf_df["High"].max() - mpf_df["Low"].min()
    for bi in bi_list:
        try:
            s_date, e_date = bi.get("start_date"), bi.get("end_date")
            if not _in_visible_range(s_date, mpf_df) and not _in_visible_range(e_date, mpf_df):
                continue
            x0, x1 = mdates.date2num(pd.Timestamp(s_date)), mdates.date2num(pd.Timestamp(e_date))
            y0, y1 = bi["start_price"], bi["end_price"]
            color = BI_UP_COLOR if bi["dir"] == "up" else BI_DOWN_COLOR
            ax.plot([x0, x1], [y0, y1], color=color, lw=1.2, ls="--", alpha=0.75, zorder=2)
            # 起点小圆点
            ax.scatter([x0], [y0], s=20, color=color, marker="o", alpha=0.9, zorder=5,
                       edgecolors="white", linewidths=0.6)
            # 终点三角：向上笔▲，向下笔▼
            marker = "^" if bi["dir"] == "up" else "v"
            ax.scatter([x1], [y1], s=40, color=color, marker=marker, alpha=0.95, zorder=10,
                       edgecolors="white", linewidths=1.2)
            # 终点价格标注（偏移避开K线和标记）
            offset = price_range * 0.022
            va = "bottom" if bi["dir"] == "down" else "top"
            y_text = y1 + offset if bi["dir"] == "down" else y1 - offset
            ax.annotate(f"{y1:.2f}", xy=(x1, y1), xytext=(x1, y_text),
                        fontsize=5.5, color=color, alpha=0.8, ha="center", va=va,
                        zorder=6)
        except (KeyError, ValueError, TypeError):
            continue


def _draw_segments(ax, segment_list: List[dict], mpf_df):
    """画线段：粗实线 + 端点方块标记（zorder 在K线上方）"""
    if not segment_list:
        return
    for seg in segment_list:
        try:
            s_date, e_date = seg.get("start_date"), seg.get("end_date")
            if not _in_visible_range(s_date, mpf_df) and not _in_visible_range(e_date, mpf_df):
                continue
            x0, x1 = mdates.date2num(pd.Timestamp(s_date)), mdates.date2num(pd.Timestamp(e_date))
            y0, y1 = seg["start_price"], seg["end_price"]
            color = SEG_UP_COLOR if seg["dir"] == "up" else SEG_DOWN_COLOR
            ax.plot([x0, x1], [y0, y1], color=color, lw=3.0, alpha=0.95, zorder=2)
            # 端点方块标记
            ax.scatter([x0, x1], [y0, y1], s=28, color=color, marker="s",
                       alpha=0.95, zorder=8, edgecolors="white", linewidths=1.0)
        except (KeyError, ValueError, TypeError):
            continue


def _draw_zhongshu(ax, zhongshu_list: List[dict], mpf_df):
    """画中枢：fill_between 限时矩形 + 加粗 ZG/ZD/ZZ 线 + 边缘标注"""
    if not zhongshu_list:
        return
    n = len(zhongshu_list)
    for i, zs in enumerate(zhongshu_list):
        try:
            s_date = zs.get("start_date")
            e_date = zs.get("end_date")
            if not s_date or not e_date:
                continue
            if not _in_visible_range(s_date, mpf_df) and not _in_visible_range(e_date, mpf_df):
                continue
            x0 = mdates.date2num(pd.Timestamp(s_date))
            x1 = mdates.date2num(pd.Timestamp(e_date))
            zg, zd = zs["zg"], zs["zd"]
            color = ZS_COLORS[min(i, len(ZS_COLORS) - 1)]

            # 中枢区间矩形 — fill_between 限制在时间范围
            ax.fill_between([x0, x1], zd, zg, facecolor=color, alpha=ZS_ALPHA, zorder=1)
            # ZG 上沿（粗虚线）
            ax.axhline(y=zg, color=color, lw=1.2, ls="--", alpha=0.7)
            # ZD 下沿（粗虚线）
            ax.axhline(y=zd, color=color, lw=1.2, ls="--", alpha=0.7)
            # ZZ 中轴（细点划线）
            zz = zs.get("zz", (zg + zd) / 2)
            ax.axhline(y=zz, color=color, lw=0.6, ls="-.", alpha=0.45)
            # ZG / ZD 文字标注在右边缘（带白底，避免和K线重叠）
            edge_x = x1 + (x1 - x0) * 0.05
            ax.annotate(f"ZG {zg:.2f}", xy=(edge_x, zg), fontsize=6, color=color,
                        alpha=0.9, ha="left", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.75, pad=1, edgecolor="none"))
            ax.annotate(f"ZD {zd:.2f}", xy=(edge_x, zd), fontsize=6, color=color,
                        alpha=0.9, ha="left", va="top",
                        bbox=dict(facecolor="white", alpha=0.75, pad=1, edgecolor="none"))

            # 中枢标签
            label = zs.get("label", "")
            mid_x = (x0 + x1) / 2
            ax.annotate(f"{label}ZS{i+1}\n{zg:.2f} / {zd:.2f}",
                        xy=(mid_x, zg), fontsize=6.5, color=color, alpha=0.85,
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor=color, alpha=0.8))
        except (KeyError, ValueError, TypeError):
            continue


def _draw_trade_points(ax, trade_points: dict, mpf_df, bi_list: List[dict] = None):
    """画买卖点标记 — 定位在实际笔端点位置（向下笔终点=潜在买点，向上笔终点=潜在卖点）"""
    if not trade_points:
        return
    if bi_list is None:
        bi_list = []

    buy_points = trade_points.get("buy_points", [])
    sell_points = trade_points.get("sell_points", [])

    # 收集可见范围内的笔端点
    down_ends = []  # (x, y) 向下笔终点 → 潜在买点位
    up_ends = []    # (x, y) 向上笔终点 → 潜在卖点位
    for bi in bi_list:
        try:
            e_date = bi.get("end_date")
            if not e_date or not _in_visible_range(e_date, mpf_df):
                continue
            x = mdates.date2num(pd.Timestamp(e_date))
            y = bi["end_price"]
            if bi["dir"] == "down":
                down_ends.append((x, y))
            else:
                up_ends.append((x, y))
        except (KeyError, ValueError, TypeError):
            continue

    price_range = mpf_df["High"].max() - mpf_df["Low"].min()
    last_x = mdates.date2num(mpf_df.index[-1])

    # 买点：红色▲，放在向下笔终点（下跌结束位）
    for i, bp in enumerate(buy_points):
        if i < len(down_ends):
            x, y = down_ends[-(i + 1)]
        else:
            x, y = last_x, mpf_df["Low"].min()
        tp_type = bp.get("type", "买点")
        ax.scatter([x], [y], s=80, color=BUY_COLOR, marker="^",
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(tp_type, xy=(x, y),
                    xytext=(0, -14), textcoords="offset points",
                    fontsize=7, color=BUY_COLOR, fontweight="bold",
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=BUY_COLOR, alpha=0.85))

    # 卖点：绿色▼，放在向上笔终点（上涨结束位）
    for i, sp in enumerate(sell_points):
        if i < len(up_ends):
            x, y = up_ends[-(i + 1)]
        else:
            x, y = last_x, mpf_df["High"].max()
        tp_type = sp.get("type", "卖点")
        ax.scatter([x], [y], s=80, color=SELL_COLOR, marker="v",
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(tp_type, xy=(x, y),
                    xytext=(0, 14), textcoords="offset points",
                    fontsize=7, color=SELL_COLOR, fontweight="bold",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=SELL_COLOR, alpha=0.85))


# ============================================================
# 便捷入口：从分析结果直接画图
# ============================================================

def plot_from_result(single_tf_result: dict,
                     df: pd.DataFrame = None,
                     symbol: str = "",
                     name: str = "",
                     period: str = "daily",
                     start_date: str = None,
                     end_date: str = None,
                     save_path: str = None) -> Optional[str]:
    """
    从单周期分析结果字典直接出图
    """
    return plot_chan_chart(
        df=df,
        bi_list=single_tf_result.get("bi_list", []),
        segment_list=single_tf_result.get("segment_list", []),
        zhongshu_list=single_tf_result.get("zhongshu_list", []),
        trade_points=single_tf_result.get("trade_points", {}),
        symbol=symbol, name=name, period=period,
        start_date=start_date, end_date=end_date,
        max_bars=300,
        save_path=save_path,
    )


def chart_to_base64(df: pd.DataFrame,
                    bi_list: List[dict] = None,
                    segment_list: List[dict] = None,
                    zhongshu_list: List[dict] = None,
                    trade_points: dict = None,
                    symbol: str = "",
                    name: str = "",
                    period: str = "daily",
                    start_date: str = None,
                    end_date: str = None,
                    max_bars: int = 300) -> Optional[str]:
    """
    生成K线图并返回 base64 编码的 PNG 数据（用于嵌入HTML报告）

    Returns:
        base64 字符串，失败返回 None
    """
    import base64
    from io import BytesIO

    if bi_list is None:
        bi_list = []
    if segment_list is None:
        segment_list = []
    if zhongshu_list is None:
        zhongshu_list = []
    if trade_points is None:
        trade_points = {}

    plot_df = df.copy()
    if "date" not in plot_df.columns:
        return None
    if not isinstance(plot_df["date"].iloc[0], pd.Timestamp):
        plot_df["date"] = pd.to_datetime(plot_df["date"])

    if start_date:
        plot_df = plot_df[plot_df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        plot_df = plot_df[plot_df["date"] <= pd.Timestamp(end_date)]

    if len(plot_df) < 5:
        return None

    mpf_df = plot_df.set_index("date")[["open", "high", "low", "close", "volume"]].copy()
    mpf_df.columns = ["Open", "High", "Low", "Close", "Volume"]

    # ---- 数据清洗：去除非交易日空白 + 聚焦最近 N 根K线 ----
    mpf_df = mpf_df[~mpf_df.index.duplicated()].sort_index()
    mpf_df = mpf_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if max_bars and len(mpf_df) > max_bars:
        mpf_df = mpf_df.tail(max_bars).copy()
        keep_dates = set(mpf_df.index)
        plot_df = plot_df[plot_df["date"].isin(keep_dates)]
        plot_df = plot_df.drop_duplicates(subset=["date"])

    apds = []
    if "diff" in plot_df.columns and "dea" in plot_df.columns:
        macd_df = plot_df.set_index("date")[["diff", "dea", "macd_hist"]].copy()
        macd_df.columns = ["DIFF", "DEA", "MACD"]
        macd_pos = macd_df["MACD"].copy()
        macd_neg = macd_df["MACD"].copy()
        macd_pos[macd_pos < 0] = 0
        macd_neg[macd_neg > 0] = 0
        apds.append(mpf.make_addplot(macd_df["DIFF"], panel=2, color="#e67e22", width=0.8))
        apds.append(mpf.make_addplot(macd_df["DEA"], panel=2, color="#3498db", width=0.8))
        apds.append(mpf.make_addplot(macd_pos, type="bar", panel=2, color=UP_COLOR, width=0.7))
        apds.append(mpf.make_addplot(macd_neg, type="bar", panel=2, color=DOWN_COLOR, width=0.7))

    mc = mpf.make_marketcolors(
        up=UP_COLOR, down=DOWN_COLOR, edge="inherit", wick="inherit",
        volume={"up": UP_COLOR, "down": DOWN_COLOR}, alpha=0.85,
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, gridcolor="#e0e0e0", gridstyle="--",
        facecolor="#fafbfc", figcolor="white", y_on_right=False,
    )
    style["rc"] = {"font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun"],
                   "axes.unicode_minus": False}

    title_str = f"{symbol} {name}  {_period_cn(period)}"
    if start_date or end_date:
        sd = start_date or plot_df["date"].iloc[0].strftime("%Y%m%d")
        ed = end_date or plot_df["date"].iloc[-1].strftime("%Y%m%d")
        title_str += f"  [{sd}-{ed}]"

    n_candles = len(mpf_df)
    fig_w = max(18, min(44, n_candles * 0.085))
    fig_h = max(10, fig_w * 0.42)

    fig, axlist = mpf.plot(
        mpf_df, type="candle", style=style, title=title_str,
        volume=True, addplot=apds,
        panel_ratios=(4, 1.2, 1.5) if apds else (4, 1.2),
        returnfig=True, figsize=(fig_w, fig_h),
        warn_too_much_data=len(mpf_df) + 500,
        show_nontrading=False,
        tight_layout=True,
    )

    ax_main = axlist[0]
    _draw_segments(ax_main, segment_list, mpf_df)
    _draw_bi(ax_main, bi_list, mpf_df)
    _draw_zhongshu(ax_main, zhongshu_list, mpf_df)
    _draw_trade_points(ax_main, trade_points, mpf_df, bi_list)

    # 图例
    legend_elements = []
    if bi_list:
        legend_elements.append(Line2D([0], [0], color=BI_UP_COLOR, lw=1.2, ls="--", label="笔"))
    if segment_list:
        legend_elements.append(Line2D([0], [0], color=SEG_UP_COLOR, lw=1.8, label="线段"))
    if zhongshu_list:
        legend_elements.append(mpatches.Patch(color=ZS_COLORS[0], alpha=ZS_ALPHA, label="中枢"))
    buy_pts = trade_points.get("buy_points", [])
    sell_pts = trade_points.get("sell_points", [])
    if buy_pts:
        legend_elements.append(Line2D([0], [0], marker="^", color="w", markerfacecolor=BUY_COLOR,
                                      markersize=8, label="买点"))
    if sell_pts:
        legend_elements.append(Line2D([0], [0], marker="v", color="w", markerfacecolor=SELL_COLOR,
                                      markersize=8, label="卖点"))
    if legend_elements:
        ax_main.legend(handles=legend_elements, loc="upper left", fontsize=7,
                       framealpha=0.85, ncol=min(len(legend_elements), 4))

    # ---- 统一后处理（坐标轴 + 布局） ----
    # ---- 轻量轴美化（不破坏 mplfinance 布局） ----
    _finalize_axes(axlist, mpf_df, apds)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
