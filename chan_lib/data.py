#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多周期数据获取模块（带本地缓存 + 智能重试）
"""

import pandas as pd
import time
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional


# ============================================================
# 缓存配置
# ============================================================

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
CACHE_INDEX = os.path.join(CACHE_DIR, "_index.json")

# TTL（秒）：不同周期的数据有效期
TTL_MAP = {
    "daily": 24 * 3600,       # 日线：24小时
    "weekly": 7 * 24 * 3600,  # 周线：7天
    "monthly": 30 * 24 * 3600, # 月线：30天
    "60": 8 * 3600,           # 60分钟：8小时
    "30": 4 * 3600,           # 30分钟：4小时
    "5": 1 * 3600,            # 5分钟：1小时
    "1": 1800,                # 1分钟：30分钟
}
DEFAULT_TTL = 4 * 3600


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(symbol: str, period: str, adjust: str = "qfq") -> str:
    code = _clean_symbol(symbol)
    return os.path.join(CACHE_DIR, f"{code}_{period}_{adjust}.csv")


def _load_index() -> dict:
    _ensure_cache_dir()
    if os.path.exists(CACHE_INDEX):
        try:
            with open(CACHE_INDEX, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_index(index: dict):
    _ensure_cache_dir()
    with open(CACHE_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _read_cache(symbol: str, period: str, adjust: str = "qfq") -> Optional[pd.DataFrame]:
    """读取缓存。返回None表示缓存不可用。"""
    path = _cache_path(symbol, period, adjust)
    if not os.path.exists(path):
        return None

    index = _load_index()
    cache_key = f"{_clean_symbol(symbol)}_{period}_{adjust}"
    entry = index.get(cache_key, {})
    cached_at = entry.get("cached_at", 0)
    ttl = TTL_MAP.get(period, DEFAULT_TTL)
    age = time.time() - cached_at

    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return None
        # 修复从CSV读取后的数据类型
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "close", "high", "low", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df  # 新鲜或过期都返回（过期由调用方决定是否用）
    except Exception:
        return None


def _write_cache(df: pd.DataFrame, symbol: str, period: str, adjust: str = "qfq"):
    """写入缓存"""
    if df is None or len(df) == 0:
        return
    _ensure_cache_dir()
    path = _cache_path(symbol, period, adjust)
    df.to_csv(path, index=False)

    index = _load_index()
    cache_key = f"{_clean_symbol(symbol)}_{period}_{adjust}"
    index[cache_key] = {
        "cached_at": time.time(),
        "rows": len(df),
        "date_range": [str(df["date"].min()), str(df["date"].max())],
    }
    _save_index(index)


def _cache_age(symbol: str, period: str, adjust: str = "qfq") -> Optional[float]:
    """返回缓存年龄（秒），无缓存返回None"""
    index = _load_index()
    cache_key = f"{_clean_symbol(symbol)}_{period}_{adjust}"
    entry = index.get(cache_key)
    if entry and "cached_at" in entry:
        return time.time() - entry["cached_at"]
    return None


# ============================================================
# 网络请求工具
# ============================================================

def _clean_symbol(symbol: str) -> str:
    """清洗股票代码。若已是 sh.000001 / sz.399001 格式则保留（用于区分指数与股票）"""
    symbol = symbol.strip()
    # 已是 "xx.XXXXXX" 格式 → 保留，避免 sh.000001(上证指数) 与 sz.000001(平安银行) 混淆
    if "." in symbol:
        parts = symbol.split(".")
        if len(parts) == 2 and parts[0].lower() in ("sh", "sz") and len(parts[1]) == 6 and parts[1].isdigit():
            return f"{parts[0].lower()}.{parts[1]}"
    # 先处理带点的格式 sh.600519 / sz.000333
    symbol = symbol.replace("sh.", "").replace("sz.", "").replace("SH.", "").replace("SZ.", "")
    # 再处理不带点的格式 sh600519 / sz000333
    symbol = symbol.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "")
    return symbol


def _normalize_date_arg(value: str = None) -> str:
    """把 YYYY-MM-DD / YYYYMMDD 统一为 YYYYMMDD；空值表示今天。"""
    if not value:
        return datetime.now().strftime("%Y%m%d")
    return str(value).replace("-", "")


def _date_arg_to_timestamp(value: str) -> pd.Timestamp:
    value = _normalize_date_arg(value)
    return pd.Timestamp(datetime.strptime(value, "%Y%m%d"))


def _filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """按起止日期裁剪数据。"""
    if df is None or len(df) == 0 or "date" not in df.columns:
        return df
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    start_ts = _date_arg_to_timestamp(start_date)
    end_ts = _date_arg_to_timestamp(end_date)
    result = result[(result["date"] >= start_ts) & (result["date"] <= end_ts)]
    result.reset_index(drop=True, inplace=True)
    return result


def _cache_covers_range(df: pd.DataFrame, start_date: str, end_date: str,
                        require_end: bool) -> bool:
    """判断缓存是否覆盖本次请求区间。未指定截止日时不强制要求覆盖今天。"""
    if df is None or len(df) == 0 or "date" not in df.columns:
        return False
    dates = pd.to_datetime(df["date"])
    if dates.min() > _date_arg_to_timestamp(start_date):
        return False
    if require_end and dates.max() < _date_arg_to_timestamp(end_date):
        return False
    return True


# ============================================================
# 核心数据获取
# ============================================================

def fetch_kline(symbol: str, period: str = "daily", adjust: str = "qfq",
                max_retries: int = 5, start_date: str = "20200101",
                end_date: str = None, use_cache: bool = True,
                force_refresh: bool = False) -> pd.DataFrame:
    """
    获取A股K线数据（带本地缓存 + 指数退避重试）

    缓存策略：
    1. force_refresh=True → 忽略缓存，强制拉取
    2. 缓存有效（未过期）→ 直接返回缓存
    3. 缓存过期 → 尝试拉取新数据，拉取失败则返回过期缓存
    4. 无缓存 → 拉取，带指数退避重试

    period: daily / weekly / monthly / 60 / 30 / 5 / 1
    """
    code = _clean_symbol(symbol)
    requested_end_date = end_date
    end_date = _normalize_date_arg(end_date)

    # ---- 缓存检查 ----
    if use_cache and not force_refresh:
        cached = _read_cache(symbol, period, adjust)
        if cached is not None:
            age = _cache_age(symbol, period, adjust)
            ttl = TTL_MAP.get(period, DEFAULT_TTL)
            covers_range = _cache_covers_range(
                cached, start_date, end_date, require_end=bool(requested_end_date)
            )
            if age is not None and age < ttl and covers_range:
                # 新鲜缓存，直接返回
                return _filter_date_range(cached, start_date, end_date)
            # 过期缓存：先尝试拉取新数据
            stale_cache = cached
        else:
            stale_cache = None
    else:
        stale_cache = None

    # ---- 网络请求（Baostock） ----
    df = None
    last_error = None

    attempts = max(1, int(max_retries or 1))
    if stale_cache is not None:
        attempts = min(attempts, 2)
    for attempt in range(attempts):
        try:
            df = _fetch_baostock(symbol, period, adjust, start_date, end_date)
            if df is not None and len(df) > 0:
                break
        except Exception as e:
            last_error = e
        if attempt < attempts - 1:
            time.sleep(min(2 ** attempt, 8))

    # 仍然失败 → 使用过期缓存或报错
    if df is None or len(df) == 0:
        if stale_cache is not None:
            age_hours = _cache_age(symbol, period, adjust) / 3600
            print(f"    网络不可用，使用{age_hours:.1f}小时前的缓存数据（{len(stale_cache)}条）")
            return _filter_date_range(stale_cache, start_date, end_date)
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"无法获取{symbol}的{period}数据，且无本地缓存")

    # ---- 处理数据 ----
    col_map = {"日期": "date", "开盘": "open", "收盘": "close",
               "最高": "high", "最低": "low", "成交量": "volume"}
    existing_cols = set(df.columns)
    for cn, en in col_map.items():
        if cn in existing_cols:
            df.rename(columns={cn: en}, inplace=True)
    for c in ["open", "close", "high", "low", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = _filter_date_range(df, start_date, end_date)

    # 写入缓存
    if use_cache:
        _write_cache(df, symbol, period, adjust)

    return df


def _bs_code(symbol: str) -> str:
    """转换为 Baostock 格式: sz.000001 / sh.600519。若已是标准格式则直接透传"""
    symbol = symbol.strip()
    # 已是标准格式 → 直接使用（支持 sh.000001 上证指数等）
    if "." in symbol:
        parts = symbol.split(".")
        if len(parts) == 2 and parts[0].lower() in ("sh", "sz") and len(parts[1]) == 6 and parts[1].isdigit():
            return f"{parts[0].lower()}.{parts[1]}"
    # 自动判断交易所
    code = _clean_symbol(symbol)
    if code.startswith(("0", "3")):
        return f"sz.{code}"
    elif code.startswith(("6", "9")):
        return f"sh.{code}"
    return f"sz.{code}"


def _fetch_baostock(symbol: str, period: str, adjust: str,
                    start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Baostock 数据源（免费、无限流、无需token）
    支持：daily/weekly/monthly/60/30/5/1 分钟线
    """
    try:
        import baostock as bs
    except ImportError:
        return None

    bs_code = _bs_code(symbol)

    # 频率映射
    freq_map = {
        "daily": "d", "weekly": "w", "monthly": "m",
        "60": "60", "30": "30", "5": "5", "1": "1",
    }
    frequency = freq_map.get(period, "d")

    # 复权映射
    adj_map = {"qfq": "2", "hfq": "1", "": "3"}
    adjustflag = adj_map.get(adjust, "2")

    # 日期格式转换
    sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

    lg = bs.login()
    if lg.error_code != "0":
        return None

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=sd, end_date=ed,
            frequency=frequency, adjustflag=adjustflag,
        )
        if rs.error_code != "0":
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return None

        df = pd.DataFrame(data_list, columns=["date", "open", "high", "low", "close", "volume"])
        return df
    finally:
        bs.logout()


# ============================================================
# 多周期数据获取
# ============================================================

def fetch_multi_timeframe(symbol: str,
                          periods: List[str] = None,
                          use_cache: bool = True,
                          start_date: str = "20200101",
                          end_date: str = None) -> Dict[str, pd.DataFrame]:
    """
    获取多周期K线数据
    默认：30分钟 + 日线 + 周线
    start_date: 日线/周线的起始日期。分钟级别会自动使用更晚的日期以保证数据量合理。
    """
    if periods is None:
        periods = ["30", "daily", "weekly", "monthly"]

    result = {}
    for period in periods:
        label = {"30": "30min", "60": "60min", "5": "5min", "1": "1min"}.get(period, period)

        # 各周期的默认起始日期
        if period == "30":
            p_start = "20230101"
        elif period in ("5", "1"):
            p_start = "20240101"
        elif period == "monthly":
            p_start = "20100101"  # 月线需要更长的历史数据
        elif period == "weekly":
            p_start = "20150101"  # 周线推荐至少5年以上
        else:
            p_start = start_date  # 日线使用用户指定的 start_date
        if end_date and p_start > _normalize_date_arg(end_date):
            p_start = start_date

        try:
            df = fetch_kline(symbol, period=period, start_date=p_start,
                             end_date=end_date, use_cache=use_cache,
                             max_retries=2)
            df.attrs["period"] = label
            result[label] = df
        except Exception as e:
            print(f"    {label} 周期获取失败: {e}")
            # 尝试从缓存读取（即使过期）
            cached = _read_cache(symbol, period)
            if cached is not None and len(cached) > 0:
                print(f"    {label} 使用过期缓存（{len(cached)}条）")
                cached.attrs["period"] = label
                result[label] = cached
            else:
                result[label] = pd.DataFrame()

    return result


# ============================================================
# 沪深300成分股获取（带缓存）
# ============================================================

_CSI300_CACHE = os.path.join(CACHE_DIR, "_csi300_stocks.json")


def fetch_csi300_stocks(use_cache: bool = True) -> List[tuple]:
    """获取沪深300成分股列表（缓存24小时）"""
    if use_cache and os.path.exists(_CSI300_CACHE):
        try:
            with open(_CSI300_CACHE, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = data.get("cached_at", 0)
            if time.time() - cached_at < 24 * 3600:
                return [(item[0], item[1]) for item in data.get("stocks", [])]
        except (json.JSONDecodeError, IOError, KeyError):
            pass

    try:
        import baostock as bs
    except ImportError:
        raise RuntimeError("无法获取沪深300成分股列表: baostock 未安装")

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"无法获取沪深300成分股列表: baostock 登录失败 ({lg.error_msg})")

    try:
        rs = bs.query_hs300_stocks()
        if rs.error_code != "0":
            raise RuntimeError(f"无法获取沪深300成分股列表: {rs.error_msg}")

        stocks = []
        while rs.next():
            row = rs.get_row_data()
            code = row[1].replace("sh.", "").replace("sz.", "")
            name = row[2] if len(row) > 2 else ""
            stocks.append((code, name))

        if not stocks:
            raise RuntimeError("无法获取沪深300成分股列表: 返回空列表")
    finally:
        bs.logout()

    # 缓存
    _ensure_cache_dir()
    with open(_CSI300_CACHE, "w", encoding="utf-8") as f:
        json.dump({"cached_at": time.time(), "stocks": stocks}, f, ensure_ascii=False)

    return stocks


# ============================================================
# 缓存管理工具
# ============================================================

def cache_info() -> dict:
    """查看缓存状态"""
    index = _load_index()
    total_files = 0
    total_rows = 0
    fresh = 0
    stale = 0

    for key, entry in index.items():
        total_files += 1
        total_rows += entry.get("rows", 0)
        age = time.time() - entry.get("cached_at", 0)
        parts = key.rsplit("_", 2)
        period = parts[1] if len(parts) >= 2 else "unknown"
        ttl = TTL_MAP.get(period, DEFAULT_TTL)
        if age < ttl:
            fresh += 1
        else:
            stale += 1

    return {
        "total_files": total_files,
        "total_rows": total_rows,
        "fresh": fresh,
        "stale": stale,
        "cache_dir": CACHE_DIR,
        "entries": index,
    }


def clear_cache(older_than_days: int = None):
    """清理缓存"""
    index = _load_index()
    keys_to_delete = []

    for key, entry in index.items():
        if older_than_days is not None:
            age_days = (time.time() - entry.get("cached_at", 0)) / 86400
            if age_days > older_than_days:
                keys_to_delete.append(key)
                # 删除文件
                parts = key.rsplit("_", 2)
                if len(parts) >= 3:
                    path = os.path.join(CACHE_DIR, f"{parts[0]}_{parts[1]}_{parts[2]}.csv")
                    if os.path.exists(path):
                        os.remove(path)
        else:
            keys_to_delete.append(key)
            parts = key.rsplit("_", 2)
            if len(parts) >= 3:
                path = os.path.join(CACHE_DIR, f"{parts[0]}_{parts[1]}_{parts[2]}.csv")
                if os.path.exists(path):
                    os.remove(path)

    for key in keys_to_delete:
        del index[key]

    _save_index(index)
    return len(keys_to_delete)
