#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自选池管理器 —— 持久化存储，用户可自行增删
"""

import os
import json

_WATCHLIST_FILE = os.path.join(
    os.path.expanduser("~"), ".chanlun_pro", "watchlist.json"
)

_DEFAULT = [
    ("601689", "拓普集团"),
    ("688082", "盛美上海"),
    ("600522", "中天科技"),
    ("605117", "德业股份"),
    ("688047", "龙芯中科"),
    ("600584", "长电科技"),
    ("000001", "平安银行"),
    ("600519", "贵州茅台"),
    ("000858", "五粮液"),
    ("601318", "中国平安"),
    ("300750", "宁德时代"),
    ("600036", "招商银行"),
    ("000333", "美的集团"),
    ("600276", "恒瑞医药"),
]


def _ensure_dir():
    os.makedirs(os.path.dirname(_WATCHLIST_FILE), exist_ok=True)


def load_watchlist():
    """加载自选池，返回 [(code, name), ...]"""
    if not os.path.exists(_WATCHLIST_FILE):
        _ensure_dir()
        save_watchlist(_DEFAULT)
        return list(_DEFAULT)

    try:
        with open(_WATCHLIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(item["code"], item["name"]) for item in data]
    except Exception:
        return list(_DEFAULT)


def save_watchlist(stocks):
    """保存自选池，stocks 为 [(code, name), ...]"""
    _ensure_dir()
    data = [{"code": code, "name": name} for code, name in stocks]
    with open(_WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_stock(code: str, name: str):
    """添加一只股票"""
    stocks = load_watchlist()
    # 去重
    stocks = [(c, n) for c, n in stocks if c != code]
    stocks.append((code.strip(), name.strip()))
    save_watchlist(stocks)


def remove_stock(code: str):
    """删除一只股票"""
    stocks = load_watchlist()
    stocks = [(c, n) for c, n in stocks if c != code]
    save_watchlist(stocks)


def reset_to_default():
    """重置为默认自选池"""
    save_watchlist(_DEFAULT)


def lookup_stock_name(code: str) -> str:
    """根据股票代码查询名称，失败返回空字符串"""
    code = code.strip()
    if len(code) != 6 or not code.isdigit():
        return ""
    try:
        import baostock as bs
    except ImportError:
        return ""

    # 判断交易所
    if code.startswith(("0", "3")):
        bs_code = f"sz.{code}"
    else:
        bs_code = f"sh.{code}"

    try:
        lg = bs.login()
        if lg.error_code != "0":
            return ""
        rs = bs.query_stock_basic(code=bs_code)
        if rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            # row: [code, code_name, ipoDate, outDate, type, status]
            return row[1] if len(row) > 1 else ""
        return ""
    except Exception:
        return ""
    finally:
        try:
            bs.logout()
        except Exception:
            pass
