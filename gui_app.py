#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
《缠论》股析 —— 桌面GUI
激活码保护 + 单股分析 + 自选池管理
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import io
import os
from datetime import datetime

if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from activation import get_hardware_id, verify_key, is_activated, save_activation
from watchlist_manager import (load_watchlist, add_stock,
                               remove_stock, save_watchlist,
                               lookup_stock_name)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("《缠论》股析 V1.1")
        self.geometry("960x720")
        self.minsize(800, 600)
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)
        if is_activated():
            self._show_main()
        else:
            self._show_activation()

    def _clear_container(self):
        for w in self.container.winfo_children():
            w.destroy()

    def _show_activation(self):
        self._clear_container()
        ActivationFrame(self.container, on_done=self._on_activated).pack(
            fill="both", expand=True)

    def _show_main(self):
        self._clear_container()
        MainFrame(self.container).pack(fill="both", expand=True)

    def _on_activated(self):
        self._show_main()


# ============================================================
# 激活界面
# ============================================================
class ActivationFrame(tk.Frame):
    def __init__(self, parent, on_done):
        super().__init__(parent, bg="#f1f3f5")
        self.on_done = on_done
        self._build()

    def _build(self):
        card = tk.Frame(self, bg="#ffffff", padx=44, pady=34,
                        highlightbackground="#d5dbe0", highlightthickness=1)
        card.place(relx=0.5, rely=0.38, anchor="center")

        # 品牌
        tk.Label(card, text="《缠论》股析", font=("微软雅黑", 22, "bold"),
                 bg="#ffffff", fg="#0d1b2a").pack()
        tk.Label(card, text="基于缠中说禅《教你炒股票108课》完整理论体系",
                 font=("微软雅黑", 9), bg="#ffffff", fg="#7f8c8d").pack(pady=(4, 22))

        # 分隔线
        tk.Frame(card, bg="#e8ecf1", height=1).pack(fill="x")

        # 机器码
        tk.Label(card, text="本机机器码", font=("微软雅黑", 10, "bold"),
                 bg="#ffffff", fg="#2c3e50").pack(anchor="w", pady=(18, 6))

        hwid_frame = tk.Frame(card, bg="#f8f9fb", padx=14, pady=10)
        hwid_frame.pack(fill="x")
        self._hwid = get_hardware_id()
        tk.Label(hwid_frame, text=self._hwid, font=("Consolas", 14),
                 bg="#f8f9fb", fg="#2471a3").pack(side="left")
        tk.Button(hwid_frame, text="  复 制  ", font=("微软雅黑", 9, "bold"),
                  bg="#2471a3", fg="white", activebackground="#1a5276",
                  relief="flat", borderwidth=0, cursor="hand2",
                  padx=10, pady=2, command=self._copy_hwid).pack(side="right")

        # 激活码
        tk.Label(card, text="激活码", font=("微软雅黑", 10, "bold"),
                 bg="#ffffff", fg="#2c3e50").pack(anchor="w", pady=(18, 6))
        self._key_var = tk.StringVar()
        key_entry = tk.Entry(card, textvariable=self._key_var,
                             font=("Consolas", 17), justify="center",
                             width=22, relief="solid", borderwidth=1,
                             highlightcolor="#2471a3",
                             highlightbackground="#d5dbe0")
        key_entry.pack(ipady=5)
        key_entry.bind("<KeyRelease>", self._on_key_type)

        self._act_btn = tk.Button(card, text="  激    活  ",
                                  font=("微软雅黑", 14, "bold"),
                                  bg="#27ae60", fg="white",
                                  activebackground="#1e8449",
                                  relief="flat", borderwidth=0,
                                  cursor="hand2",
                                  padx=36, pady=6,
                                  command=self._activate)
        self._act_btn.pack(pady=(18, 10))

        self._status = tk.Label(card, text="", font=("微软雅黑", 9),
                                bg="#ffffff", fg="#c0392b")
        self._status.pack()

        tk.Label(card,
                 text="将机器码发给卖家获取激活码，一机一码永久使用",
                 font=("微软雅黑", 8), bg="#ffffff", fg="#b0b8c0").pack(pady=(14, 0))

    def _copy_hwid(self):
        self.clipboard_clear()
        self.clipboard_append(self._hwid)
        self._status.config(text="✓ 机器码已复制，发给卖家获取激活码", fg="#27ae60")

    def _on_key_type(self, event):
        raw = self._key_var.get().upper()
        filtered = "".join(c for c in raw if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if len(filtered) > 16:
            filtered = filtered[:16]
        if len(filtered) > 4 and len(filtered) <= 16:
            parts = [filtered[i:i+4] for i in range(0, len(filtered), 4)]
            formatted = "-".join(parts)
            self._key_var.set(formatted)
            entry = event.widget
            entry.icursor(len(formatted))

    def _activate(self):
        key = self._key_var.get().strip()
        if len(key.replace("-", "")) < 16:
            self._status.config(text="激活码不完整，请检查", fg="#c0392b")
            return
        if verify_key(key):
            save_activation(key)
            self._status.config(text="✓ 激活成功！正在进入...", fg="#27ae60")
            self._act_btn.config(state="disabled")
            self.after(800, self.on_done)
        else:
            self._status.config(text="激活码无效，请检查是否为本机机器码生成的", fg="#c0392b")


# ============================================================
# 主分析界面
# ============================================================
class MainFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._analysis_running = False
        self._watchlist = load_watchlist()
        self._build()

    def _build(self):
        # ---- 顶部标题栏 ----
        header = tk.Frame(self, bg="#0d1b2a", height=52)
        header.pack(fill="x")
        header.pack_propagate(False)

        # 左侧品牌区
        brand_frame = tk.Frame(header, bg="#0d1b2a")
        brand_frame.pack(side="left", padx=20, pady=8)
        tk.Label(brand_frame, text="《缠论》股析",
                 font=("微软雅黑", 16, "bold"),
                 bg="#0d1b2a", fg="#ffffff").pack(side="left")
        tk.Label(brand_frame, text="  |  Chan Theory Analyzer",
                 font=("微软雅黑", 8),
                 bg="#0d1b2a", fg="#5a7a9a").pack(side="left", pady=(4, 0))

        # 右侧版本信息
        info_frame = tk.Frame(header, bg="#0d1b2a")
        info_frame.pack(side="right", padx=20, pady=10)
        v_badge = tk.Frame(info_frame, bg="#1e3b5a")
        v_badge.pack(side="right")
        tk.Label(v_badge, text=" V1.1 ",
                 font=("微软雅黑", 8, "bold"), bg="#1e3b5a", fg="#64b5f6").pack(padx=6, pady=1)
        act_badge = tk.Frame(info_frame, bg="#1a3c2a")
        act_badge.pack(side="right", padx=(0, 6))
        tk.Label(act_badge, text=" 已激活 ",
                 font=("微软雅黑", 8, "bold"), bg="#1a3c2a", fg="#66bb6a").pack(padx=6, pady=1)

        # ---- 主体：左右分栏 ----
        paned = tk.PanedWindow(self, orient="horizontal", bg="#c8ccd0", sashwidth=2)
        paned.pack(fill="both", expand=True, padx=10, pady=10)

        # 左侧：自选池
        left_panel = tk.Frame(paned, bg="#ffffff", width=220)
        paned.add(left_panel, minsize=180)
        self._build_watchlist_panel(left_panel)

        # 右侧：分析区
        right_panel = tk.Frame(paned, bg="#f1f3f5")
        paned.add(right_panel, minsize=400)
        self._build_analysis_panel(right_panel)

        # ---- 底部状态栏 ----
        status_bar = tk.Frame(self, bg="#e8ecf1", height=26)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        self._status_text = tk.StringVar(value="就绪")
        tk.Label(status_bar, textvariable=self._status_text,
                 font=("微软雅黑", 8), bg="#e8ecf1", fg="#7f8c8d").pack(
            side="left", padx=14, pady=3)
        tk.Label(status_bar, text="Baostock 数据源",
                 font=("微软雅黑", 7), bg="#e8ecf1", fg="#b0b8c0").pack(
            side="right", padx=14, pady=3)

    # ============================================================
    # 左侧：自选池面板
    # ============================================================
    def _build_watchlist_panel(self, parent):
        # 标题栏
        title_frame = tk.Frame(parent, bg="#f8f9fb", height=40)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)
        tk.Label(title_frame, text="我的自选", font=("微软雅黑", 11, "bold"),
                 bg="#f8f9fb", fg="#2c3e50").pack(side="left", padx=14, pady=(9, 0))
        tk.Label(title_frame, text="双击填入 →", font=("微软雅黑", 7),
                 bg="#f8f9fb", fg="#5a7a9a").pack(side="left", pady=(10, 0))
        count_badge = tk.Frame(title_frame, bg="#e8ecf1")
        count_badge.pack(side="right", padx=10, pady=9)
        self._wl_count_label = tk.Label(count_badge,
            text=f" {len(self._watchlist)} 只 ",
            font=("微软雅黑", 8, "bold"), bg="#e8ecf1", fg="#5a7a9a")
        self._wl_count_label.pack(padx=4, pady=1)

        # 分隔线
        tk.Frame(parent, bg="#e8ecf1", height=1).pack(fill="x")

        # 列表
        list_frame = tk.Frame(parent, bg="white")
        list_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_frame, relief="flat", borderwidth=0)
        scrollbar.pack(side="right", fill="y")

        self._watchlist_box = tk.Listbox(
            list_frame, font=("微软雅黑", 10),
            yscrollcommand=scrollbar.set,
            selectmode="single", activestyle="none",
            relief="flat", borderwidth=0, highlightthickness=0,
            bg="#ffffff", fg="#2c3e50",
            selectbackground="#e3f2fd", selectforeground="#1565C0")
        self._watchlist_box.pack(fill="both", expand=True, padx=2, pady=2)
        scrollbar.config(command=self._watchlist_box.yview)
        self._watchlist_box.bind("<Double-Button-1>", self._on_watchlist_select)

        # 右键菜单
        self._wl_menu = tk.Menu(self._watchlist_box, tearoff=0,
                                font=("微软雅黑", 9))
        self._wl_menu.add_command(label="分析这只", command=self._wl_analyze)
        self._wl_menu.add_separator()
        self._wl_menu.add_command(label="从列表删除", command=self._wl_remove)
        self._watchlist_box.bind("<Button-3>", self._wl_right_click)

        self._refresh_watchlist_list()

        # 底部操作栏
        tk.Frame(parent, bg="#e8ecf1", height=1).pack(fill="x")
        btn_frame = tk.Frame(parent, bg="#f8f9fb")
        btn_frame.pack(fill="x")

        add_btn = tk.Button(btn_frame, text="＋ 添加",
                           font=("微软雅黑", 9, "bold"),
                           bg="#27ae60", fg="white",
                           activebackground="#1e8449",
                           padx=14, pady=3, cursor="hand2",
                           relief="flat", borderwidth=0,
                           command=self._wl_add_dialog)
        add_btn.pack(side="left", padx=8, pady=8)
        del_btn = tk.Button(btn_frame, text="－ 删除",
                           font=("微软雅黑", 9, "bold"),
                           bg="#7f8c8d", fg="white",
                           activebackground="#6c7a7d",
                           padx=14, pady=3, cursor="hand2",
                           relief="flat", borderwidth=0,
                           command=self._wl_remove)
        del_btn.pack(side="left", pady=8)

    def _refresh_watchlist_list(self):
        self._watchlist = load_watchlist()
        self._watchlist_box.delete(0, "end")
        for code, name in self._watchlist:
            self._watchlist_box.insert("end", f"  {code}  {name}")
        self._wl_count_label.config(text=f" {len(self._watchlist)} 只 ")

    def _on_watchlist_select(self, event):
        sel = self._watchlist_box.curselection()
        if sel:
            idx = sel[0]
            self._select_watchlist_stock(idx)
            # 仅填入代码和名称，用户自行调整日期后再点分析

    def _wl_analyze(self):
        sel = self._watchlist_box.curselection()
        if sel:
            self._select_watchlist_stock(sel[0])
            # 仅填入，用户自行调整后点击分析

    def _wl_right_click(self, event):
        try:
            idx = self._watchlist_box.nearest(event.y)
            self._watchlist_box.selection_clear(0, "end")
            self._watchlist_box.selection_set(idx)
            self._wl_menu.post(event.x_root, event.y_root)
        except Exception:
            pass

    def _wl_remove(self):
        sel = self._watchlist_box.curselection()
        if not sel:
            return
        idx = sel[0]
        code, name = self._watchlist[idx]
        if messagebox.askyesno("确认", f"确定删除 {code} {name}？"):
            remove_stock(code)
            self._refresh_watchlist_list()

    def _wl_add_dialog(self):
        dialog = tk.Toplevel(self)
        dialog.title("添加自选股")
        dialog.geometry("320x220")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        tk.Label(dialog, text="股票代码", font=("微软雅黑", 10)).pack(pady=(16, 2))
        code_var = tk.StringVar()
        code_entry = tk.Entry(dialog, textvariable=code_var, font=("Consolas", 13),
                              width=14, justify="center")
        code_entry.pack(ipady=3)

        tk.Label(dialog, text="股票名称", font=("微软雅黑", 10)).pack(pady=(8, 2))
        name_var = tk.StringVar()
        name_entry = tk.Entry(dialog, textvariable=name_var, font=("微软雅黑", 12),
                              width=14, justify="center")
        name_entry.pack(ipady=3)

        lookup_label = tk.Label(dialog, text="", font=("微软雅黑", 8), fg="#999")
        lookup_label.pack(pady=(2, 0))

        def auto_lookup(event=None):
            c = code_var.get().strip()
            if len(c) == 6 and c.isdigit():
                lookup_label.config(text="查询中...", fg="#999")
                dialog.update()
                found = lookup_stock_name(c)
                if found:
                    name_var.set(found)
                    lookup_label.config(text=f"已找到：{found}", fg="#27ae60")
                else:
                    lookup_label.config(text="未查到名称，请手动输入", fg="#FF9800")
            else:
                lookup_label.config(text="")

        code_entry.bind("<FocusOut>", auto_lookup)

        def do_add():
            c = code_var.get().strip()
            n = name_var.get().strip()
            if not c:
                return
            if len(c) != 6 or not c.isdigit():
                messagebox.showwarning("提示", "请输入6位数字股票代码", parent=dialog)
                return
            add_stock(c, n or c)
            self._refresh_watchlist_list()
            dialog.destroy()

        tk.Button(dialog, text="添加", font=("微软雅黑", 10),
                  bg="#27ae60", fg="white", width=12,
                  command=do_add).pack(pady=(12, 0))

        code_entry.focus_set()

    def _select_watchlist_stock(self, idx):
        if 0 <= idx < len(self._watchlist):
            code, name = self._watchlist[idx]
            self._code_var.set(code)
            self._name_var.set(name)

    # ============================================================
    # 右侧：分析面板（全新设计）
    # ============================================================
    def _build_analysis_panel(self, parent):
        # ---- 输入卡片 ----
        input_card = tk.Frame(parent, bg="#ffffff",
                              highlightbackground="#d5dbe0", highlightthickness=1)
        input_card.pack(fill="x", padx=0, pady=(0, 10))

        inner = tk.Frame(input_card, bg="#ffffff")
        inner.pack(padx=20, pady=16, fill="x")

        # ---- 第1行：股票代码 + 名称 ----
        row1 = tk.Frame(inner, bg="#ffffff")
        row1.pack(fill="x", pady=(0, 10))
        tk.Label(row1, text="股票代码", font=("微软雅黑", 10, "bold"),
                 bg="#ffffff", fg="#2c3e50").pack(side="left")
        self._code_var = tk.StringVar()
        code_entry = tk.Entry(row1, textvariable=self._code_var,
                              font=("Consolas", 15), width=8,
                              relief="solid", borderwidth=1,
                              highlightthickness=0,
                              highlightcolor="#2471a3",
                              highlightbackground="#d5dbe0")
        code_entry.pack(side="left", padx=(8, 20), ipady=4)
        code_entry.bind("<FocusOut>", self._on_code_focus_out)

        tk.Label(row1, text="名称", font=("微软雅黑", 10, "bold"),
                 bg="#ffffff", fg="#2c3e50").pack(side="left")
        self._name_var = tk.StringVar()
        name_entry = tk.Entry(row1, textvariable=self._name_var,
                              font=("微软雅黑", 14), width=9,
                              relief="solid", borderwidth=1,
                              highlightthickness=0,
                              highlightcolor="#2471a3",
                              highlightbackground="#d5dbe0")
        name_entry.pack(side="left", padx=(8, 10), ipady=4)

        self._name_lookup_label = tk.Label(row1, text="", font=("微软雅黑", 8),
                                           bg="#ffffff", fg="#7f8c8d")
        self._name_lookup_label.pack(side="left")

        # ---- 分隔 ----
        tk.Frame(inner, bg="#e8ecf1", height=1).pack(fill="x", pady=(0, 10))

        # ---- 第2行：日期选择区 ----
        date_frame = tk.Frame(inner, bg="#ffffff")
        date_frame.pack(fill="x", pady=(0, 8))

        now = datetime.now()
        now_year = now.year
        years = [str(y) for y in range(2010, now_year + 1)]
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]

        # 起始日期
        start_group = tk.Frame(date_frame, bg="#f8f9fb", padx=12, pady=10)
        start_group.pack(side="left", padx=(0, 12))
        tk.Label(start_group, text="起始日期", font=("微软雅黑", 10, "bold"),
                 bg="#f8f9fb", fg="#2c3e50").pack(side="left", padx=(0, 8))

        self._start_year = tk.StringVar(value="2020")
        self._start_month = tk.StringVar(value="01")
        self._start_day = tk.StringVar(value="01")

        cb_year_s = ttk.Combobox(start_group, textvariable=self._start_year,
                                  values=years, width=5, font=("微软雅黑", 10),
                                  state="readonly")
        cb_year_s.pack(side="left", padx=(0, 1))
        tk.Label(start_group, text="年", font=("微软雅黑", 9), bg="#f8f9fb", fg="#7f8c8d").pack(side="left")
        cb_month_s = ttk.Combobox(start_group, textvariable=self._start_month,
                                   values=months, width=4, font=("微软雅黑", 10),
                                   state="readonly")
        cb_month_s.pack(side="left", padx=(4, 1))
        tk.Label(start_group, text="月", font=("微软雅黑", 9), bg="#f8f9fb", fg="#7f8c8d").pack(side="left")
        cb_day_s = ttk.Combobox(start_group, textvariable=self._start_day,
                                 values=days, width=4, font=("微软雅黑", 10),
                                 state="readonly")
        cb_day_s.pack(side="left", padx=(4, 1))
        tk.Label(start_group, text="日", font=("微软雅黑", 9), bg="#f8f9fb", fg="#7f8c8d").pack(side="left")

        # 截止日期
        end_group = tk.Frame(date_frame, bg="#f8f9fb", padx=12, pady=6)
        end_group.pack(side="left")

        # Row 1: 标签 + 日期选择器
        end_row1 = tk.Frame(end_group, bg="#f8f9fb")
        end_row1.pack(fill="x")
        tk.Label(end_row1, text="截止日期", font=("微软雅黑", 10, "bold"),
                 bg="#f8f9fb", fg="#2c3e50").pack(side="left", padx=(0, 8))

        self._end_year = tk.StringVar(value=str(now_year))
        self._end_month = tk.StringVar(value=f"{now.month:02d}")
        self._end_day = tk.StringVar(value=f"{now.day:02d}")

        self._cb_end_year = ttk.Combobox(end_row1, textvariable=self._end_year,
                                          values=years, width=5, font=("微软雅黑", 10),
                                          state="readonly")
        self._cb_end_year.pack(side="left", padx=(0, 1))
        tk.Label(end_row1, text="年", font=("微软雅黑", 9), bg="#f8f9fb", fg="#7f8c8d").pack(side="left")
        self._cb_end_month = ttk.Combobox(end_row1, textvariable=self._end_month,
                                           values=months, width=4, font=("微软雅黑", 10),
                                           state="readonly")
        self._cb_end_month.pack(side="left", padx=(4, 1))
        tk.Label(end_row1, text="月", font=("微软雅黑", 9), bg="#f8f9fb", fg="#7f8c8d").pack(side="left")
        self._cb_end_day = ttk.Combobox(end_row1, textvariable=self._end_day,
                                         values=days, width=4, font=("微软雅黑", 10),
                                         state="readonly")
        self._cb_end_day.pack(side="left", padx=(4, 1))
        tk.Label(end_row1, text="日", font=("微软雅黑", 9), bg="#f8f9fb", fg="#7f8c8d").pack(side="left")

        # Row 2: "到今天" 复选框（独立一行，避免文字溢出）
        end_row2 = tk.Frame(end_group, bg="#f8f9fb")
        end_row2.pack(fill="x", pady=(3, 0))
        self._to_today_var = tk.BooleanVar(value=True)
        self._to_today_cb = tk.Checkbutton(end_row2, text="到今天",
                                            variable=self._to_today_var,
                                            font=("微软雅黑", 9, "bold"),
                                            bg="#f8f9fb", fg="#e74c3c",
                                            selectcolor="#f8f9fb",
                                            activebackground="#f8f9fb",
                                            command=self._toggle_end_date)
        self._to_today_cb.pack(side="left", padx=(0, 0))
        self._toggle_end_date()

        # ---- 理论建议 ----
        theory_note = tk.Frame(inner, bg="#fef9e7")
        theory_note.pack(fill="x", pady=(0, 12))
        tk.Label(theory_note,
                 text="  \U0001f4a1 缠论建议：日线分析推荐 2-3 年数据（约500根K线）；"
                      "联立分析推荐 5 年起（保证周线足够K线构建中枢）。"
                      "截止日期默认\"到今天\"，也可指定回测日期。",
                 font=("微软雅黑", 8), bg="#fef9e7", fg="#7d6608",
                 justify="left", wraplength=520).pack(padx=10, pady=6, anchor="w")

        # ---- 操作按钮 ----
        btn_row = tk.Frame(inner, bg="#ffffff")
        btn_row.pack(fill="x")

        self._btn_daily = tk.Button(btn_row, text="  日线分析  ",
                                    font=("微软雅黑", 11, "bold"),
                                    bg="#2471a3", fg="white",
                                    activebackground="#1a5276",
                                    padx=22, pady=7, cursor="hand2",
                                    relief="flat", borderwidth=0,
                                    command=lambda: self._run_analysis("daily"))
        self._btn_daily.pack(side="left", padx=(0, 10))

        self._btn_full = tk.Button(btn_row, text="  多级别联立分析  ",
                                   font=("微软雅黑", 11, "bold"),
                                   bg="#e67e22", fg="white",
                                   activebackground="#ba4a00",
                                   padx=22, pady=7, cursor="hand2",
                                   relief="flat", borderwidth=0,
                                   command=lambda: self._run_analysis("full"))
        self._btn_full.pack(side="left")

        # 进度条
        self._progress = ttk.Progressbar(input_card, mode="indeterminate")
        self._progress.pack(fill="x", padx=20, pady=(0, 12))

        # ---- 结果区 ----
        result_header = tk.Frame(parent, bg="#f1f3f5")
        result_header.pack(fill="x")
        tk.Label(result_header, text="分析报告", font=("微软雅黑", 11, "bold"),
                 bg="#f1f3f5", fg="#2c3e50").pack(side="left")

        # 完整报告按钮
        self._chart_btn = tk.Button(result_header, text="  查看完整报告 (含K线图)  ",
                                    font=("微软雅黑", 10, "bold"),
                                    bg="#27ae60", fg="white",
                                    activebackground="#1e8449",
                                    padx=16, pady=5, cursor="hand2",
                                    relief="flat", borderwidth=0,
                                    command=self._view_chart, state="disabled")
        self._chart_btn.pack(side="right", padx=(0, 2))

        # 复制按钮
        copy_btn = tk.Button(result_header, text="  复制报告  ",
                           font=("微软雅黑", 9),
                           bg="#7f8c8d", fg="white",
                           activebackground="#6c7a7d",
                           padx=10, pady=3, cursor="hand2",
                           relief="flat", borderwidth=0,
                           command=self._copy_report)
        copy_btn.pack(side="right", padx=4)

        # ---- 结果文本框 ----
        self._result_text = scrolledtext.ScrolledText(
            parent, font=("微软雅黑", 10), wrap="word",
            bg="#ffffff", fg="#2c3e50", insertbackground="#2c3e50",
            relief="solid", borderwidth=1,
            highlightbackground="#d5dbe0",
            padx=16, pady=14)
        self._result_text.pack(fill="both", expand=True, pady=(4, 0))

        # 文本样式标签
        self._result_text.tag_configure("section", foreground="#1565C0",
                                        font=("微软雅黑", 11, "bold"),
                                        spacing1=10, spacing3=4)
        self._result_text.tag_configure("buy", foreground="#2E7D32",
                                        font=("微软雅黑", 10, "bold"))
        self._result_text.tag_configure("sell", foreground="#C62828",
                                        font=("微软雅黑", 10, "bold"))
        self._result_text.tag_configure("warn", foreground="#E65100",
                                        font=("微软雅黑", 10, "bold"))
        self._result_text.tag_configure("header", foreground="#1a1a2e",
                                        font=("微软雅黑", 13, "bold"),
                                        spacing1=6, spacing3=6)
        self._result_text.tag_configure("separator", foreground="#bdbdbd",
                                        font=("Consolas", 7))
        self._result_text.tag_configure("price", foreground="#1565C0",
                                        font=("微软雅黑", 10, "bold"))
        self._result_text.tag_configure("decision", foreground="#1a1a2e",
                                        font=("微软雅黑", 12, "bold"),
                                        spacing1=8, spacing3=8)
        self._result_text.tag_configure("guide", foreground="#1565C0",
                                        font=("微软雅黑", 10, "bold"),
                                        background="#e3f2fd",
                                        spacing1=6, spacing3=3,
                                        lmargin1=10, lmargin2=10)
        self._result_text.tag_configure("guide_btn", foreground="#1e8449",
                                        font=("微软雅黑", 11, "bold"),
                                        background="#d5f5e3",
                                        spacing1=4, spacing3=4,
                                        justify="center")

    # ---- 股票名称自动查询 ----
    def _on_code_focus_out(self, event=None):
        """输入代码后自动查名称"""
        code = self._code_var.get().strip()
        if len(code) == 6 and code.isdigit():
            self._name_lookup_label.config(text="查询中...", fg="#999")
            self.update()
            found = lookup_stock_name(code)
            if found:
                self._name_var.set(found)
                self._name_lookup_label.config(text=f"✓ {found}", fg="#27ae60")
            else:
                self._name_lookup_label.config(text="未查到，请手动输入", fg="#FF9800")
        else:
            self._name_lookup_label.config(text="")

    # ---- 截止日期切换 ----
    def _toggle_end_date(self):
        """到今天勾选时禁用截止日期下拉框"""
        if self._to_today_var.get():
            state = "disabled"
        else:
            state = "readonly"
        self._cb_end_year.config(state=state)
        self._cb_end_month.config(state=state)
        self._cb_end_day.config(state=state)

    # ---- 获取当前设置的日期 ----
    def _get_start_date(self) -> str:
        return f"{self._start_year.get()}{self._start_month.get()}{self._start_day.get()}"

    def _get_end_date(self) -> str:
        if self._to_today_var.get():
            return ""
        return f"{self._end_year.get()}{self._end_month.get()}{self._end_day.get()}"

    # ============================================================
    # 分析执行
    # ============================================================
    def _run_analysis(self, mode: str):
        if self._analysis_running:
            return

        start_date = self._get_start_date()
        end_date = self._get_end_date()

        # 校验起止日期合法性
        if end_date and end_date < start_date:
            self._result_text.delete("1.0", "end")
            self._result_text.insert("1.0",
                f"日期设置错误：截止日期（{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}）"
                f"早于起始日期（{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}），"
                f"请重新设置。\n")
            return

        self._analysis_running = True
        self._btn_daily.config(state="disabled")
        self._btn_full.config(state="disabled")
        self._chart_btn.config(state="disabled")
        self._progress.start(10)
        self._result_text.delete("1.0", "end")
        self._result_text.insert("1.0", "分析中，请稍候...\n")
        self._status_text.set("正在获取数据...")

        self._html_path = None
        self._chart_params = None

        thread = threading.Thread(
            target=self._analysis_worker, args=(mode,), daemon=True)
        thread.start()

    def _analysis_worker(self, mode: str):
        try:
            start_date = self._get_start_date()
            end_date = self._get_end_date()
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                if mode == "daily":
                    self._run_single_daily(start_date, end_date)
                else:
                    self._run_single_full(start_date, end_date)
            finally:
                sys.stdout = old_stdout
            result = buf.getvalue()
            self.after(0, lambda: self._on_done(result, None))
        except Exception as e:
            self.after(0, lambda: self._on_done("", str(e)))

    def _run_single_daily(self, start_date: str = "20200101", end_date: str = ""):
        from chan_analyzer import analyze_stock, print_full_report
        from chan_lib.strategy import trading_decision
        code = self._code_var.get().strip()
        name = self._name_var.get().strip()
        if not code:
            print("请先输入股票代码")
            return

        self._update_status("正在获取日线数据...")
        result = analyze_stock(code, name, quiet=True, start_date=start_date,
                               end_date=end_date)
        if not result:
            print(f"分析失败：{code} {name}，请检查代码是否正确或网络是否正常")
            return

        self._update_status("正在生成报告...")
        dec = trading_decision(result)
        wrapped = {
            "symbol": code, "name": name,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "levels_analyzed": ["daily"],
            "timeframe_results": {"daily": result},
            "single_decisions": {"daily": dec},
        }
        print_full_report(wrapped)

        # 存储图表生成参数，由主线程 _generate_html_report 统一处理
        self._chart_params = {
            "wrapped": wrapped,
            "code": code,
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "timeframe_results": {"daily": result},
        }

    def _run_single_full(self, start_date: str = "20200101", end_date: str = ""):
        from chan_analyzer import analyze_multi_timeframe, print_full_report

        code = self._code_var.get().strip()
        name = self._name_var.get().strip()
        if not code:
            print("请先输入股票代码")
            return

        self._update_status("正在获取多周期数据...")
        result = analyze_multi_timeframe(code, name, start_date=start_date,
                                         end_date=end_date)
        if not result or not result.get("timeframe_results"):
            print(f"分析失败：{code} {name}，请检查代码或网络")
            return

        self._update_status("正在生成综合报告...")
        print_full_report(result)

        # 存储图表生成参数，由主线程 _generate_html_report 统一处理
        self._chart_params = {
            "wrapped": result,
            "code": code,
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "timeframe_results": result.get("timeframe_results", {}),
        }

    def _update_status(self, msg):
        self.after(0, lambda: self._status_text.set(msg))

    def _on_done(self, result: str, error: str):
        self._analysis_running = False
        self._progress.stop()
        self._btn_daily.config(state="normal")
        self._btn_full.config(state="normal")
        self._status_text.set("分析完成")
        if error:
            self._result_text.delete("1.0", "end")
            self._result_text.insert("1.0", f"分析出错：{error}\n")
            self._chart_btn.config(state="disabled")
        else:
            self._result_text.delete("1.0", "end")
            self._result_text.insert("end", result)
            self._apply_report_highlighting()
            self._chart_btn.config(state="disabled")
            # 在主线程中生成HTML报告
            self.after(10, self._generate_html_report)

    def _apply_report_highlighting(self):
        """给报告文本加上颜色标注"""
        content = self._result_text.get("1.0", "end-1c")
        lines = content.split("\n")

        for i, line in enumerate(lines):
            idx = f"{i + 1}.0"
            idx_end = f"{i + 1}.end"

            if "《缠论》股析" in line or "缠论多级别联立分析" in line:
                self._result_text.tag_add("header", idx, idx_end)
            elif "综合决策" in line or "【" in line and "】" in line:
                self._result_text.tag_add("decision", idx, idx_end)
            elif line.strip().startswith("[") and "]" in line:
                self._result_text.tag_add("section", idx, idx_end)
            elif "买点信号" in line or "买入" in line or "一买" in line or "二买" in line or "三买" in line:
                if "买点信号" in line:
                    self._result_text.tag_add("section", idx, idx_end)
                else:
                    self._result_text.tag_add("buy", idx, idx_end)
            elif "卖点信号" in line or "卖出" in line or "一卖" in line or "二卖" in line or "三卖" in line:
                if "卖点信号" in line:
                    self._result_text.tag_add("section", idx, idx_end)
                else:
                    self._result_text.tag_add("sell", idx, idx_end)
            elif line.strip().startswith("!") or "禁止操作" in line:
                self._result_text.tag_add("warn", idx, idx_end)
            elif line.strip().startswith("=") or line.strip().startswith("─"):
                self._result_text.tag_add("separator", idx, idx_end)
            elif "最新价格" in line:
                self._result_text.tag_add("price", idx, idx_end)

    def _copy_report(self):
        content = self._result_text.get("1.0", "end-1c")
        if content.strip():
            self.clipboard_clear()
            self.clipboard_append(content)
            self._status_text.set("报告已复制到剪贴板")

    def _generate_html_report(self):
        """在主线程中统一生成HTML图文报告（日线和多级别联立共用）"""
        params = getattr(self, "_chart_params", None)
        if not params:
            return

        wrapped = params["wrapped"]
        code = params["code"]
        name = params["name"]
        start_date = params["start_date"]
        end_date = params["end_date"]
        timeframe_results = params["timeframe_results"]

        self._html_path = None
        self._status_text.set("正在生成K线图表...")

        try:
            from chan_analyzer import generate_html_report
            from chan_lib.visual import chart_to_base64

            chart_images = {}
            for level_key, r in timeframe_results.items():
                df = r.get("_raw_df")
                if df is None or len(df) < 5:
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
                self._html_path = generate_html_report(wrapped, chart_images)
        except Exception:
            import traceback
            print(f"  [HTML报告] 生成异常：{traceback.format_exc()}")

        # 更新界面
        if getattr(self, "_html_path", None):
            # 在报告开头插入醒目的完整报告引导
            guide_lines = [
                "━" * 58,
                "",
                "  \U0001f4ca  完整图文报告（含K线图）已自动在浏览器中打开",
                "",
                "  如未自动打开，请点击右上角按钮  ",
                "      ▶  查看完整报告 (含K线图)  ",
                "",
                "  ──────────────────────────────",
                "  下方为简要文字摘要，完整版含缠论K线图叠加 + 各级别分析",
            ]
            guide_text = "\n".join(guide_lines) + "\n\n"
            self._result_text.insert("1.0", guide_text)
            n = len(guide_lines)
            self._result_text.tag_add("guide", "1.0", f"{n + 2}.0")
            self._result_text.tag_add("guide_btn", f"{n - 3}.0", f"{n - 1}.0")

            self._chart_btn.config(state="normal")
            self._status_text.set("报告生成完成")
            # 自动打开HTML报告
            import webbrowser
            webbrowser.open(f"file:///{self._html_path}")
        else:
            self._status_text.set("报告生成完成（无图表）")

        # 清理临时参数
        self._chart_params = None

    def _view_chart(self):
        """打开HTML报告（含K线图）"""
        html_path = getattr(self, "_html_path", None)
        if not html_path:
            messagebox.showinfo("提示", "报告尚未生成")
            return
        import webbrowser
        webbrowser.open(f"file:///{html_path}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
