"""Manual Tcl/Tk collection for the GUI executable."""

import sys
from pathlib import Path

python_root = Path(sys.base_prefix)
tcl_root = python_root / "tcl"
dll_root = python_root / "DLLs"

datas = [
    (str(tcl_root / "tcl8.6"), "_tcl_data"),
    (str(tcl_root / "tk8.6"), "_tk_data"),
    (str(tcl_root / "tcl8"), "tcl8"),
    (str(tcl_root / "dde1.4"), "dde1.4"),
    (str(tcl_root / "reg1.3"), "reg1.3"),
]

binaries = [
    (str(dll_root / "_tkinter.pyd"), "."),
    (str(dll_root / "tcl86t.dll"), "."),
    (str(dll_root / "tk86t.dll"), "."),
]
