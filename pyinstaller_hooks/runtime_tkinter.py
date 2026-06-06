"""Point frozen tkinter at the Tcl/Tk data bundled by local hooks."""

import os
import sys

base_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
os.environ.setdefault("TCL_LIBRARY", os.path.join(base_dir, "_tcl_data"))
os.environ.setdefault("TK_LIBRARY", os.path.join(base_dir, "_tk_data"))
