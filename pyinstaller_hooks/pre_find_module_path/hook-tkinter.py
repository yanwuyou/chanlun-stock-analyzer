"""Keep tkinter discoverable even when PyInstaller's Tcl probe fails.

The local Python 3.13 install can import tkinter, but PyInstaller's isolated
Tcl probe returns unavailable and the bundled pre-find hook excludes tkinter.
This project ships Tcl/Tk explicitly via hook-_tkinter.py and a runtime hook.
"""


def pre_find_module_path(hook_api):
    return
