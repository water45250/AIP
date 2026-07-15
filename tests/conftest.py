"""确保仓库根目录（含 src 包）在 sys.path 上，便于 import src.aip_core。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
