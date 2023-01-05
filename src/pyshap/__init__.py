import os
import sys

try:
    SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    SRC_DIR = os.path.dirname(os.getcwd())

sys.path.append(SRC_DIR)