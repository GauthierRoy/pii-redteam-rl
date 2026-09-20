"""Test package: put src/ on sys.path so `pii_redteam` imports without install."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
