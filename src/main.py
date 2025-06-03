import sys
import os

# Ensure src is in sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.mainGUI import launch_ui

if __name__ == "__main__":
    launch_ui()
