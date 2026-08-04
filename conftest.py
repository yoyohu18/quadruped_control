"""把仓库根目录加入 sys.path，使得在任意位置都能 ``import kinematics``。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
