"""模块入口，支持 python -m wind_farm_opt 调用。"""

from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
