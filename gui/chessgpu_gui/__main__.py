import sys

from .app import main

if __name__ == "__main__":  # (worker processes import this module too)
    sys.exit(main())
