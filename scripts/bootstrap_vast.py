"""Prepare a Vast.ai container; default to a read-only preparation plan."""

from hccr.ops.vast_bootstrap import main

if __name__ == "__main__":
    raise SystemExit(main())
