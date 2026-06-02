"""Enable ``python -m qex <command> ...`` as an alias for the ``qex`` CLI."""

from qex.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
