#!/usr/bin/env python3
"""Build a skiver train_context_error_models model-config JSON from a
comma-separated list of component strings.

Usage: build_model_config.py "AdditiveContext(5),AdditiveContext(7)"
Writes {"models": [{"id": ..., "components": ...}, ...]} to stdout.
"""

import json
import re
import sys


def main() -> int:
    raw = sys.argv[1] if len(sys.argv) > 1 else ""
    specs = []
    for i, comp in enumerate(c.strip() for c in raw.split(",") if c.strip()):
        slug = re.sub(r"[^0-9A-Za-z]+", "", comp).lower()
        specs.append({"id": f"m{i}_{slug}", "components": comp})
    if not specs:
        sys.exit("No model components provided")
    json.dump({"models": specs}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
