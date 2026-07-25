#!/usr/bin/env python3
"""Pick the minimum-AIC maximum-likelihood model from a skiver
context_model_aic.csv and print its model_id to stdout.

Usage: pick_best_model.py context_model_aic.csv <platform>
"""

import csv
import sys


def main() -> int:
    csv_path = sys.argv[1]
    best_id, best_aic = None, float("inf")
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("inference") != "maximum_likelihood":
                continue
            aic = row.get("aic")
            if not aic:
                continue
            value = float(aic)
            if value < best_aic:
                best_aic, best_id = value, row["model_id"]
    if best_id is None:
        sys.exit(f"No maximum_likelihood model with an AIC found in {csv_path}")
    sys.stdout.write(best_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
