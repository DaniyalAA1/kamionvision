"""Select the US subset to download.

The user asked for a Turkey-heavy corpus, so the US side is a sample rather than
the full ~500-tractor eligible pool.

Allocation is proportional by model year, not round-robin across strata: a
condition/valuation dataset needs the real age mix, and equal-weighting the
strata would over-represent the handful of 2002-2015 units and under-represent
the 2021-2023 bulk. A per-dealer cap keeps any single SelecTrucks centre from
dominating, since photography style is dealer-specific.
"""

import pathlib as _pathlib
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import paths as P
import collections
import json
import math
import random

SRC = P.listing_meta("us_selectrucks")
DST = P.listing_meta("us_selectrucks_sample")
TARGET = 150
MIN_IMAGES = 8
MAX_PER_DEALER = 12

def main():
    random.seed(20260912)
    records = [json.loads(l) for l in open(SRC, encoding="utf-8")]

    eligible = [r for r in records
                if r.get("vpic_body_class") == "Truck-Tractor"
                and r.get("n_images", 0) >= MIN_IMAGES
                and r.get("price") and r.get("mileage_mi")]
    print(f"eligible tractors (>={MIN_IMAGES} photos, priced, with mileage): {len(eligible)}")

    by_year = collections.defaultdict(list)
    for r in eligible:
        by_year[r["year"]].append(r)
    for v in by_year.values():
        random.shuffle(v)

    # Proportional allocation, largest-remainder, with a floor of 1 per year present.
    quota = {y: max(1, math.floor(TARGET * len(v) / len(eligible))) for y, v in by_year.items()}
    picked, per_dealer = [], collections.Counter()
    for year in sorted(by_year, reverse=True):
        for r in by_year[year]:
            if len([p for p in picked if p["year"] == year]) >= quota[year]:
                break
            if per_dealer[r.get("dealer_name")] >= MAX_PER_DEALER:
                continue
            picked.append(r)
            per_dealer[r.get("dealer_name")] += 1

    # Top up to TARGET from whatever remains, still respecting the dealer cap.
    chosen = {id(r) for r in picked}
    for r in sorted(eligible, key=lambda x: -x["n_images"]):
        if len(picked) >= TARGET:
            break
        if id(r) not in chosen and per_dealer[r.get("dealer_name")] < MAX_PER_DEALER:
            picked.append(r)
            chosen.add(id(r))
            per_dealer[r.get("dealer_name")] += 1

    with open(DST, "w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"sampled {len(picked)} listings, {sum(r['n_images'] for r in picked)} images -> {DST}")
    print("by make:", dict(collections.Counter(r["make"] for r in picked).most_common()))
    print("by year:", dict(sorted(collections.Counter(r["year"] for r in picked).items())))
    print(f"dealers: {len(per_dealer)} distinct, max per dealer = {max(per_dealer.values())}")


if __name__ == "__main__":
    main()
