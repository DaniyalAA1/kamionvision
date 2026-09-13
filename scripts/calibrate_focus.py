"""Derive FOCUS_SCENERY_RATIO from the corpus, and measure what it buys.

The subject of a photograph is what the photographer focused on. A vehicle box
markedly less sharp than the frame around it is the yard behind the subject.
This is the only subject rule that does not consult the CLIP view tag, which
matters because 36.8% of the corpus is tagged into an exterior view and the
four exterior classes carry median confidences of 0.34 to 0.55 - they are where
a zero-shot classifier puts the frames it cannot place.

Two distributions:
  POSITIVE  boxes on confident whole-vehicle frames. These ARE the subject, so
            the threshold must almost never suppress them.
  NEGATIVE  boxes under the part-view area floor on confident part views. These
            are the failure the rule exists to catch.

The cut goes where false suppression on POSITIVE is under FALSE_SUPPRESS_MAX.
"""
import json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from app import gate, subject

FALSE_SUPPRESS_MAX = 0.05
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def main() -> None:
    df = pd.read_csv("data/metadata/images.csv")
    o = df[df.variant == "original"]
    random.seed(17)
    rows = o.sample(min(N, len(o)), random_state=17)
    paths = [Path("data") / p for p in rows.path]
    paths = [p for p in paths if p.exists()]

    pos, neg, off_tag = [], [], []
    B = 40
    for i in range(0, len(paths), B):
        checks, _ = gate.inspect(paths[i:i + B])
        for c in checks:
            for cand in subject.candidates(c):
                fr = cand.focus_ratio
                if (c.view in subject.WHOLE_VEHICLE_VIEWS and c.view_conf >= 0.50
                        and cand.area_frac >= 0.12):
                    pos.append(fr)
                elif (c.view in subject.TRUCK_PART_VIEWS and c.view_conf >= 0.50
                      and cand.area_frac < subject.part_view_subject_area()):
                    neg.append(fr)
                # the frames the view rule CANNOT see: not a confident part view,
                # yet carrying a small unfocused box
                elif cand.area_frac < subject.part_view_subject_area():
                    off_tag.append(fr)

    pos, neg, off_tag = np.array(pos), np.array(neg), np.array(off_tag)
    print(f"frames scanned {len(paths)}   positive {len(pos)}  negative {len(neg)}  "
          f"off-tag {len(off_tag)}")
    for name, a in (("POSITIVE (whole-vehicle subjects)", pos),
                    ("NEGATIVE (small boxes on part views)", neg),
                    ("OFF-TAG (small boxes the view rule misses)", off_tag)):
        if len(a):
            q = np.percentile(a, [5, 25, 50, 75, 95])
            print(f"  {name:<44} q05 {q[0]:5.2f}  q25 {q[1]:5.2f}  med {q[2]:5.2f}  "
                  f"q75 {q[3]:5.2f}  q95 {q[4]:5.2f}")

    print(f"\n  {'cut':>5} {'false-suppress POS':>20} {'caught NEG':>12} {'caught OFF-TAG':>16}")
    chosen = None
    for cut in np.arange(0.30, 1.05, 0.05):
        fs = float((pos < cut).mean()) if len(pos) else 0.0
        cn = float((neg < cut).mean()) if len(neg) else 0.0
        co = float((off_tag < cut).mean()) if len(off_tag) else 0.0
        star = ""
        if fs <= FALSE_SUPPRESS_MAX:
            chosen = round(float(cut), 2)
            star = "  <-"
        print(f"  {cut:5.2f} {fs:19.1%} {cn:11.1%} {co:15.1%}{star}")

    print(f"\n  FOCUS_SCENERY_RATIO = {chosen}  "
          f"(largest cut with false suppression <= {FALSE_SUPPRESS_MAX:.0%})")
    print(f"  shipped value is {subject.FOCUS_SCENERY_RATIO}")
    out = Path("data/metadata/focus_calibration.json")
    out.write_text(json.dumps({
        "method": "largest cut whose false-suppression rate on confident "
                  "whole-vehicle subject boxes stays under 5%",
        "false_suppress_max": FALSE_SUPPRESS_MAX,
        "frames": len(paths),
        "n_positive": len(pos), "n_negative": len(neg), "n_off_tag": len(off_tag),
        "recommended": chosen,
        "shipped": subject.FOCUS_SCENERY_RATIO,
        "caught_negative_at_recommended":
            round(float((neg < (chosen or 0)).mean()), 4) if len(neg) else None,
        "caught_off_tag_at_recommended":
            round(float((off_tag < (chosen or 0)).mean()), 4) if len(off_tag) else None,
    }, indent=2) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
