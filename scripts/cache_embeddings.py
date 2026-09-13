"""Cache the CLIP image embedding for every image in the corpus.

`app/vision.py` already computes a 512-d L2-normalised embedding for every
photo it tags and then throws it away. Everything trained in `app/perception/`
learns on that vector, so it is worth computing once for the whole corpus and
keeping: the alternative is a full CLIP pass on every experiment.

Writes `data/metadata/embeddings.npz` with two aligned arrays, `image_id`
(str) and `emb` (float32, N x 512). Resumable - images already present in the
file are not recomputed, so an interrupted run costs only what it had not
reached.

    .venv/bin/python scripts/cache_embeddings.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, vision  # noqa: E402

OUT = config.META / "embeddings.npz"
BATCH = 64


def existing() -> dict[str, np.ndarray]:
    if not OUT.exists():
        return {}
    z = np.load(OUT, allow_pickle=False)
    return dict(zip(z["image_id"].tolist(), z["emb"]))


def main() -> int:
    df = pd.read_csv(config.IMAGES_CSV)
    have = existing()
    todo = [(r.image_id, config.DATA / r.path) for r in df.itertuples()
            if r.image_id not in have]
    print(f"{len(df)} images in corpus, {len(have)} already cached, {len(todo)} to do")
    if not todo:
        print("nothing to do")
        return 0

    tagger = vision.clip()
    t0 = time.time()
    done = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        pils, ids = [], []
        for image_id, path in chunk:
            try:
                pils.append(Image.open(path))
                ids.append(image_id)
            except Exception as exc:                      # noqa: BLE001
                print(f"  skip {image_id}: {exc}")
        if not pils:
            continue
        have.update(zip(ids, embed(tagger, pils)))
        done += len(ids)
        if i % (BATCH * 10) == 0 or i + BATCH >= len(todo):
            rate = done / max(1e-6, time.time() - t0)
            print(f"  {done}/{len(todo)}  {rate:.0f} img/s")

    ids = sorted(have)
    np.savez_compressed(OUT, image_id=np.array(ids),
                        emb=np.stack([have[i] for i in ids]).astype(np.float32))
    print(f"wrote {OUT}  ({len(ids)} embeddings, {OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


def embed(tagger, pils: list) -> np.ndarray:
    """The image half of ClipTagger.tag, without the text banks.

    Delegates rather than duplicates: this used to reach into
    `tagger.preprocess` and `tagger.model` from outside, so a change to the
    preprocessing in app/vision.py would silently leave the cached corpus
    embeddings in a different space from the live ones.
    """
    return tagger.embed(pils)


if __name__ == "__main__":
    raise SystemExit(main())
