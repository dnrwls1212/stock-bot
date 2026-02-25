# src/tools/label_decisions.py
from __future__ import annotations

import os
import argparse

from src.eval.decision_labeler import DecisionLabeler, LabelConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.environ.get("DECISION_LOG_PATH", "data/decisions.jsonl"))
    ap.add_argument("--out", default="data/decisions_labeled.jsonl")
    ap.add_argument("--summary", default="data/decisions_summary.json")
    ap.add_argument("--cache_dir", default=os.environ.get("PRICE_CACHE_DIR", "data/price_cache"))
    ap.add_argument("--horizons", default="1h,1d,3d,7d")
    ap.add_argument("--min_abs_ret", type=float, default=float(os.environ.get("LABEL_MIN_ABS_RET", "0.002")))
    ap.add_argument("--min_conf", type=float, default=float(os.environ.get("LABEL_MIN_CONF", "0.0")))
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    cfg = LabelConfig(
        horizons=horizons,
        min_abs_ret_for_success=args.min_abs_ret,
        require_confidence=(args.min_conf if args.min_conf > 0 else None),
    )

    labeler = DecisionLabeler(cache_dir=args.cache_dir)
    stats = labeler.label_file(
        input_path=args.input,
        output_labeled_path=args.out,
        output_summary_path=args.summary,
        config=cfg,
        overwrite=True,
        max_rows=(args.max_rows if args.max_rows > 0 else None),
    )

    print("[OK] labeled decisions written:")
    print(f" - labeled: {args.out}")
    print(f" - summary: {args.summary}")
    print("Top-level summary:")
    for h, s in stats.get("by_horizon", {}).items():
        print(f"  {h}: n_eval={s.get('n_eval')} avg_ret={s.get('avg_ret'):.4f} win_rate={s.get('win_rate'):.3f}")


if __name__ == "__main__":
    main()