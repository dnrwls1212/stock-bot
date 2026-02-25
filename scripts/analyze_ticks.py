import argparse
import json
from datetime import datetime
from collections import defaultdict

import numpy as np


def parse_dt(s: str):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/ticks.jsonl")
    ap.add_argument("--ticker", default="ALL", help="e.g. NVDA, or ALL")
    ap.add_argument("--regular_only", action="store_true",
                    help="정규장(대략 22:30~05:00 KST)만 필터(간이)")
    args = ap.parse_args()

    scores = defaultdict(list)

    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            t = str(r.get("ticker", "")).upper()
            if not t:
                continue

            if args.ticker != "ALL" and t != args.ticker.upper():
                continue

            ts = r.get("ts")
            dt = parse_dt(ts) if isinstance(ts, str) else None

            # 간이 정규장 필터(한국시간 기준)
            if args.regular_only and dt is not None:
                h = dt.hour
                m = dt.minute
                # 22:30 ~ 23:59 or 00:00 ~ 05:00
                in_regular = ((h == 22 and m >= 30) or (h >= 23) or (h <= 5))
                if not in_regular:
                    continue

            sc = r.get("total_score", None)
            if sc is None:
                continue
            try:
                sc = float(sc)
            except Exception:
                continue

            scores[t].append(sc)

    if not scores:
        print("No records matched. (maybe wrong path / filters?)")
        return

    def summarize(arr):
        a = np.array(arr, dtype=float)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=0)),
            "min": float(a.min()),
            "p01": float(np.percentile(a, 1)),
            "p05": float(np.percentile(a, 5)),
            "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }

    print(f"=== TICK SCORE SUMMARY ({args.path}) ===")
    for t in sorted(scores.keys()):
        s = summarize(scores[t])
        print(f"\n[{t}] n={s['n']}")
        print(f"  mean={s['mean']:.4f} std={s['std']:.4f} min={s['min']:.4f} max={s['max']:.4f}")
        print(f"  p01={s['p01']:.4f} p05={s['p05']:.4f} p10={s['p10']:.4f} p50={s['p50']:.4f} "
              f"p90={s['p90']:.4f} p95={s['p95']:.4f} p99={s['p99']:.4f}")

        # 추천 임계값(거래 빈도 조절용)
        buy_p95 = s["p95"]
        sell_p05 = s["p05"]
        buy_p90 = s["p90"]
        sell_p10 = s["p10"]

        print("  -> Suggested thresholds (starter):")
        print(f"     conservative: BUY_TH≈p95({buy_p95:.4f}), SELL_TH≈p05({sell_p05:.4f})")
        print(f"     more active : BUY_TH≈p90({buy_p90:.4f}), SELL_TH≈p10({sell_p10:.4f})")

    print("\nTip:")
    print("- 먼저 정규장만(--regular_only)으로 보고 임계값 잡는 걸 추천")
    print("- HOLD만 나오면 p95->p90으로 완화, 너무 잦으면 p90->p95/p99로 강화")


if __name__ == "__main__":
    main()
