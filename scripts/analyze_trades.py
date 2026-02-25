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
    ap.add_argument("--path", default="data/trades.jsonl")
    ap.add_argument("--ticker", default="ALL")
    args = ap.parse_args()

    trades = []
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

            action = str(r.get("action", "")).upper()
            if action not in ("BUY", "SELL"):
                continue

            ts = r.get("ts")
            dt = parse_dt(ts) if isinstance(ts, str) else None

            price = r.get("price", None)
            total = r.get("total_score", None)
            strength = r.get("strength", None)

            try:
                price = float(price)
            except Exception:
                price = None
            try:
                total = float(total)
            except Exception:
                total = None
            try:
                strength = float(strength)
            except Exception:
                strength = None

            trades.append({
                "dt": dt,
                "ticker": t,
                "action": action,
                "price": price,
                "total": total,
                "strength": strength,
            })

    if not trades:
        print("No trades found. (No BUY/SELL yet, or wrong path/ticker)")
        return

    by_ticker = defaultdict(list)
    for tr in trades:
        by_ticker[tr["ticker"]].append(tr)

    print(f"=== TRADES SUMMARY ({args.path}) ===")
    print(f"total trades: {len(trades)}")

    for t in sorted(by_ticker.keys()):
        lst = by_ticker[t]
        buys = [x for x in lst if x["action"] == "BUY"]
        sells = [x for x in lst if x["action"] == "SELL"]

        totals = [x["total"] for x in lst if x["total"] is not None]
        strengths = [x["strength"] for x in lst if x["strength"] is not None]

        print(f"\n[{t}] trades={len(lst)} (BUY={len(buys)}, SELL={len(sells)})")

        if totals:
            a = np.array(totals, dtype=float)
            print(f"  total_score: mean={a.mean():.4f} p50={np.percentile(a,50):.4f} "
                  f"p90={np.percentile(a,90):.4f} min={a.min():.4f} max={a.max():.4f}")
        if strengths:
            a = np.array(strengths, dtype=float)
            print(f"  strength:    mean={a.mean():.4f} p50={np.percentile(a,50):.4f} "
                  f"p90={np.percentile(a,90):.4f} min={a.min():.4f} max={a.max():.4f}")

        # 최근 3개
        lst_sorted = sorted([x for x in lst if x["dt"] is not None], key=lambda x: x["dt"])
        tail = lst_sorted[-3:] if len(lst_sorted) >= 3 else lst_sorted
        if tail:
            print("  last trades:")
            for x in tail:
                dt_s = x["dt"].isoformat() if x["dt"] else "N/A"
                print(f"   - {dt_s} {x['action']} @ {x['price']} total={x['total']} strength={x['strength']}")

    print("\nNext step (optional):")
    print("- 체결 후 다음 N분/다음날 가격으로 수익률을 계산해 '신호 품질' 평가로 확장 가능")


if __name__ == "__main__":
    main()
