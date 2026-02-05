import argparse
import collections
import os

from dataset_ds import _read_events_table, _resolve_subject_files


def _top_counts(values, k=12):
    c = collections.Counter(values)
    return c.most_common(k)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect ds003825 BIDS events.tsv columns and value distributions.")
    ap.add_argument("--bids-root", required=True, help="BIDS root (contains dataset_description.json)")
    ap.add_argument("--subject", required=True, help="Subject id, e.g. sub-01")
    ap.add_argument("--show-rows", type=int, default=3, help="Print first N raw rows")
    ap.add_argument("--topk", type=int, default=12, help="Top-K value counts to show per key")
    args = ap.parse_args()

    bids_root = args.bids_root
    subject = args.subject
    if not os.path.isdir(bids_root):
        raise SystemExit(f"Not a directory: {bids_root}")

    _, events_path = _resolve_subject_files(bids_root, subject)
    rows = _read_events_table(events_path)
    if not rows:
        print(f"[events] empty: {events_path}")
        return 0

    keys = sorted({k for r in rows for k in r.keys()})
    print(f"[events] path={events_path}")
    print(f"[events] rows={len(rows)} keys={keys}")

    if args.show_rows > 0:
        print("[events] first_rows:")
        for i, r in enumerate(rows[: args.show_rows]):
            print(f"  [{i}] {r}")

    # Common columns in ds003825:
    # - onset (seconds)
    # - sample (optional)
    # - objectnumber (0..1853)
    # - object (concept string)
    # - istarget (0/1)
    # - trial_type / trigger / event / value ... (varies)
    focus_keys = [
        "trial_type",
        "event",
        "trigger",
        "value",
        "stim_type",
        "istarget",
        "objectnumber",
        "object",
    ]
    for k in focus_keys:
        if k not in keys:
            continue
        vals = [r.get(k, "") for r in rows]
        print(f"[events] {k}: top{args.topk}={_top_counts(vals, k=args.topk)}")

    # Also show how many rows look like "stimulus" rows by objectnumber presence.
    def _is_intlike(x: str) -> bool:
        try:
            int(float(str(x).strip()))
            return True
        except Exception:
            return False

    objnum_present = sum(1 for r in rows if _is_intlike(r.get("objectnumber", "")))
    istarget_present = sum(1 for r in rows if r.get("istarget", "").strip() != "")
    print(f"[events] rows_with_objectnumber={objnum_present}/{len(rows)} rows_with_istarget={istarget_present}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

