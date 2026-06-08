"""Load utterance IDs to exclude from WER evaluation."""

from __future__ import annotations

from pathlib import Path


def load_exclude_utt_ids(
    *,
    utt_ids: list[str] | None = None,
    exclude_file: Path | None = None,
) -> frozenset[str]:
    out: set[str] = set()
    if utt_ids:
        for uid in utt_ids:
            uid = uid.strip()
            if uid:
                out.add(uid)
    if exclude_file is not None:
        path = exclude_file.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Exclude file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(line)
    return frozenset(out)


def partition_rows(
    rows: list[dict],
    exclude_utt_ids: frozenset[str],
) -> tuple[list[dict], list[str]]:
    """Return (included rows, excluded utt_id list in input order)."""
    if not exclude_utt_ids:
        return rows, []
    included: list[dict] = []
    excluded: list[str] = []
    for row in rows:
        utt_id = str(row.get("utt_id", ""))
        if utt_id in exclude_utt_ids:
            excluded.append(utt_id)
        else:
            included.append(row)
    return included, excluded
