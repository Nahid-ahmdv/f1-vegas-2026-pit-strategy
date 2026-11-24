"""Download the public timing inputs and record the exact bytes used."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "sources.json"

RACE_2026 = 11420
FP3_2026 = 11415
RACE_2025 = 9858
FIA = "https://www.fia.com/sites/default/files"
OPENF1 = "https://api.openf1.org/v1"
F1_PITS = "https://www.formula1.com/en/results/2025/races/1274/las-vegas/pit-stop-summary"

SOURCES = {}
for key, endpoints in {
    RACE_2026: ("sessions", "drivers", "laps", "pit", "stints", "race_control"),
    FP3_2026: ("sessions", "laps", "stints", "race_control"),
    RACE_2025: ("sessions", "laps", "pit", "stints", "race_control"),
}.items():
    for endpoint in endpoints:
        SOURCES[f"{key}_{endpoint}.json"] = f"{OPENF1}/{endpoint}?session_key={key}"
for year, round_no, reports in [
    (2026, 21, ("racehistorychart", "racelapanalysis", "racepitstopsummary", "raceracecontrolmessages")),
    (2025, 22, ("racelapanalysis", "raceracecontrolmessages")),
]:
    for report in reports:
        SOURCES[f"{year}_{report}.pdf"] = f"{FIA}/{year}_{round_no}_usa_f1_r0_timing_{report}_v01.pdf"
SOURCES["2026_fp3_laps.pdf"] = f"{FIA}/2026_21_usa_f1_p3_timing_thirdpracticesessionlaptimes_v01.pdf"
SOURCES["2025_f1_pits.html"] = F1_PITS


def _validate_payload(name, payload):
    if name.endswith(".json"):
        rows = json.loads(payload)
        key = int(name.split("_", 1)[0])
        if not rows or not isinstance(rows, list) or any(row["session_key"] != key for row in rows):
            raise ValueError(f"Expected nonempty OpenF1 records for session {key} in {name}")
    elif name.endswith(".pdf") and not payload.startswith(b"%PDF"):
        raise ValueError(f"Expected a PDF for {name}")
    elif name.endswith(".html") and b"pit-stop-summary" not in payload:
        raise ValueError(f"Expected the Formula 1 pit-stop result page for {name}")


def _download(url):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; VegasPitStrategy/1.0)"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=45) as response:
                return response.read()
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            retry_after = error.headers.get("Retry-After", "")
            sleep(min(60, int(retry_after)) if retry_after.isdigit() else 2 ** (attempt + 1))
        except (URLError, TimeoutError):
            if attempt == 4:
                raise
            sleep(2 ** (attempt + 1))


def _write_atomic(path, payload):
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary_path = Path(output.name)
            output.write(payload)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def fetch(*, refresh=False, seed_dir=None):
    RAW.mkdir(parents=True, exist_ok=True)
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest = {}
    updates = {}
    for name, url in SOURCES.items():
        path = RAW / name
        if seed_dir is not None:
            payload = (seed_dir / name).read_bytes()
        elif refresh or not path.exists():
            payload = _download(url)
            if url.startswith(OPENF1):
                sleep(0.6)
        else:
            payload = path.read_bytes()
        _validate_payload(name, payload)
        digest = hashlib.sha256(payload).hexdigest()
        old = previous.get(name, {})
        if old.get("sha256") and old["sha256"] != digest:
            print(f"{name} changed since the previous snapshot")
        if seed_dir is not None or refresh or not path.exists():
            updates[path] = payload
        fetched_at = old.get("fetched_utc") if old.get("sha256") == digest else None
        manifest[name] = {
            "url": url,
            "sha256": digest,
            "bytes": len(payload),
            "fetched_utc": fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    for path, payload in updates.items():
        _write_atomic(path, payload)
    _write_atomic(MANIFEST, (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def verify_cache():
    if not MANIFEST.exists():
        raise FileNotFoundError("Run `uv run python -m vegas.sources` from the project directory.")
    manifest = json.loads(MANIFEST.read_text())
    for name, url in SOURCES.items():
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run `uv run python -m vegas.sources`.")
        if manifest[name]["url"] != url or hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]["sha256"]:
            raise ValueError(f"Source manifest mismatch for {name}. Restore the cache or fetch again.")
    return manifest


def openf1(key, endpoint):
    return json.loads((RAW / f"{key}_{endpoint}.json").read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Download all inputs again.")
    args = parser.parse_args()
    print(f"Verified and recorded {len(fetch(refresh=args.refresh))} timing inputs in {RAW}")
