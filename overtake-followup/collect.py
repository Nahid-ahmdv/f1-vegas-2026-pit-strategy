"""Cache the race feeds used by the notebook in this folder."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / ".cache"
SESSION_KEY = 11420
FIA_TIMING = "https://www.fia.com/sites/default/files/2026_21_usa_f1_r0_timing_"
OPENF1 = "https://api.openf1.org/v1"
SOURCES = {
    "2026_racesectoranalysis.pdf": FIA_TIMING + "racesectoranalysis_v01.pdf",
    "2026_raceracecontrolmessages.pdf": FIA_TIMING + "raceracecontrolmessages_v01.pdf",
    "11420_race_control.json": f"{OPENF1}/race_control?session_key={SESSION_KEY}",
}
PILOT_CARS = {3, 6, 63}
RESTART_FIRST_LAP = 26
RESTART_LAST_LAP = 38
RESTART_CACHE = "restart_26_38"


def download(url, path, *, retries=2):
    if not path.exists():
        request = Request(url, headers={"User-Agent": "VegasOvertakeAnalysis/1.0"})
        for attempt in range(retries + 1):
            try:
                with urlopen(request, timeout=35) as response:
                    payload = response.read()
                if path.suffix == ".pdf" and not payload.startswith(b"%PDF"):
                    raise ValueError(f"{url} did not return a PDF")
                if path.suffix == ".json" and not isinstance(json.loads(payload), list):
                    raise ValueError(f"{url} did not return a list")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".part")
                temporary.write_bytes(payload)
                temporary.replace(path)
                break
            except (URLError, TimeoutError) as error:
                permanent_http_error = isinstance(error, HTTPError) and error.code not in (
                    408, 429, 500, 502, 503, 504
                )
                if permanent_http_error or attempt == retries:
                    raise
                sleep(2 ** attempt)
    return path


def input_path(name):
    original_cache = ROOT / "data" / "raw" / name
    if original_cache.exists():
        return original_cache
    return download(SOURCES[name], CACHE / name)


def official_timing(*, max_lap=18):
    """Use the parent notebook's reconciled FIA crossing clock and add FIA splits."""
    timing = pd.read_csv(ROOT / "outputs" / "2026_driver_laps.csv").sort_values(
        ["driver_number", "lap"]
    )
    race_start = (
        pd.to_datetime(timing.lap_end_utc, format="mixed", utc=True)
        - pd.to_timedelta(timing.elapsed_at_finish_s, unit="s")
    ).mode().iat[0]
    timing["finish_s"] = timing.elapsed_at_finish_s
    timing["start_s"] = timing.groupby("driver_number").finish_s.shift(fill_value=0)
    timing = timing.loc[timing.lap.le(max_lap)].copy()

    rows = []
    time = r"(\d{1,2}\.\d{3})"
    speed = r"\d{1,3}\.\d"
    pattern = re.compile(rf"^\s*(\d{{1,2}})\s+P?\s*(?:{time}\s+)?{speed}\s+"
                         rf"{time}\s+{speed}\s+{time}\s+(?:{speed}\s+)?"
                         r"(?:\d+:)?\d+:\d{2}(?:\.\d{3})?\s")
    for page in PdfReader(input_path("2026_racesectoranalysis.pdf")).pages:
        car = None
        column = None
        for line in page.extract_text(extraction_mode="layout").splitlines():
            if header := re.match(r"^\s*(\d{1,2})\s{2,}[A-Z][a-z]", line):
                car = int(header[1])
            lap_headers = list(re.finditer(r"\bLAP\s+TIME\b", line))
            if len(lap_headers) == 2:
                column = lap_headers[1].start() - 3
            if car is None or column is None:
                continue
            for side in (line[:column], line[column:]):
                entry = pattern.match(side + " ")
                if entry and 1 <= int(entry[1]) <= max_lap:
                    lap = int(entry[1])
                    rows.append((car, lap, *(float(s) if s else np.nan for s in entry.groups()[1:])))
    splits = pd.DataFrame(rows, columns=["driver_number", "lap", "s1_s", "s2_s", "s3_s"])
    timing = timing.merge(splits, how="left", on=["driver_number", "lap"], validate="one_to_one")
    timed = timing[["s1_s", "s2_s", "s3_s"]].notna().all(axis=1)
    if not np.allclose(
        timing.loc[timed, ["s1_s", "s2_s", "s3_s"]].sum(axis=1), timing.loc[timed, "lap_time_s"],
        rtol=0, atol=0.002,
    ):
        raise ValueError("FIA sector times do not add up to the FIA lap times")
    timing["s2_crossing_s"] = timing.finish_s - timing.s3_s
    timing["s1_crossing_s"] = timing.s2_crossing_s - timing.s2_s
    return timing, race_start


def restart_laps(timing):
    """Retain any car-lap overlapping the leaders' restart window."""
    first_start = timing.loc[timing.lap.eq(RESTART_FIRST_LAP - 1), "finish_s"].min()
    last_finish = timing.loc[timing.lap.eq(RESTART_LAST_LAP), "finish_s"].min()
    return timing.loc[
        timing.finish_s.ge(first_start - 3) & timing.start_s.le(last_finish + 3)
    ].copy()


def position_and_car_bounds(timing, max_lap):
    selected = timing.loc[timing.lap.le(max_lap)]
    if max_lap == 2:
        selected = selected.loc[selected.driver_number.isin(PILOT_CARS)]
    return (selected.groupby("driver_number").agg(start_s=("start_s", "min"),
                                                  finish_s=("finish_s", "max")))


def normalized_samples(data, endpoint, car, race_start):
    fields = ["x", "y"] if endpoint == "location" else ["speed"]
    frame = data.rename(columns={"Date": "date", "X": "x", "Y": "y", "Speed": "speed"})
    timestamps = pd.to_datetime(frame.date, format="mixed", utc=True)
    samples = frame[fields].assign(t_s=(timestamps - race_start).dt.total_seconds(), car=car)
    if endpoint == "location":
        samples[fields] = samples[fields] / 10  # both providers report tenths of a metre
    return samples[["car", "t_s", *fields]].dropna().drop_duplicates("t_s").sort_values("t_s")


def collect_fastf1(bounds, race_start):
    import fastf1

    cache = CACHE / "fastf1"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(cache))
    session = fastf1.get_session(2026, "Las Vegas", "R", backend="fastf1")
    session.load(laps=True, telemetry=True, weather=False, messages=True)
    if session.laps is None or session.laps.empty or "3" not in session.pos_data or "3" not in session.car_data:
        raise RuntimeError("FastF1 did not return Verstappen's race timing, position and car feeds")
    tables = {}
    for endpoint, raw in [("location", session.pos_data), ("car_data", session.car_data)]:
        pieces = []
        for car, row in bounds.iterrows():
            if str(car) in raw:
                frame = normalized_samples(raw[str(car)], endpoint, car, race_start)
                pieces.append(frame.loc[frame.t_s.between(row.start_s - 4, row.finish_s + 4)])
        tables[endpoint] = pd.concat(pieces, ignore_index=True)
    feed_timing = session.laps[["DriverNumber", "LapNumber", "LapTime", "Sector1Time",
                                "Sector2Time", "Sector3Time"]].copy()
    for column in feed_timing.columns[2:]:
        feed_timing[column] = feed_timing[column].dt.total_seconds()
    feed_timing.to_csv(CACHE / "fastf1_laps.csv", index=False)
    return tables, {"source": "FastF1", "version": fastf1.__version__, "api_path": session.api_path}


def collect_openf1(bounds, race_start, cache_key):
    def fetch(task):
        endpoint, car, row = task
        start = race_start + pd.Timedelta(seconds=row.start_s - 4)
        end = race_start + pd.Timedelta(seconds=row.finish_s + 4)
        params = urlencode({"session_key": SESSION_KEY, "driver_number": car,
                            "date>": start.isoformat(), "date<": end.isoformat()})
        url = f"{OPENF1}/{endpoint}?{params}"
        path = CACHE / "openf1" / f"{cache_key}_{endpoint}_{car}.json"
        try:
            data = pd.DataFrame(json.loads(download(url, path).read_bytes()))
        except (URLError, TimeoutError) as error:
            return endpoint, None, {"url": url, "error": str(error)}
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        info = {"url": url, "sha256": digest, "rows": len(data)}
        if data.empty:
            return endpoint, None, info
        if not data.session_key.eq(SESSION_KEY).all() or not data.driver_number.eq(car).all():
            raise ValueError(f"Incorrect OpenF1 session or driver in {path}")
        return endpoint, normalized_samples(data, endpoint, car, race_start), info

    tasks = [(ep, car, row) for ep in ("location", "car_data") for car, row in bounds.iterrows()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(fetch, tasks))
    tables = {}
    for endpoint in ("location", "car_data"):
        pieces = [frame for ep, frame, _ in responses if ep == endpoint and frame is not None]
        tables[endpoint] = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
            columns=["car", "t_s", *(["x", "y"] if endpoint == "location" else ["speed"])]
        )
    return tables, {"source": "OpenF1", "session_key": SESSION_KEY,
                    "requests": [info for _, _, info in responses]}


def collect(max_lap, source, *, restart=False):
    # Keep library caches local, including any matplotlib cache created on import.
    CACHE.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
    timing, race_start = official_timing(max_lap=max_lap)
    input_path("11420_race_control.json")
    input_path("2026_raceracecontrolmessages.pdf")
    bounds = position_and_car_bounds(restart_laps(timing) if restart else timing, max_lap)
    cache_key = RESTART_CACHE if restart else str(max_lap)
    fastf1_error = None
    if source != "openf1":
        try:
            tables, metadata = collect_fastf1(bounds, race_start)
        except Exception as error:
            if source == "fastf1":
                raise
            fastf1_error = f"{type(error).__name__}: {error}"
            print(f"FastF1 attempt failed: {fastf1_error}", flush=True)
    if source == "openf1" or fastf1_error:
        tables, metadata = collect_openf1(bounds, race_start, cache_key)
        metadata["fastf1_error"] = fastf1_error
    metadata["race_start_utc"] = race_start.isoformat()
    metadata["sector_analysis_sha256"] = hashlib.sha256(
        input_path("2026_racesectoranalysis.pdf").read_bytes()
    ).hexdigest()
    metadata["rows"] = {name: len(table) for name, table in tables.items()}
    for endpoint, table in tables.items():
        table.to_csv(CACHE / f"{cache_key}_{endpoint}.csv.gz", index=False, float_format="%.6f")
    (CACHE / f"{cache_key}_inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metadata["source"], metadata["rows"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "fastf1", "openf1"], default="auto")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lap2", action="store_true", help="Fetch only the lap 2 pilot and lap 1 anchors.")
    group.add_argument("--restart", action="store_true", help="Fetch the separate laps 26–38 restart window.")
    args = parser.parse_args()
    max_lap = RESTART_LAST_LAP if args.restart else (2 if args.lap2 else 18)
    collect(max_lap, args.source, restart=args.restart)
