"""Parse the official reports and reconcile them with the OpenF1 timing feed."""

from dataclasses import dataclass
from functools import lru_cache
import re

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from pypdf import PdfReader

from .sources import FP3_2026, RACE_2025, RACE_2026, RAW, openf1

KEY = ["driver_number", "lap"]
LAP_TIME = r"\d+:\d{2}\.\d{3}"
TIME_OF_DAY = r"\d{2}:\d{2}:\d{2}"
VEGAS_TZ = "America/Los_Angeles"


def seconds(text):
    minutes, remainder = text.split(":")
    return 60 * int(minutes) + float(remainder)


@lru_cache
def pdf_pages(name):
    return [page.extract_text(extraction_mode="layout") for page in PdfReader(RAW / name).pages]


def fia_lap_analysis(name):
    """The FIA prints one driver's laps across each pair of TIME columns."""
    rows = []
    for page in pdf_pages(name):
        lines = page.splitlines()
        headers = [i for i, line in enumerate(lines) if re.search(r"\b(?:LAP|NO)\s+TIME", line)]
        ends = [header - 2 for header in headers[1:]] + [len(lines)]
        for header, end in zip(headers, ends, strict=True):
            time_columns = [m.end() for m in re.finditer(r"\b(?:LAP|NO)\s+TIME\b", lines[header])]
            drivers = [int(car) for car in re.findall(r"\b(\d{1,2})\s+[A-Z][a-z]", lines[header - 2])]
            if len(time_columns) != 2 * len(drivers):
                raise ValueError(f"Unexpected driver columns in {name}")
            for line in lines[header + 1 : end]:
                for entry in re.finditer(rf"\b(\d{{1,2}})\s+(P\s+)?({TIME_OF_DAY}|{LAP_TIME})\b", line):
                    lap, pit, time = entry.groups()
                    column = min(range(len(time_columns)), key=lambda i: abs(time_columns[i] - entry.end()))
                    clock_time = ":" in time.split(":", 1)[1]
                    rows.append({
                        "driver_number": drivers[column // 2],
                        "lap": int(lap),
                        "fia_lap_analysis_s": np.nan if clock_time else seconds(time),
                        "fia_first_crossing_local": time if clock_time else None,
                        "fia_lap_analysis_pit_in": bool(pit),
                    })
    return pd.DataFrame(rows).set_index(KEY, verify_integrity=True).sort_index().reset_index()


def fia_history():
    rows = []
    for page in pdf_pages("2026_racehistorychart.pdf"):
        lines = page.splitlines()
        header = next(i for i, line in enumerate(lines) if re.search(r"\bLAP\s+\d+\s+GAP", line))
        columns = list(re.finditer(r"\bLAP\s+(\d+)\s+GAP\s+TIME", lines[header]))
        for i, column in enumerate(columns):
            end = columns[i + 1].start() if i + 1 < len(columns) else None
            position = 0
            for line in lines[header + 1 :]:
                data = re.fullmatch(
                    rf"\s*(\d{{1,2}})\s+(?:(PIT|\d+\.\d{{3}})\s+)?({LAP_TIME})\s*",
                    line[column.start() : end],
                )
                if data is None:
                    continue
                driver, gap, time = data.groups()
                position += 1
                rows.append({
                    "driver_number": int(driver),
                    "fia_history_leader_lap": int(column[1]),
                    "fia_history_row": position,
                    "fia_history_gap_s": float(gap) if gap and gap != "PIT" else (0.0 if position == 1 else np.nan),
                    "fia_history_lap_s": seconds(time),
                    "fia_history_pit_in": gap == "PIT",
                })
    # Lapped cars may appear twice in a leader column, or skip one.
    history = pd.DataFrame(rows)
    history["lap"] = history.groupby("driver_number").cumcount() + 1
    return history.sort_values(KEY).reset_index(drop=True)


def fia_pit_summary():
    pattern = rf"^[ \t]*(\d{{1,2}})\s+.+?\s+(\d{{1,2}})\s+({TIME_OF_DAY})\s+(\d+)\s+(\d+\.\d{{3}})\s+(\d+\.\d{{3}})[ \t]*$"
    return pd.DataFrame(
        [(int(driver), int(lap), int(stop), float(time), clock)
         for driver, lap, clock, stop, time, _ in re.findall(pattern, pdf_pages("2026_racepitstopsummary.pdf")[0], re.M)],
        columns=[*KEY, "stop_number", "official_lane_s", "official_pit_in_local"],
    )


def f1_pit_summary_2025():
    page = BeautifulSoup((RAW / "2025_f1_pits.html").read_bytes(), "html.parser")
    rows = []
    for row in page.find("table").select("tbody tr"):
        stop, car, name, _, lap, clock, duration, _ = (cell.get_text(" ", strip=True) for cell in row.find_all("td"))
        rows.append((int(car), int(lap), int(stop), float(duration), clock, name.split()[-1]))
    return pd.DataFrame(rows, columns=[*KEY, "stop_number", "official_lane_s", "official_pit_in_local", "driver"])


def load_laps(session):
    return (
        pd.DataFrame(openf1(session, "laps"))
        .rename(columns={
            "lap_number": "lap", "lap_duration": "openf1_lap_s",
            "date_start": "openf1_start_utc", "is_pit_out_lap": "openf1_pit_out",
        })
        .assign(openf1_start_utc=lambda data: pd.to_datetime(data.openf1_start_utc, format="mixed", utc=True))
        .set_index(KEY, verify_integrity=True).sort_index().reset_index()
    )


def load_pits(session):
    return (
        pd.DataFrame(openf1(session, "pit"))
        .rename(columns={"lap_number": "lap", "lane_duration": "openf1_lane_s", "stop_duration": "stationary_s"})
        .assign(openf1_pit_timestamp_utc=lambda data: pd.to_datetime(data.date, format="mixed", utc=True))
        [[*KEY, "openf1_lane_s", "stationary_s", "openf1_pit_timestamp_utc"]]
    )


def tyre_laps(session):
    rows = []
    for stint in openf1(session, "stints"):
        for lap in range(stint["lap_start"], stint["lap_end"] + 1):
            age = stint["tyre_age_at_start"] + lap - stint["lap_start"]
            rows.append({
                "driver_number": stint["driver_number"], "lap": lap,
                "stint": stint["stint_number"], "compound": stint["compound"],
                "tyre_age_start_laps": age, "tyre_age_end_laps": age + 1,
                "set_age_at_stint_start_laps": stint["tyre_age_at_start"],
            })
    return pd.DataFrame(rows).set_index(KEY, verify_integrity=True).sort_index().reset_index()


@dataclass(frozen=True)
class Neutralization:
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    leader_lap_start: int
    leader_lap_end: int


def neutralizations(session):
    start_messages = {
        "SAFETY CAR DEPLOYED": "SC", "VIRTUAL SAFETY CAR DEPLOYED": "VSC",
    }
    periods = []
    start = None
    for event in sorted(openf1(session, "race_control"), key=lambda row: row["date"]):
        message = event["message"]
        if message in start_messages:
            if start is not None:
                raise ValueError("Overlapping race neutralizations")
            start = event
        elif start is not None and message == "TRACK CLEAR":
            periods.append(Neutralization(
                start_messages[start["message"]], pd.Timestamp(start["date"]), pd.Timestamp(event["date"]),
                start["lap_number"], event["lap_number"],
            ))
            start = None
    if start is not None:
        raise ValueError("Race control did not close a neutralization")
    return periods


def verify_race_control(year, periods):
    """Match deployment and clearance minute by minute in FIA local time."""
    text = "\n".join(pdf_pages(f"{year}_raceracecontrolmessages.pdf"))
    for period in periods:
        message = "SAFETY CAR DEPLOYED" if period.kind == "SC" else "VIRTUAL SAFETY CAR DEPLOYED"
        for timestamp, label in [(period.start, message), (period.end, "TRACK CLEAR")]:
            minute = timestamp.tz_convert(VEGAS_TZ).strftime("%H:%M")
            if not re.search(rf"\b{minute}\s+{label}\b", text):
                raise ValueError(f"The FIA did not confirm {label} at {minute} in {year}")


def _assign_status(frame, start_column, end_column, periods):
    status = pd.Series("GREEN", index=frame.index)
    for period in periods:
        overlap = frame[start_column].lt(period.end) & frame[end_column].gt(period.start)
        status.loc[overlap] = period.kind
    return status


def _with_stops(frame, stops):
    pit_keys = stops[KEY].assign(pit_in=True)
    out_keys = stops[KEY].assign(lap=lambda data: data.lap + 1, pit_out=True)
    return (
        frame.merge(pit_keys, on=KEY, how="left", validate="one_to_one")
        .merge(out_keys, on=KEY, how="left", validate="one_to_one")
        .assign(pit_in=lambda data: data.pit_in.eq(True), pit_out=lambda data: data.pit_out.eq(True))
    )


def _verify_pits(official, openf1_pits):
    check = official.merge(openf1_pits, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not check["_merge"].eq("both").all():
        raise ValueError("Official pit-stop summary and OpenF1 disagree on stop laps")
    check["lane_delta_openf1_s"] = check.openf1_lane_s - check.official_lane_s
    return check.drop(columns="_merge")


def race_2026():
    laps = load_laps(RACE_2026)
    analysis = fia_lap_analysis("2026_racelapanalysis.pdf")
    history = fia_history()
    frame = analysis.merge(history, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not frame["_merge"].eq("both").all():
        raise ValueError("FIA History Chart and Lap Analysis disagree on which laps were completed")
    comparable = frame.fia_lap_analysis_s.notna()
    if not np.allclose(frame.loc[comparable, "fia_lap_analysis_s"], frame.loc[comparable, "fia_history_lap_s"], atol=0.001):
        raise ValueError("FIA History Chart and Lap Analysis disagree on lap times")
    frame = frame.drop(columns="_merge").merge(
        laps[[*KEY, "openf1_lap_s", "openf1_pit_out", "openf1_start_utc"]],
        on=KEY, how="left", validate="one_to_one",
    )
    frame["lap_time_s"] = frame.fia_lap_analysis_s.fillna(frame.fia_history_lap_s)
    frame["lap_time_delta_openf1_s"] = frame.openf1_lap_s - frame.lap_time_s
    frame = frame.sort_values(KEY).reset_index(drop=True)
    frame["elapsed_at_finish_s"] = frame.groupby("driver_number").lap_time_s.cumsum().round(3)
    t0 = laps.loc[laps.lap.eq(1), "openf1_start_utc"].mode().iat[0]
    frame["lap_end_utc"] = t0 + pd.to_timedelta(frame.elapsed_at_finish_s, unit="s")
    frame["lap_start_utc"] = frame.lap_end_utc - pd.to_timedelta(frame.lap_time_s, unit="s")
    frame["gap_to_leader_s"] = (
        frame.elapsed_at_finish_s - frame.groupby("lap").elapsed_at_finish_s.transform("min")
    ).round(3)
    frame["gap_ahead_s"] = (
        frame.sort_values(["lap", "elapsed_at_finish_s"]).groupby("lap").elapsed_at_finish_s.diff().round(3)
    )
    leader_crossings = frame.groupby("lap").elapsed_at_finish_s.min().to_numpy()
    frame["leader_lap_at_finish"] = np.searchsorted(leader_crossings, frame.elapsed_at_finish_s, side="right")
    frame["laps_down_at_finish"] = frame.leader_lap_at_finish - frame.lap
    # OpenF1 lap-start timestamps offer a separate check on FIA crossing gaps.
    next_crossing = laps.assign(lap=laps.lap - 1)[[*KEY, "openf1_start_utc"]]
    next_crossing = next_crossing.rename(columns={"openf1_start_utc": "openf1_crossing_utc"})
    frame = frame.merge(next_crossing, on=KEY, how="left", validate="one_to_one")
    frame["openf1_clock_gap_s"] = (
        frame.openf1_crossing_utc - frame.groupby("lap").openf1_crossing_utc.transform("min")
    ).dt.total_seconds()
    stops = _verify_pits(fia_pit_summary(), load_pits(RACE_2026))
    if stops.lane_delta_openf1_s.abs().gt(0.001).any():
        raise ValueError("FIA and OpenF1 disagree on a 2026 pit-lane duration")
    frame = _with_stops(frame, stops)
    if not frame.pit_in.eq(frame.fia_lap_analysis_pit_in).all():
        raise ValueError("FIA stop summary and lap analysis disagree on a pit-in flag")
    if not frame.pit_in.eq(frame.fia_history_pit_in).all():
        raise ValueError("FIA stop summary and History Chart disagree on a pit-in flag")
    periods = neutralizations(RACE_2026)
    verify_race_control(2026, periods)
    frame["track_status"] = _assign_status(frame, "lap_start_utc", "lap_end_utc", periods)
    frame["safety_car"] = frame.track_status.eq("SC")
    drivers = pd.DataFrame(openf1(RACE_2026, "drivers"))
    frame = frame.merge(tyre_laps(RACE_2026), on=KEY, how="left", validate="one_to_one").merge(
        drivers[["driver_number", "name_acronym", "full_name", "team_name"]].rename(
            columns={"name_acronym": "driver"}),
        on="driver_number", validate="many_to_one",
    )
    if frame.compound.isna().any():
        raise ValueError("OpenF1 stints are missing for a completed lap")
    return frame, stops, periods


def historical_pit_losses():
    laps = load_laps(RACE_2025)
    official = f1_pit_summary_2025()
    stops = _verify_pits(official, load_pits(RACE_2025))
    laps = _with_stops(laps, stops).merge(tyre_laps(RACE_2025), on=KEY, how="left", validate="one_to_one")
    periods = neutralizations(RACE_2025)
    verify_race_control(2025, periods)
    laps["openf1_end_utc"] = laps.openf1_start_utc + pd.to_timedelta(laps.openf1_lap_s, unit="s")
    laps["track_status"] = _assign_status(laps, "openf1_start_utc", "openf1_end_utc", periods)
    open_laps = laps.set_index(KEY)
    fia_laps = fia_lap_analysis("2025_racelapanalysis.pdf").set_index(KEY)
    rows = []
    for stop in stops.loc[stops.lap.between(20, 29)].itertuples(index=False):
        driver, lap = stop.driver_number, stop.lap
        before = list(range(lap - 3, lap))
        settled = list(range(lap + 3, lap + 6))
        window = [*before, lap, lap + 1, lap + 2, *settled]
        feed = open_laps.loc[driver].reindex(window)
        verified = fia_laps.loc[driver].reindex(window)
        if (
            feed.openf1_lap_s.isna().any() or verified.fia_lap_analysis_s.isna().any()
            or not feed.track_status.eq("GREEN").all()
            or not feed.index[feed.pit_in].tolist() == [lap]
            or not feed.index[feed.openf1_pit_out].tolist() == [lap + 1]
            or not verified.index[verified.fia_lap_analysis_pit_in].tolist() == [lap]
        ):
            raise ValueError(f"Incomplete or contaminated 2025 window for {stop.driver} on lap {lap}")
        times = feed.openf1_lap_s
        fia = verified.fia_lap_analysis_s
        pre, post = times.loc[before].median(), times.loc[settled].median()
        rows.append({
            "driver": stop.driver, "driver_number": driver, "stop_lap": lap,
            "compound_in": feed.at[lap, "compound"], "compound_out": feed.at[lap + 1, "compound"],
            "pit_in_s": times.at[lap], "pit_out_s": times.at[lap + 1],
            "pre_median_s": pre, "post_median_s": post,
            "effective_loss_s": times.loc[[lap, lap + 1]].sum() - pre - post,
            "fia_effective_loss_s": fia.loc[[lap, lap + 1]].sum() - fia.loc[before].median() - fia.loc[settled].median(),
            "official_lane_s": stop.official_lane_s,
            "openf1_lane_s": stop.openf1_lane_s,
            "first_flying_vs_settled_s": times.at[lap + 2] - post,
        })
    estimates = pd.DataFrame(rows).sort_values(["stop_lap", "driver"]).reset_index(drop=True)
    return estimates, stops


def fp3_2026():
    laps = load_laps(FP3_2026).merge(
        fia_lap_analysis("2026_fp3_laps.pdf"), on=KEY, how="left", validate="one_to_one",
    )
    deleted = set()
    for event in openf1(FP3_2026, "race_control"):
        match = re.search(r"\bCAR (\d+) .*\bDELETED\b.*\bLAP (\d+)\b", event["message"])
        if match:
            deleted.add(tuple(map(int, match.groups())))
    laps["deleted"] = laps.set_index(KEY).index.isin(deleted)
    return laps.merge(tyre_laps(FP3_2026), on=KEY, how="left", validate="one_to_one")


def fp3_push_pairs(fp3):
    """Descriptive first and second representative flying laps on the same set."""
    valid = fp3.loc[
        fp3.compound.isin(["SOFT", "MEDIUM"]) & fp3.lap.ge(2)
        & fp3.openf1_lap_s.notna() & fp3.fia_lap_analysis_s.notna()
        & ~fp3.openf1_pit_out & fp3.fia_lap_analysis_pit_in.eq(False) & ~fp3.deleted
        & (fp3.openf1_lap_s - fp3.fia_lap_analysis_s).abs().le(0.001)
    ]
    pairs = []
    for (driver, stint), group in valid.groupby(["driver_number", "stint"]):
        if len(group) < 3 or not group.set_age_at_stint_start_laps.eq(0).all():
            continue
        target = group.openf1_lap_s.min() + 2.0
        first_two = group.loc[group.openf1_lap_s.le(target)].sort_values("lap").head(2)
        if len(first_two) < 2:
            continue
        first, second = first_two.itertuples(index=False)
        if second.lap - first.lap <= 4 and first.tyre_age_start_laps <= 4:
            pairs.append({
                "driver_number": driver, "stint": stint, "compound": first.compound,
                "first_push_lap": first.lap, "second_push_lap": second.lap,
                "first_push_s": first.openf1_lap_s, "second_push_s": second.openf1_lap_s,
                "second_minus_first_s": second.openf1_lap_s - first.openf1_lap_s,
            })
    return pd.DataFrame(pairs)
