"""Estimate the time each car crossed a common physical line."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from pypdf import PdfReader

from collect import CACHE, input_path

FIA_KNOTS = np.array([0.0, 1693.0, 3428.0, 6201.0])
DETECTION_M = 5299.0
ACTIVATION_M = 5400.0
DETECTION_GAP_S = 1.0
DETECTION_OFFSETS_M = tuple(range(-40, 41, 5))
MAX_SAMPLE_GAP_S = 1.0
PRIMARY = "VER geometry, FIA splits"
MIN_CLOCK_PAD_S = 0.050


@dataclass
class Course:
    """Distance along one reference position trace, in metres."""

    time: np.ndarray
    xy: np.ndarray
    distance: np.ndarray
    knots: np.ndarray

    def point(self, distance):
        return np.array([np.interp(distance, self.distance, axis) for axis in self.xy.T]).T

    def from_fia(self, distance):
        return np.interp(distance, FIA_KNOTS, self.knots)

    def direction(self, distance):
        tangent = self.point(distance + 8) - self.point(distance - 8)
        return tangent / np.linalg.norm(tangent)


def build_course(lap, position):
    time = position.t_s.to_numpy()
    boundaries = np.array([lap.start_s, lap.s1_crossing_s, lap.s2_crossing_s, lap.finish_s])
    if np.any(np.diff(boundaries) <= 0) or not (time[0] < boundaries[0] < boundaries[-1] < time[-1]):
        raise ValueError("Reference lap needs all FIA splits and position samples across the full lap")
    trace = np.r_[boundaries[0], time[(time > boundaries[0]) & (time < boundaries[-1])], boundaries[-1]]
    xy = np.column_stack([np.interp(trace, time, position[axis]) for axis in ("x", "y")])
    distance = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    knots = np.interp(boundaries, trace, distance)
    if not 6000 < distance[-1] < 6350 or np.linalg.norm(xy[-1] - xy[0]) > 30:
        raise ValueError("Reference geometry failed the lap length or XY closure check")
    return Course(trace, xy, distance, knots)


def cross_gate(course, distance, samples, expected_s, radius_s):
    """Interpolate a forward crossing of the transverse line through the reference XY."""
    window = samples.loc[samples.t_s.between(expected_s - radius_s, expected_s + radius_s)]
    t = window.t_s.to_numpy()
    xy = window[["x", "y"]].to_numpy() - course.point(distance)
    ahead = xy @ course.direction(distance)
    indices = np.flatnonzero((ahead[:-1] <= 0) & (ahead[1:] > 0))
    if not len(indices):
        return np.nan, np.nan, np.nan
    sample_gap = t[indices + 1] - t[indices]
    fraction = -ahead[indices] / (ahead[indices + 1] - ahead[indices])
    at_gate = xy[indices] + fraction[:, None] * (xy[indices + 1] - xy[indices])
    crossing = t[indices] + fraction * sample_gap
    valid = np.linalg.norm(at_gate, axis=1) < 25
    if valid.sum() != 1:
        return np.nan, np.nan, np.nan
    bracket = sample_gap[valid].item()
    value = crossing[valid].item() if bracket <= MAX_SAMPLE_GAP_S else np.nan
    return value, bracket, t[indices[valid]].item()


def align_to_s3(crossing_s, raw_s2_s, raw_finish_s, lap):
    # Pin the measured transit time to the two official sector-three boundaries.
    if lap.pit_in or not (raw_s2_s <= crossing_s <= raw_finish_s):
        return np.nan
    return lap.s2_crossing_s + lap.s3_s * (crossing_s - raw_s2_s) / (raw_finish_s - raw_s2_s)


def speed_crossing(samples, raw_s2_s, raw_finish_s, lap):
    """Diagnose how an independent speed-integrated axis places the line."""
    if samples.empty or not np.isfinite([raw_s2_s, raw_finish_s]).all() or lap.pit_in:
        return np.nan
    time = samples.t_s.to_numpy()
    if not time[0] <= raw_s2_s < raw_finish_s <= time[-1]:
        return np.nan
    t = np.r_[raw_s2_s, time[(time > raw_s2_s) & (time < raw_finish_s)], raw_finish_s]
    if np.diff(t).max() > MAX_SAMPLE_GAP_S:
        return np.nan
    speed = np.interp(t, time, samples.speed) / 3.6
    path = np.r_[0, np.cumsum(np.diff(t) * (speed[:-1] + speed[1:]) / 2)]
    s2, finish = FIA_KNOTS[2:]
    if not 0.95 * (finish - s2) < path[-1] < 1.05 * (finish - s2):
        return np.nan
    return np.interp(path[-1] * (DETECTION_M - s2) / (finish - s2), path, t)


def estimate(timing, position, car_data, *, max_lap=18):
    """Return one timing row per car-lap and named joint sensitivity variants."""
    position_by_car = dict(tuple(position.groupby("car")))
    speed_by_car = dict(tuple(car_data.groupby("car")))
    courses = {}
    for driver in ("VER", "RUS"):
        lap = timing.loc[timing.driver.eq(driver) & timing.lap.eq(2)].iloc[0]
        courses[driver] = build_course(lap, position_by_car[lap.driver_number])
    course = courses["VER"]
    # Perturb the line jointly for every car. Also check full-lap scaling and
    # a second racing line under identical FIA sector constraints.
    distances = {
        f"shift {offset:+d} m": course.from_fia(DETECTION_M + offset)
        for offset in DETECTION_OFFSETS_M if offset
    }
    distances[PRIMARY] = course.from_fia(DETECTION_M)
    distances["whole-lap scale"] = course.distance[-1] * DETECTION_M / FIA_KNOTS[-1]
    distances["same metres to finish"] = course.distance[-1] - (FIA_KNOTS[-1] - DETECTION_M)
    variants = [PRIMARY, "shared UTC", *[name for name in distances if name != PRIMARY],
                "RUS geometry"]
    selection = timing.loc[timing.lap.le(max_lap) & timing.driver_number.isin(position_by_car)]
    rows = []
    for lap in selection.itertuples(index=False):
        p = position_by_car[lap.driver_number]
        speed = speed_by_car.get(lap.driver_number, pd.DataFrame())
        data = {}
        bracket_starts = []
        for driver, axis in courses.items():
            clocks = [lap.s1_crossing_s, lap.s2_crossing_s, lap.finish_s]
            raw = [cross_gate(axis, distance, p, clock, 2.2)[0]
                   for distance, clock in zip(axis.knots[1:], clocks, strict=True)]
            s1, s2, finish = raw
            line_distances = distances if driver == "VER" else {"RUS geometry": axis.from_fia(DETECTION_M)}
            for name, distance in line_distances.items():
                reference_transit = axis.time[-1] - np.interp(distance, axis.distance, axis.time)
                # If a car entered the pits, use S2 to time the search. Its
                # pit-lane finish can be many seconds later than our control gate.
                expected = lap.finish_s - reference_transit
                if lap.pit_in:
                    expected = lap.s2_crossing_s + np.interp(distance, axis.distance, axis.time) - np.interp(
                        axis.knots[2], axis.distance, axis.time
                    )
                at_line, bracket, bracket_start = cross_gate(axis, distance, p, expected, 6)
                bracket_starts.append(bracket_start)
                pinned = align_to_s3(at_line, s2, finish, lap)
                data[name] = pinned if np.isfinite(pinned) else at_line
                if name == PRIMARY:
                    activation = axis.from_fia(ACTIVATION_M)
                    at_activation, _, _ = cross_gate(axis, activation, p, expected + 1.7, 6)
                    activation_s = align_to_s3(at_activation, s2, finish, lap)
                    from_speed = speed_crossing(speed, s2, finish, lap)
                    speed_xy = np.array([np.interp(from_speed, p.t_s, p[d]) for d in ("x", "y")])
                    data.update({
                        "shared UTC": at_line,
                        "sector_aligned": np.isfinite(pinned),
                        "detection_bracket_s": bracket,
                        "activation_s": activation_s if np.isfinite(activation_s) else at_activation,
                        "raw_s1_s": s1,
                        "raw_s2_s": s2,
                        "raw_finish_s": finish,
                        "speed_inferred_crossing_s": align_to_s3(from_speed, s2, finish, lap),
                        "speed_inferred_gate_error_m": (speed_xy - axis.point(distance))
                        @ axis.direction(distance),
                    })
        data["earliest_bracket_s"] = min((s for s in bracket_starts if np.isfinite(s)), default=np.nan)
        rows.append({"lap": lap.lap, "driver": lap.driver, **data})
    estimates = timing.loc[timing.lap.le(max_lap)].merge(
        pd.DataFrame(rows), how="left", on=["lap", "driver"], validate="one_to_one"
    )
    return estimates, courses, variants


def clock_checks(estimates):
    """Compare unpinned pair gaps at both sector splits and the FIA control line."""
    anchors = (
        ("S1", "raw_s1_s", "s1_crossing_s"),
        ("S2", "raw_s2_s", "s2_crossing_s"),
        ("Finish", "raw_finish_s", "finish_s"),
    )
    rows = []
    for (lap, driver), car in estimates.loc[estimates.driver.isin(["RUS", "HAD"])].set_index(
        ["lap", "driver"]
    ).iterrows():
        ver = estimates.loc[estimates.lap.eq(lap) & estimates.driver.eq("VER")].iloc[0]
        for landmark, raw, official in anchors:
            signed_raw = car[raw] - ver[raw]
            signed_fia = car[official] - ver[official]
            rows.append((lap, driver, landmark, signed_fia, signed_raw, signed_raw - signed_fia))
    return pd.DataFrame(rows, columns=["lap", "driver", "landmark", "fia_gap_s", "raw_gap_s", "error_s"])


def overtake_messages(race_start):
    logs = json.loads(input_path("11420_race_control.json").read_text())
    messages = pd.DataFrame(
        ({"lap": row["lap_number"], "message": row["message"], "utc": pd.Timestamp(row["date"])}
         for row in logs if row["message"] in {"OVERTAKE ENABLED", "OVERTAKE DISABLED",
                                              "SAFETY CAR DEPLOYED"})
    ).sort_values("utc")
    messages["t_s"] = (messages.utc - race_start).dt.total_seconds()
    report = "\n".join(p.extract_text() for p in PdfReader(input_path("2026_raceracecontrolmessages.pdf")).pages)
    for row in messages.itertuples(index=False):
        local_minute = row.utc.tz_convert("America/Los_Angeles").strftime("%H:%M")
        if f"{local_minute}\n{row.message}" not in report:
            raise ValueError(f"Missing FIA race-control confirmation of {row.message} at {local_minute}")
    return messages


def difference_bounds(later, earlier, variants, pad):
    """Joint variants move the line/clock for all cars together."""
    differences = (later[variants] - earlier[variants]).astype(float).dropna()
    if differences.empty:
        return -np.inf, np.inf
    return differences.min() - pad, differences.max() + pad


def summarize_laps(estimates, variants, messages):
    checks = clock_checks(estimates)
    pad = max(MIN_CLOCK_PAD_S, checks.error_s.abs().quantile(0.95))
    resolved = estimates.dropna(subset=[PRIMARY])
    unresolved = estimates.loc[estimates[PRIMARY].isna()].copy()
    # When GPS cannot resolve a crossing, keep a lower bound from the earliest
    # bracket. The FIA S2 split plus travel at 400 km/h gives another lower bound.
    quickest = (DETECTION_M + min(DETECTION_OFFSETS_M) - FIA_KNOTS[2]) / (400 / 3.6)
    unresolved["earliest_s"] = unresolved[["earliest_bracket_s"]].join(
        (unresolved.s2_crossing_s + quickest).rename("s2_bound")
    ).max(axis=1).fillna(-np.inf)
    rows = []
    for number, lap in estimates.groupby("lap"):
        notes = []
        ver = lap.loc[lap.driver.eq("VER")].iloc[0]
        rus = lap.loc[lap.driver.eq("RUS")].iloc[0]
        finish = lap.loc[lap.finish_s.idxmin()]
        finish_gap = rus.finish_s - ver.finish_s
        if not np.isfinite(ver[PRIMARY]):
            rows.append(dict(lap=number, finish_leader=finish.driver, finish_rus_minus_ver_s=finish_gap,
                             overtake_eligibility="uncertain", pattern="uncertain",
                             uncertainty_notes="No resolvable VER detection crossing"))
            continue
        timed = resolved.loc[resolved.lap.eq(number)].sort_values(PRIMARY)
        # A lapped car can still qualify another car for Overtake. Check other
        # lap numbers around VER's passage on the same physical loop.
        nearby_other_laps = resolved.loc[
            resolved.lap.ne(number) & resolved.driver.ne("VER") & resolved[PRIMARY].sub(ver[PRIMARY]).abs().le(2)
        ]
        candidates = pd.concat([timed.loc[timed.driver.ne("VER")], nearby_other_laps])
        preceding = candidates.loc[candidates[PRIMARY].lt(ver[PRIMARY])].sort_values(PRIMARY)
        ahead = preceding.iloc[-1] if not preceding.empty else None
        gap = rus[PRIMARY] - ver[PRIMARY]
        gap_lo, gap_hi = difference_bounds(rus, ver, variants, pad)
        third_cars = candidates.loc[~(candidates.driver.eq("RUS") & candidates.lap.eq(number))]
        definitely_between = []
        possibly_between = []
        for _, car in third_cars.iterrows():
            lo_rus, hi_rus = difference_bounds(car, rus, variants, pad)
            lo_ver, hi_ver = difference_bounds(car, ver, variants, pad)
            if lo_rus > 0 and hi_ver < 0:
                definitely_between.append(car.driver)
            elif hi_rus > 0 and lo_ver < 0:
                possibly_between.append(car.driver)

        # Only a missing crossing around the race leaders can affect this call.
        missing_in_window = []
        for _, car in unresolved.iterrows():
            if car.earliest_s > ver[variants].max() + pad + 0.5 or car.finish_s < ver[variants].min() - 2:
                continue
            missing_in_window.append(car.driver)
        local_missing = unresolved.loc[unresolved.lap.eq(number), "driver"].tolist()
        if local_missing:
            notes.append("No precise crossing for " + ", ".join(local_missing)
                         + (" in the comparison window" if missing_in_window else " well behind the leaders"))
        if gap_lo <= 0 <= gap_hi:
            notes.append("VER/RUS detection order overlaps a tie")
        elif min(abs(gap_lo), abs(gap_hi)) < 0.05:
            notes.append("Small margin to a tie after the location and timing checks")
        if gap_lo <= -DETECTION_GAP_S <= gap_hi:
            notes.append("RUS-to-VER gap overlaps the 1.0 s threshold")
        if gap_lo > 0 or definitely_between:
            direct = "no"
        elif gap_hi < 0 and not possibly_between and not missing_in_window:
            direct = "yes"
        else:
            direct = "uncertain"
            if possibly_between:
                notes.append("Possible intervening car " + ", ".join(sorted(set(possibly_between))))

        prior = messages.loc[messages.t_s.le(ver["shared UTC"]) & messages.message.str.startswith("OVERTAKE")]
        status = prior.message.iloc[-1].split()[-1].lower() if not prior.empty else "unknown"
        if number == 1:
            notes.append("Standing start. First leader detection enables Overtake under B7.2.2(b)")
        if not (ver.sector_aligned and rus.sector_aligned):
            notes.append("Shared UTC used for at least one of VER or RUS")

        deficits = [difference_bounds(ver, car, variants, pad) for _, car in candidates.iterrows()]
        qualifies = any(lo > 0 and hi < DETECTION_GAP_S for lo, hi in deficits)
        maybe = any(hi > 0 and lo < DETECTION_GAP_S for lo, hi in deficits)
        if status == "enabled" and qualifies:
            eligible = "estimated yes"
        elif status == "disabled" or (not maybe and not missing_in_window):
            eligible = "no"
        else:
            eligible = "uncertain"
            notes.append("VER eligibility varies with the preceding-car gap or Overtake status")

        if finish.driver != "VER" or status == "disabled" or direct == "no" or gap_hi <= -DETECTION_GAP_S:
            pattern = "no"
        elif direct == "yes" and gap_lo > -DETECTION_GAP_S and eligible == "estimated yes":
            pattern = "yes"
        else:
            pattern = "uncertain"

        leader = timed.iloc[0]
        leader_close = any(difference_bounds(other, leader, variants, pad)[0] <= 0
                           for _, other in timed.iloc[1:].iterrows())
        if leader_close or missing_in_window:
            contenders = [
                other.driver for _, other in timed.iloc[1:].iterrows()
                if difference_bounds(other, leader, variants, pad)[0] <= 0
            ] + missing_in_window
            notes.append("Uncertain leader at detection between " + ", ".join([leader.driver, *contenders]))
        rows.append({
            "lap": number,
            "detection_leader": leader.driver,
            "detection_leader_uncertain": bool(leader_close or missing_in_window),
            "detection_rus_minus_ver_s": gap,
            "detection_gap_low_s": gap_lo,
            "detection_gap_high_s": gap_hi,
            "ver_car_ahead_at_detection": ahead.driver if ahead is not None else "",
            "ver_gap_to_car_ahead_s": ver[PRIMARY] - ahead[PRIMARY] if ahead is not None else np.nan,
            "rus_directly_ahead_of_ver": direct,
            "activation_rus_minus_ver_s": rus.activation_s - ver.activation_s,
            "finish_leader": finish.driver,
            "finish_rus_minus_ver_s": finish_gap,
            "overtake_race_control": "first detection, enabled on lap 1" if number == 1 else status,
            "overtake_eligibility": eligible,
            "pattern": pattern,
            "deployment_evidence": "unobserved",
            "uncertainty_notes": ". ".join(notes),
        })
    return pd.DataFrame(rows), pad
