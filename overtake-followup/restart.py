"""Field-order and timing checks for the race laps after the restart."""

import numpy as np
import pandas as pd

from collect import RESTART_FIRST_LAP, RESTART_LAST_LAP
from crossings import (
    DETECTION_GAP_S,
    DETECTION_M,
    FIA_KNOTS,
    PRIMARY,
    cross_gate,
    summarize_laps,
)

FOCUS = ("VER", "RUS", "ANT")


def consensus(values):
    seen = set(values)
    return next(iter(seen)) if len(seen) == 1 else "varies"


def race_order(timing, lap):
    return (timing.loc[timing.fia_history_leader_lap.eq(lap)].sort_values("fia_history_row")
            .drop_duplicates("driver").driver.tolist())


def detection_snapshot(estimates, lap, column, messages):
    """Sort the same-lap field and include nearby different-lap crossings for adjacency."""
    field = estimates.loc[estimates.lap.eq(lap)].dropna(subset=[column]).sort_values(column)
    first = field.head(3)
    indexed = field.set_index("driver")
    other_laps = estimates.loc[estimates.lap.ne(lap)].dropna(subset=[column])
    out = {"top3": "/".join(first.driver)}
    for driver in FOCUS:
        key = driver.lower()
        clock = indexed.loc[driver, column] if driver in indexed.index else np.nan
        preceding = field.loc[field[column].lt(clock)]
        lapped = other_laps.loc[
            other_laps.driver.ne(driver) & other_laps[column].ge(clock - 2) & other_laps[column].lt(clock)
        ]
        preceding = pd.concat([preceding, lapped]).nlargest(1, column)
        ahead = preceding.driver.iat[0] if len(preceding) else ""
        gap = clock - preceding[column].iat[0] if len(preceding) else np.nan
        out[f"{key}_ahead"] = ahead
        out[f"{key}_gap"] = gap
        if not np.isfinite(clock):
            status = "unknown"
        else:
            prior = messages.loc[messages.t_s.le(clock) & messages.message.str.startswith("OVERTAKE")]
            status = prior.message.iat[-1].split()[-1].lower() if len(prior) else "unknown"
        out[f"{key}_state"] = status
        out[f"{key}_within_1s"] = (
            "unknown" if not np.isfinite(clock) else
            "yes" if 0 < gap < DETECTION_GAP_S else "no"
        )
    return out


def restart_summary(estimates, timing, variants, messages, race_start):
    """Keep nominal crossings, marker sensitivity, and race-control status distinct."""
    restart = estimates.loc[estimates.lap.between(RESTART_FIRST_LAP - 1, RESTART_LAST_LAP)]
    guarded, pad = summarize_laps(restart, variants, messages)
    guarded = guarded.set_index("lap")
    marker_columns = [PRIMARY, *[name for name in variants if name.startswith("shift ")]]
    rows = []
    for lap in range(RESTART_FIRST_LAP, RESTART_LAST_LAP + 1):
        sampled = restart.loc[restart.lap.eq(lap)].set_index("driver")
        ver, rus, ant = (sampled.loc[code] for code in FOCUS)
        start, finish = (race_order(timing, number) for number in (lap - 1, lap))
        snapshots = [detection_snapshot(restart, lap, column, messages) for column in marker_columns]
        nominal = snapshots[0]
        top = nominal["top3"].split("/")
        pair = (rus[marker_columns] - ver[marker_columns]).astype(float)
        kimi = (ant[marker_columns] - ver[marker_columns]).astype(float)
        expanded = (rus[variants] - ver[variants]).astype(float)
        guard = guarded.loc[lap]

        def repeats(s):
            if finish[0] != "VER":
                return "no"
            return "yes" if (s["top3"].split("/")[0] == "RUS" and s["ver_ahead"] == "RUS"
                             and s["ver_within_1s"] == "yes" and s["ver_state"] == "enabled") else "no"

        exact = consensus(repeats(s) for s in snapshots)
        reverse = consensus(
            "yes" if (finish[0] == "RUS" and s["top3"].split("/")[0] == "VER"
                      and s["rus_ahead"] == "VER" and s["rus_within_1s"] == "yes"
                      and s["rus_state"] == "enabled") else "no"
            for s in snapshots
        )
        row = {
            "lap": lap,
            "fia_start_order": "/".join(start),
            "fia_finish_order": "/".join(finish),
            "fia_start_top3": "/".join(start[:3]),
            "fia_finish_top3": "/".join(finish[:3]),
            "fia_start_rus_minus_ver_s": rus.start_s - ver.start_s,
            "fia_finish_rus_minus_ver_s": rus.finish_s - ver.finish_s,
            "fia_start_ant_minus_ver_s": ant.start_s - ver.start_s,
            "fia_finish_ant_minus_ver_s": ant.finish_s - ver.finish_s,
            "det_p1": top[0],
            "det_p2": top[1],
            "det_p3": top[2],
            "det_p2_to_leader_s": sampled.at[top[1], PRIMARY] - sampled.at[top[0], PRIMARY],
            "det_p3_to_leader_s": sampled.at[top[2], PRIMARY] - sampled.at[top[0], PRIMARY],
            "det_top3_across_pm40_m": " | ".join(dict.fromkeys(s["top3"] for s in snapshots)),
            "det_rus_minus_ver_s": pair[PRIMARY],
            "det_rus_minus_ver_min_pm40_s": pair.min(),
            "det_rus_minus_ver_max_pm40_s": pair.max(),
            "rus_ver_order_pm40": consensus("unknown" if pd.isna(x) else "RUS" if x < 0 else
                                            "VER" if x > 0 else "tie" for x in pair),
            "rus_ver_within_1s_pm40": consensus("unknown" if pd.isna(x) else
                                               "yes" if abs(x) < DETECTION_GAP_S else "no" for x in pair),
            "det_rus_minus_ver_min_all_variants_s": expanded.min(),
            "det_rus_minus_ver_max_all_variants_s": expanded.max(),
            "det_ant_minus_ver_s": kimi[PRIMARY],
            "det_ant_minus_ver_min_pm40_s": kimi.min(),
            "det_ant_minus_ver_max_pm40_s": kimi.max(),
            "ver_detection_utc_est": race_start + pd.Timedelta(seconds=ver[PRIMARY]),
            "overtake_enabled_at_ver_detection": nominal["ver_state"] == "enabled",
            "opening_pattern_pm40": exact,
            "opening_pattern_with_clock_guard": guard.pattern if exact != "varies" else "uncertain",
            "reverse_pattern_pm40": reverse,
            "other_driver_ahead_then_ver_led": (
                nominal["ver_ahead"] if finish[0] == "VER" and nominal["ver_ahead"] not in ("", "RUS")
                and nominal["ver_within_1s"] == "yes" and nominal["ver_state"] == "enabled" else ""
            ),
            "ver_eligibility_with_clock_guard": guard.overtake_eligibility,
            "ver_clock_uncertainty_notes": guard.uncertainty_notes,
            "deployment_evidence": "unobserved",
        }
        for driver in FOCUS:
            key = driver.lower()
            row[f"driver_directly_ahead_of_{key}_det"] = nominal[f"{key}_ahead"]
            row[f"{key}_gap_to_ahead_det_s"] = nominal[f"{key}_gap"]
            row[f"{key}_ahead_across_pm40_m"] = " | ".join(dict.fromkeys(
                s[f"{key}_ahead"] or "none" for s in snapshots
            ))
            row[f"{key}_within_1s_of_ahead_pm40"] = consensus(s[f"{key}_within_1s"] for s in snapshots)
            row[f"{key}_estimated_eligible_pm40"] = consensus(
                s[f"{key}_within_1s"] if s[f"{key}_state"] == "enabled" else
                "no" if s[f"{key}_state"] == "disabled" else "unknown" for s in snapshots
            )
        rows.append(row)
    return pd.DataFrame(rows), pad


def compare_reference(summary, supplied):
    """Reconcile the supplied preliminary table with the independent recomputation."""
    actual = summary.set_index("lap")
    if supplied.lap.duplicated().any() or set(supplied.lap) != set(actual.index):
        raise ValueError("Supplied restart table must contain one row per race lap 26–38")
    ref = supplied.set_index("lap").reindex(actual.index)
    result = pd.DataFrame(index=actual.index)
    result["top3_match"] = actual[["det_p1", "det_p2", "det_p3"]].eq(
        ref[["detection_leader_est", "detection_second_est", "detection_third_est"]]
        .set_axis(["det_p1", "det_p2", "det_p3"], axis=1)
    ).all(axis=1)
    for driver in ("ver", "rus"):
        result[f"{driver}_ahead_match"] = actual[f"driver_directly_ahead_of_{driver}_det"].eq(
            ref[f"driver_directly_ahead_of_{driver}_at_detection_est"].fillna("")
        )
    result["supplied_gap_s"] = ref.detection_rus_crossing_minus_ver_s_est
    result["recomputed_gap_s"] = actual.det_rus_minus_ver_s
    result["recomputed_minus_supplied_s"] = result.recomputed_gap_s - result.supplied_gap_s
    low = ref.detection_rus_minus_ver_min_s_across_marker_and_reference_sensitivity
    high = ref.detection_rus_minus_ver_max_s_across_marker_and_reference_sensitivity
    result["nominal_inside_supplied_range"] = result.recomputed_gap_s.between(low, high)
    result["sensitivity_ranges_overlap"] = (
        actual.det_rus_minus_ver_min_all_variants_s.le(high)
        & actual.det_rus_minus_ver_max_all_variants_s.ge(low)
    )
    result["ver_crossing_time_difference_s"] = (
        actual.ver_detection_utc_est - pd.to_datetime(ref.ver_detection_time_utc_est, utc=True)
    ).dt.total_seconds()
    result["fia_finish_match"] = (
        actual.fia_finish_top3.str.split("/").str[0].eq(ref.fia_finish_leader)
        & np.isclose(actual.fia_finish_rus_minus_ver_s, ref.fia_finish_rus_crossing_minus_ver_s, atol=0.001)
    )
    result["enabled_match"] = actual.overtake_enabled_at_ver_detection.eq(ref.overtake_enabled_at_detection)
    result["opening_pattern_match_pm40"] = actual.opening_pattern_pm40.eq(
        ref.ver_behind_rus_at_detection_within_1s_then_leads_at_finish.map({True: "yes", False: "no"})
    )
    result["reverse_pattern_match_pm40"] = actual.reverse_pattern_pm40.eq(
        ref.rus_behind_ver_at_detection_within_1s_then_leads_at_finish.map({True: "yes", False: "no"})
    )
    return result


def opening_sector(estimates, position, course, laps=(29, 32, 38)):
    """Estimate VER/RUS order on a coarse first-sector grid pinned to FIA splits."""
    rows = []
    for number in laps:
        field = estimates.loc[estimates.lap.eq(number)].set_index("driver")
        distances = np.r_[np.arange(0, FIA_KNOTS[1], 25), FIA_KNOTS[1]]
        elapsed = np.interp(course.from_fia(distances), course.distance, course.time) - course.time[0]
        crossings = {}
        for driver in ("VER", "RUS"):
            car = field.loc[driver]
            samples = position.loc[position.car.eq(car.driver_number)]
            raw_start = cross_gate(course, course.knots[0], samples, car.start_s, 2)[0]
            raw_gates = np.array([
                cross_gate(course, course.from_fia(distance), samples, car.start_s + offset, 3)[0]
                for distance, offset in zip(distances[1:-1], elapsed[1:-1], strict=True)
            ])
            fraction = (raw_gates - raw_start) / (car.raw_s1_s - raw_start)
            fraction[(fraction < 0) | (fraction > 1)] = np.nan
            crossings[driver] = np.r_[
                car.start_s, car.start_s + fraction * (car.s1_crossing_s - car.start_s),
                car.s1_crossing_s,
            ]
        rows.extend(zip([number] * len(distances), distances,
                        crossings["RUS"] - crossings["VER"], strict=True))
    return pd.DataFrame(rows, columns=["lap", "fia_distance_m", "rus_minus_ver_s"])


def bracket_retakes(samples, pad):
    """Bracket the final opening-sector sign change, with a timing-margin check."""
    rows = []
    for lap, sector in samples.groupby("lap"):
        scan = sector.dropna().sort_values("fia_distance_m")
        if scan.rus_minus_ver_s.iat[0] >= 0 or scan.rus_minus_ver_s.iat[-1] <= 0:
            continue
        last_rus = scan.loc[scan.rus_minus_ver_s.lt(0), "fia_distance_m"].max()
        after = scan.loc[scan.fia_distance_m.gt(last_rus)]
        rows.append({
            "lap": lap,
            "fia_opening_gap_s": scan.rus_minus_ver_s.iat[0],
            "fia_s1_gap_s": scan.rus_minus_ver_s.iat[-1],
            "nominal_from_m": last_rus,
            "nominal_to_m": after.fia_distance_m.min(),
            "guarded_from_m": scan.loc[scan.fia_distance_m.le(last_rus)
                                      & scan.rus_minus_ver_s.lt(-pad), "fia_distance_m"].max(),
            "guarded_to_m": after.loc[after.rus_minus_ver_s.gt(pad), "fia_distance_m"].min(),
        })
    return pd.DataFrame(rows)
