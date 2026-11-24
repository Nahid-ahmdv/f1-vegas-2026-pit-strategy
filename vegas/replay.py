"""A lap-line field projection and a matched-car strategy comparison."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

ANCHOR_LAP = 18
RACE_LAPS = 50
WINDOW = range(22, 29)
LATE_PACE_FROM = 26
REFERENCE = "RUS"
RIVAL = "VER"
TARGETS = ("VER", "RUS", "ANT")
SOFT, MEDIUM = "SOFT", "MEDIUM"


def pre_safety_laps(race, deployment):
    return race.loc[
        race.lap.le(ANCHOR_LAP) & race.lap_end_utc.lt(deployment) & race.track_status.eq("GREEN")
    ].copy()


def seed_field(race, deployment):
    """Use completed pre-SC laps and estimate two missing lap-18 crossings."""
    pre = pre_safety_laps(race, deployment)
    leader_finish = pre.loc[pre.lap.eq(ANCHOR_LAP), "elapsed_at_finish_s"].min()
    pace_laps = pre.loc[pre.lap.between(14, ANCHOR_LAP) & ~pre.pit_in & ~pre.pit_out]
    pace = pace_laps.groupby("driver").lap_time_s.agg(["median", "size"])
    last = pre.sort_values("lap").groupby("driver", as_index=False).tail(1).set_index("driver")
    field = last[["driver_number", "lap", "lap_end_utc", "elapsed_at_finish_s", "compound", "tyre_age_end_laps"]].join(
        pace.rename(columns={"median": "recent_pace_s", "size": "pace_laps"})
    )
    field["included"] = field.lap.ge(ANCHOR_LAP - 1)
    field["lap18_basis"] = np.select(
        [field.lap.eq(ANCHOR_LAP), field.included],
        ["Observed crossing", "Projected from lap 17"],
        default="Lapped at the anchor",
    )
    field["lap18_clock_s"] = (
        field.elapsed_at_finish_s + (ANCHOR_LAP - field.lap) * field.recent_pace_s - leader_finish
    ).where(field.included)
    field["pace_offset_s"] = field.recent_pace_s - field.at[RIVAL, "recent_pace_s"]
    if field.loc[field.included, "pace_laps"].lt(3).any():
        raise ValueError("Too few pre-Safety-Car laps to project an active car")
    return field.sort_values("lap18_clock_s")


def project_field(field, spread):
    """Extrapolate a fraction of each car's pre-SC relative median pace."""
    field = field.loc[field.included]
    laps = np.arange(ANCHOR_LAP, RACE_LAPS + 1)
    clocks = field.lap18_clock_s.to_numpy() + np.outer(
        laps - ANCHOR_LAP, spread * field.pace_offset_s.to_numpy(),
    )
    return pd.DataFrame(clocks, columns=field.index, index=pd.Index(laps, name="lap"))


def _place(clocks, driver, shifted_clock_s):
    opponents = clocks.drop(driver).sort_values(kind="stable")
    ahead = opponents.loc[opponents.le(shifted_clock_s)]
    behind = opponents.loc[opponents.gt(shifted_clock_s)]
    return {
        "position": len(ahead) + 1,
        "rejoin_behind": ahead.index[-1] if len(ahead) else None,
        "gap_to_ahead_s": shifted_clock_s - ahead.iat[-1] if len(ahead) else np.nan,
        "rejoin_ahead_of": behind.index[0] if len(behind) else None,
        "gap_to_behind_s": behind.iat[0] - shifted_clock_s if len(behind) else np.nan,
    }


def observed_gap_rejoins(race, bounds):
    observed = race.loc[
        race.fia_history_leader_lap.eq(ANCHOR_LAP) & race.lap.eq(ANCHOR_LAP) & race.fia_history_gap_s.notna()
    ].set_index("driver").fia_history_gap_s.sort_values()
    rows = []
    for driver in TARGETS:
        for label, loss in bounds.items():
            rows.append({
                "driver": driver, "historical_loss_case": label, "pit_loss_s": loss,
                "observed_lap18_gap_s": observed.at[driver],
                **_place(observed, driver, observed.at[driver] + loss),
            })
    return pd.DataFrame(rows), observed


def projected_rejoins(field, bounds, spreads=(0.0, 0.5, 1.0)):
    rows = []
    for spread in spreads:
        clocks = project_field(field, spread)
        for lap in range(19, max(WINDOW) + 1):
            for driver in TARGETS:
                for label, loss in bounds.items():
                    rows.append({
                        "pit_lap": lap, "driver": driver, "spread": spread,
                        "historical_loss_case": label, "pit_loss_s": loss,
                        **_place(clocks.loc[lap], driver, clocks.at[lap, driver] + loss),
                    })
    return pd.DataFrame(rows)


def observed_stint_trends(race, deployment):
    """Median pairwise slope before the SC, with and without same-lap field pace."""
    pre = pre_safety_laps(race, deployment)
    sample = pre.loc[pre.lap.between(6, ANCHOR_LAP) & pre.stint.eq(1) & ~pre.pit_in & ~pre.pit_out].copy()
    sample["relative_s"] = sample.lap_time_s - sample.groupby("lap").lap_time_s.transform("median")
    rows = []
    for driver, group in sample.groupby("driver"):
        if len(group) < 8:
            continue
        i, j = np.triu_indices(len(group), k=1)
        age = group.tyre_age_start_laps.to_numpy()
        time = group.lap_time_s.to_numpy()
        relative = group.relative_s.to_numpy()
        rows.append({
            "driver": driver, "compound": group.compound.iat[0], "laps": len(group),
            "raw_s_per_lap": np.median((time[j] - time[i]) / (age[j] - age[i])),
            "relative_s_per_lap": np.median((relative[j] - relative[i]) / (age[j] - age[i])),
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Parameters:
    pit_loss_s: float
    field_spread: float = 0.5
    ref_out_warm_s: float = 0.60
    medium_out_warm_s: float = 0.75
    soft_out_warm_s: float = 0.15
    first_flying_fraction: float = 0.35
    medium_deg: float = 0.015
    soft_deg: float = 0.025
    early_soft_adv_s: float = 0.00
    late_soft_adv_s: float = 0.12
    soft_set_age: int = 3
    medium_graining_s: float = 0.0
    close_follow_loss_s: float = 0.50
    peak_tow_gain_s: float = 0.15
    traffic_laps: int = 5


@dataclass(frozen=True)
class Strategy:
    start: str
    pit_lap: int | None
    opening_set_age: int | None = None

    @property
    def finish(self):
        return MEDIUM if self.start == SOFT else SOFT

    @property
    def order(self):
        return f"{self.start.title()} → {self.finish.title()}"


def _warmup(compound, settings):
    return settings.soft_out_warm_s if compound == SOFT else settings.medium_out_warm_s


def _tyre_seconds(laps, plan, settings):
    change = np.zeros(len(laps), dtype=bool) if plan.pit_lap is None else laps > plan.pit_lap
    use_soft = np.where(change, plan.finish == SOFT, plan.start == SOFT)
    set_age = np.where(use_soft, settings.soft_set_age, 0)
    if plan.opening_set_age is not None:
        set_age = np.where(change, set_age, plan.opening_set_age)
    stint_laps = laps - np.where(change, plan.pit_lap or 0, 0) - 1
    age = set_age + stint_laps
    advantage = np.where(laps >= LATE_PACE_FROM, settings.late_soft_adv_s, settings.early_soft_adv_s)
    tyre_cost = np.where(use_soft, settings.soft_deg * age - advantage, settings.medium_deg * age)
    graining = (plan.start == MEDIUM) & ~change & (age >= 20)
    return tyre_cost + graining * settings.medium_graining_s


def _clean_path(base, plan, settings, opening=None):
    """Add the strategy's time delta to the car's projected opening-stint pace."""
    laps = base.index.to_numpy()[1:]
    opening = opening if opening is not None else Strategy(SOFT, None)
    delta = _tyre_seconds(laps, plan, settings) - _tyre_seconds(laps, opening, settings)
    if plan.pit_lap is not None:
        warm = _warmup(plan.finish, settings)
        # Book the effective cost at the rejoin proxy. Only the warm-up
        # adjustment relative to the historical reference is added here.
        delta[laps == plan.pit_lap] += settings.pit_loss_s - settings.ref_out_warm_s + warm
        delta[laps == plan.pit_lap + 2] += settings.first_flying_fraction * warm
    return base + np.r_[0.0, np.cumsum(delta)]


def _traffic(clean, driver, pit_lap, settings):
    """Price proximity on the first flying laps from the clean timing paths."""
    result = pd.Series(0.0, index=clean.index)
    close_laps = 0
    tow_lap_equivalents = 0.0
    if pit_lap is not None:
        for lap in range(pit_lap + 2, min(pit_lap + 2 + settings.traffic_laps, RACE_LAPS + 1)):
            previous = clean.loc[lap - 1]
            gap = _place(previous, driver, previous.at[driver])["gap_to_ahead_s"]
            if pd.isna(gap):
                continue
            close = gap <= 1.0
            tow = float(np.clip((6.0 - gap) / 4.0, 0.0, 1.0))
            close_laps += close
            tow_lap_equivalents += tow
            result.at[lap] = settings.close_follow_loss_s * close - settings.peak_tow_gain_s * tow
    return result, close_laps, tow_lap_equivalents


def run_strategy(field, settings, plan, rival_pit=None):
    baseline = project_field(field, settings.field_spread)
    clean = baseline.copy()
    clean[REFERENCE] = _clean_path(baseline[REFERENCE], plan, settings)
    clean[RIVAL] = _clean_path(baseline[RIVAL], Strategy(SOFT, rival_pit), settings)
    traffic, close, tow = _traffic(clean, REFERENCE, plan.pit_lap, settings)
    rival_traffic, _, _ = _traffic(clean, RIVAL, rival_pit, settings)
    reference = clean[REFERENCE] + traffic.cumsum()
    rival = clean[RIVAL] + rival_traffic.cumsum()
    lap28_rank = _place(clean.loc[28], REFERENCE, reference.at[28])["position"]
    summary = {
        "order": plan.order,
        "pit_lap": plan.pit_lap,
        "rival_pit_lap": rival_pit,
        "lap28_position_proxy": lap28_rank,
        "race_time_delta_vs_early_medium_s": np.nan,
        "lap28_clock_s": reference.at[28],
        "finish_clock_s": reference.at[RACE_LAPS] if plan.pit_lap is not None else np.nan,
        "clean_finish_clock_s": clean.at[RACE_LAPS, REFERENCE] if plan.pit_lap is not None else np.nan,
        "net_traffic_s": traffic.sum(),
        "close_follow_laps": close,
        "tow_lap_equivalents": tow,
        "versus_rival_after_both_outlaps_s": reference.at[rival_pit + 1] - rival.at[rival_pit + 1] if rival_pit else np.nan,
    }
    if plan.pit_lap is not None:
        pit = plan.pit_lap
        summary.update(_place(clean.loc[pit], REFERENCE, clean.at[pit, REFERENCE]))
    return summary


def strategy_scenarios(field, settings):
    rows = []
    for start in (SOFT, MEDIUM):
        for pit in (19, 22, 28, None):
            rows.append(run_strategy(field, settings, Strategy(start, pit)))
        rows.append(run_strategy(field, settings, Strategy(start, 22), rival_pit=23))
    scenarios = pd.DataFrame(rows)
    for rival_case, group in scenarios.groupby(scenarios.rival_pit_lap.notna()):
        early_medium = group.loc[group.order.eq("Soft → Medium") & group.pit_lap.eq(22), "finish_clock_s"].iat[0]
        scenarios.loc[group.index, "race_time_delta_vs_early_medium_s"] = group.finish_clock_s - early_medium
    return scenarios


def clean_late_gain(field, settings, late_pit):
    """Positive seconds favor the later Medium-to-Soft plan."""
    base = project_field(field, settings.field_spread)[REFERENCE]
    early = _clean_path(base, Strategy(SOFT, 22), settings)
    late = _clean_path(base, Strategy(MEDIUM, late_pit), settings)
    return early.at[RACE_LAPS] - late.at[RACE_LAPS]


def break_even(field, settings, late_pit, extra_traffic_s):
    """Required warmed Soft pace advantage per lap after halfway."""
    zero_advantage = clean_late_gain(field, replace(settings, late_soft_adv_s=0.0), late_pit)
    gaining_laps = RACE_LAPS - max(late_pit, LATE_PACE_FROM - 1)
    return (extra_traffic_s - zero_advantage) / gaining_laps


def _schedule_clocks(baseline, opening, plans, settings):
    clean = baseline.copy()
    for driver, plan in plans.items():
        clean[driver] = _clean_path(baseline[driver], plan, settings, opening[driver])

    clocks = clean.copy()
    for driver, plan in plans.items():
        traffic, _, _ = _traffic(clean, driver, plan.pit_lap, settings)
        clocks[driver] += traffic.cumsum()
    return clean, clocks


def cover_stress_test(field, settings, pit_wave):
    """Cross Verstappen's cover timing with a declared schedule of other stops."""
    baseline = project_field(field, settings.field_spread)
    opening = {
        driver: Strategy(row.compound, None, int(row.tyre_age_end_laps - row.lap))
        for driver, row in field.loc[field.included].iterrows()
    }
    excluded = {REFERENCE, RIVAL, "ANT"}
    if any(car not in opening or car in excluded or lap not in WINDOW for car, lap in pit_wave.items()):
        raise ValueError("Pit-wave stops must be in the pit window and belong to another active car")

    early_lap, late_lap = WINDOW.start, WINDOW.stop - 1
    russell = replace(opening[REFERENCE], pit_lap=early_lap)
    antonelli = replace(opening["ANT"], pit_lap=late_lap)
    alt_antonelli = replace(antonelli, pit_lap=early_lap + 1)
    late_reference = Strategy(MEDIUM, late_lap)
    wave = {car: replace(opening[car], pit_lap=lap) for car, lap in pit_wave.items()}
    rows = []

    for pit_wave_active, other_plans in [(False, {}), (True, wave)]:
        for cover_lap in (None, early_lap, early_lap + 1, early_lap + 2):
            opponents = other_plans.copy()
            if cover_lap is not None:
                opponents[RIVAL] = replace(opening[RIVAL], pit_lap=cover_lap)
            early_plans = {**opponents, REFERENCE: russell}
            joint_plans = {**early_plans, "ANT": antonelli}
            clean, clocks = _schedule_clocks(baseline, opening, joint_plans, settings)
            _, early_antonelli_clocks = _schedule_clocks(
                baseline, opening, {**early_plans, "ANT": alt_antonelli}, settings,
            )
            _, early_reference_clocks = _schedule_clocks(baseline, opening, early_plans, settings)
            _, late_reference_clocks = _schedule_clocks(
                baseline, opening, {**opponents, REFERENCE: late_reference}, settings,
            )
            row = {
                "front_runner_pit_wave": pit_wave_active,
                "verstappen_pit_lap": cover_lap,
                "russell_pit_lap": early_lap,
                "antonelli_pit_lap": late_lap,
                "antonelli_alternative_pit_lap": alt_antonelli.pit_lap,
                "both_out_lap": np.nan,
                "russell_lead_after_both_out_s": np.nan,
                "russell_stop_gain_vs_both_staying_out_s": np.nan,
                "russell_stop_gain_from_traffic_s": np.nan,
                "russell_lead_at_lap_28_s": clocks.at[late_lap, RIVAL] - clocks.at[late_lap, REFERENCE],
                "matched_late_soft_gain_s": (
                    early_reference_clocks.at[RACE_LAPS, REFERENCE] - late_reference_clocks.at[RACE_LAPS, REFERENCE]
                ),
                "antonelli_wait_gain_s": (
                    early_antonelli_clocks.at[RACE_LAPS, "ANT"] - clocks.at[RACE_LAPS, "ANT"]
                ),
            }
            if cover_lap is not None:
                checkpoint = max(early_lap, cover_lap) + 1
                lead = clocks.at[checkpoint, RIVAL] - clocks.at[checkpoint, REFERENCE]
                baseline_lead = baseline.at[checkpoint, RIVAL] - baseline.at[checkpoint, REFERENCE]
                clean_lead = clean.at[checkpoint, RIVAL] - clean.at[checkpoint, REFERENCE]
                row.update(
                    both_out_lap=checkpoint,
                    russell_lead_after_both_out_s=lead,
                    russell_stop_gain_vs_both_staying_out_s=lead - baseline_lead,
                    russell_stop_gain_from_traffic_s=lead - clean_lead,
                )
            for driver, name, lap in [(REFERENCE, "russell", early_lap), ("ANT", "antonelli", late_lap)]:
                place = _place(clocks.loc[lap], driver, clocks.at[lap, driver])
                row.update({f"{name}_{key}": value for key, value in place.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def sensitivity(field, settings, bounds):
    """Change one assumption at a time around the declared central case."""
    ranges = {
        "pit_loss_s": (bounds["low"], bounds["high"]),
        "field_spread": (0.0, 1.0),
        "ref_out_warm_s": (0.0, 1.20),
        "soft_out_warm_s": (0.0, 0.65),
        "medium_out_warm_s": (0.20, 1.40),
        "first_flying_fraction": (0.0, 0.65),
        "soft_deg": (0.00, 0.06),
        "medium_deg": (0.00, 0.05),
        "early_soft_adv_s": (-0.10, 0.20),
        "late_soft_adv_s": (-0.10, 0.45),
        "soft_set_age": (0, 3),
        "medium_graining_s": (0.0, 0.45),
        "close_follow_loss_s": (0.0, 0.90),
        "peak_tow_gain_s": (0.0, 0.30),
    }
    rows = []
    for parameter, values in ranges.items():
        for value in sorted(set((*values, getattr(settings, parameter)))):
            scenario = replace(settings, **{parameter: value})
            early = run_strategy(field, scenario, Strategy(SOFT, 22))
            late = run_strategy(field, scenario, Strategy(MEDIUM, 28))
            rows.append({
                "parameter": parameter, "value": value, "central_case": value == getattr(settings, parameter),
                "early_medium_rejoin": early["position"], "late_soft_rejoin": late["position"],
                "extra_traffic_late_s": late["net_traffic_s"] - early["net_traffic_s"],
                "late_soft_gain_s": early["finish_clock_s"] - late["finish_clock_s"],
                "required_late_soft_adv_s_per_lap": break_even(
                    field, scenario, 28, late["net_traffic_s"] - early["net_traffic_s"],
                ),
            })
    return pd.DataFrame(rows)
