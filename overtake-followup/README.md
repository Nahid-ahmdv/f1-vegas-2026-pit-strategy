# Did Verstappen get Overtake behind Russell in Las Vegas?

**Both methods support the detection order on laps 2, 10, 14 and 18.** On each, OpenF1 puts Russell directly ahead of Verstappen by less than one second, and the FIA records Verstappen first at the finish line. On lap 8, OpenF1 cannot confidently separate Hadjar and Verstappen at detection. Russell is clearly behind Verstappen there.

[Executed notebook](analysis.ipynb) · [Lap-by-lap CSV](lap_by_lap.csv) · [Detection vs finish figure](detection_vs_finish.png)

The signed gaps in the CSV are **Russell’s crossing time minus Verstappen’s**. A positive value means Verstappen crossed first. The finish values are FIA timing. Detection and activation are separate estimates made on a shared position trace, mapped from the FIA’s 5,299 m and 5,400 m marks. The low and high columns span alternative line locations and clock alignments with a timing margin. The FIA logs Overtake enabled on lap 1. Eligibility is estimated from the gap to the preceding car and race-control status. Public telemetry leaves actual battery deployment and driver intent unobserved.

## Cross-check in Jessica Steele’s tool

I confirmed the **Las Vegas Grand Prix 2026 Race** report header and selected the same lap for **VER** (reference) and **RUS** (comparison). I mapped the FIA’s 5,299 m mark through each axis’s S2 and finish distances. It lands near **5,260 m** on the lap 2 replay, **5,256 m** for laps 10 and 18, **5,253 m** for lap 14, and **5,282 m** on OpenF1’s shared axis. The replay’s T16 and T17 labels bracket the same kink and straight as the OpenF1 trace. [Lap Segmentation Tool](https://f1lapsegmentation-production.up.railway.app/) · [FIA circuit map](https://www.fia.com/system/files/decision-document/2026_las_vegas_grand_prix_-_competition_notes_-_circuit_map_pit_lane_drawing_emergency_exits_map_and_red_zone.pdf)

All gaps are **RUS crossing time minus VER crossing time**. Negative means Russell was ahead. I added the FIA opening gap to the replay’s lap-relative delta. The notebook verifies each finish gap from the two drivers’ official lap times and the previous lap’s gap.

| Lap | Steele replay at detection | OpenF1 at detection | FIA finish gap | Detection order |
| ---: | ---: | ---: | ---: | :--- |
| 2 | ≈−0.13 s | −0.502 s | +0.173 s | Both support RUS ahead |
| 10 | ≈−0.30 s | −0.253 s | +0.202 s | Both support RUS ahead |
| 14 | ≈−0.29 s | −0.376 s | +0.136 s | Both support RUS ahead |
| 18 | ≈−0.34 s | −0.409 s | +0.007 s | Both support RUS ahead |

Moving the marker about **±40 m** left Russell ahead and Verstappen within 1.0 s in both methods on all four laps. On lap 14, for example, adding the **+0.256 s** FIA opening gap to the replay’s **−0.542 s** delta gives **−0.286 s** at detection. Russell gained **0.120 s** over the lap, giving the FIA finish gap of **+0.136 s**. Lap 2 was close to a tie at the replay’s early test point. Its roughly **0.38 s** difference between methods persists after the distance and clock checks. The exact FIA detection-loop timestamp is unavailable to resolve it.

The OpenF1 field puts Russell directly ahead of Verstappen on all four laps. Gasly was the next car through on lap 10, about **0.91 s** behind Verstappen. Hadjar was next on laps 14 and 18, about **0.79 s** and **0.46 s** behind Verstappen, respectively. The lap 10 replay puts the later lead change about **510 m** past detection. On lap 18 it brings them level close to the finish. The exact retake locations on those laps remain uncertain. The field order, gaps and race-control status support estimated Overtake eligibility on all four laps. Battery deployment is unobserved.

## Restart, laps 26–38

[Restart lap-by-lap CSV](restart_lap_by_lap.csv) records the estimated detection top three, the gaps to the preceding car, the ±40 m checks, and the full FIA start and finish orders. Overtake was re-enabled at **04:51:34 UTC** on lap 26, about **45 s** before Verstappen's estimated detection crossing. [FIA Race Control Messages](https://www.fia.com/sites/default/files/2026_21_usa_f1_r0_timing_raceracecontrolmessages_v01.pdf)

- **Laps 28, 31 and 37.** Russell was ahead at detection and finished first. Verstappen led again by the first FIA split of laps 29, 32 and 38. Position estimates bracket those retakes at roughly **425–600 m**, **400–575 m** and **400–600 m** from the start of the new lap, respectively.
- **Lap 36.** The order reversed. Verstappen led Russell by about **0.39 s** at detection, and Russell took the line **0.042 s** ahead. Russell's gap to Verstappen stayed within 1.0 s across the marker checks.
- **Lap 34.** Norris was about **0.16 s** ahead of Verstappen at the nominal detector, and Verstappen finished first. The order held under the marker shifts. It becomes uncertain when the timing margin is included, so this is a separate possible Norris case.
- **Lap 38.** Antonelli was still behind Russell at the second FIA split. At detection he was directly behind Verstappen by about **0.56 s**, with Russell next. The Antonelli–Verstappen gap stayed within 1.0 s across the marker checks.

No lap repeats the opening-stint Russell–Verstappen pattern across the tested marker locations. Lap 29 remains a borderline case under the wider timing uncertainty. The supplied CSV agrees on all 13 nominal detection top threes, both cars' immediate predecessors, and the FIA finish leaders and pair gaps. Its gap estimates differ by up to **0.17 s** from the recomputation. The tested ranges overlap on every lap. [Supplied restart comparison](las_vegas_laps_26-38_detection.csv)

Steele's replay supports the crossing order and one-second gap on all five checked laps, including the ±40 m tests. Negative gaps mean the comparison driver crossed first.

| Lap | Signed gap | Steele at detection | OpenF1 at detection | FIA finish gap |
| ---: | :--- | ---: | ---: | ---: |
| 28 | RUS − VER | ≈−0.47 s | −0.497 s | −0.150 s |
| 31 | RUS − VER | ≈−0.51 s | −0.550 s | −0.199 s |
| 36 | RUS − VER | ≈+0.42 s | +0.390 s | −0.042 s |
| 37 | RUS − VER | ≈−0.40 s | −0.392 s | −0.184 s |
| 38 | ANT − VER | ≈+0.53 s | +0.560 s | +0.381 s |

The first stint had four Russell-to-Verstappen lead changes between detection and the same finish line. After the restart, Russell could lead at both lines, with Verstappen back ahead in the next lap's opening sector. By lap 38, Antonelli occupied the place directly behind Verstappen at detection. The gaps and enabled status estimate eligibility. Battery deployment and driver intent remain unobserved.

### Turn 17, laps 36 and 37

[The gap plot](laps_36_37_gap.png) overlays OpenF1 position estimates and [Steele’s replay readings](finish_duel_steele.csv). The replay delta includes the FIA opening gap, and both distance axes are mapped between the second sector split and the finish. The approximate Turn 17 marker at 6.05 km follows the replay’s landmark and the final bend on the FIA circuit map.

On lap 36, the nearest samples of opposite sign put Russell's move at 6,069–6,074 m in the main OpenF1 trace and about 6,124–6,174 m in the replay. These are sample brackets for the estimated curves. The OpenF1 clock-margin check widens the range to 5,969 m through the FIA finish at 6,201 m. On lap 37, both estimates keep Russell ahead throughout the plotted stretch. Run `uv run --locked python finish_duel.py` after collecting both OpenF1 caches to regenerate the plot.

## Rerun

From the repository root, with [uv](https://docs.astral.sh/uv/) installed:

```sh
cd overtake-followup
uv sync --locked --python 3.11
uv run --locked python collect.py --source openf1
uv run --locked python collect.py --restart --source openf1
uv run --locked jupyter-execute --inplace --timeout=180 analysis.ipynb
```

The collector reuses the parent project’s FIA-checked lap table and race-control cache, with downloads for missing reports and feeds stored in this folder’s `.cache/`. The restart command writes a separate cache. The notebook can run offline after both collections, and includes the manually recorded replay readings. Run the collector with `--lap2` for a three-car pilot. Use `--source fastf1` to request the same race through FastF1.

## Sources

- Jessica Steele’s Verstappen–Antonelli analysis on laps 39–48 prompted this follow-up. Her tool provided the replay cross-check above. [LinkedIn analysis](https://lnkd.in/p/esVYKb4W) · [Lap Segmentation Tool](https://f1lapsegmentation-production.up.railway.app/)
- This run uses observed OpenF1 position and car samples for race session 11420. FastF1 3.9.2 found the session, but its unauthenticated timing, car and position feeds failed to load. [FastF1](https://docs.fastf1.dev/) · [OpenF1](https://openf1.org/)
- The FIA provides the official order, splits, track landmarks, and Overtake parameters and rules. [Race History Chart](https://www.fia.com/sites/default/files/2026_21_usa_f1_r0_timing_racehistorychart_v01.pdf) · [Race Sector Analysis](https://www.fia.com/sites/default/files/2026_21_usa_f1_r0_timing_racesectoranalysis_v01.pdf) · [Race Control Messages](https://www.fia.com/sites/default/files/2026_21_usa_f1_r0_timing_raceracecontrolmessages_v01.pdf) · [Circuit map](https://www.fia.com/system/files/decision-document/2026_las_vegas_grand_prix_-_competition_notes_-_circuit_map_pit_lane_drawing_emergency_exits_map_and_red_zone.pdf) · [Power Unit Information](https://www.fia.com/system/files/decision-document/2026_las_vegas_grand_prix_-_power_unit_information.pdf) · [Sporting Regulations Issue 09, B7.2.2–3](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf)
