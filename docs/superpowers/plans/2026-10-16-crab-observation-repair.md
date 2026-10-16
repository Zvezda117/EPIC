# Crab observation model and raw-input integrity repair

The user confirmed Crab raw voltages without compensation and Beijing local time,
and requested staged fixes. Continue directly with the established repair workflow.

## Scope and design

Keep the validated CUDA arithmetic. Correct the physical contracts around it:
raw packet continuity/shared counter origin, explicit observation time and coordinate
frame, sky-directed 3-D steering, and sufficiently sampled images. Never infer a
sample epoch from file modification time or an undocumented header field.

1. Audit every 9178 header and search mounted data volumes for matching acquisition JSON.
   Decode possible BCD date/time as evidence only until its meaning is established.
2. Add a reusable packet audit and reject skipped/duplicate packets or unequal terminal
   counter origins before channelizing. Test wraparound, truncated files, malformed
   headers and cross-terminal mismatch with small synthetic packets.
3. Add explicit UTC/site/ICRS-source to local ENU geometry construction, with a target
   tangent grid, altitude and Nyquist checks, and raw/phase-tracked convention checks.
   Validate non-zenith analytic phasing and synthetic GPU recovery before real imaging.
4. Connect sky geometry to the existing FX runner with an explicit validity interval
   so a static geometry cannot silently cover arbitrarily long observations.
5. Fix converter timestamp offsets and preserve packet audit/provenance in output.
6. Run the full CPU/GPU suite; update the status with actual observation blockers.
   Regenerate a real Crab-centered product only when the physical metadata supports it.

## Known evidence at start

All six copied raw files contain 8184 complete packets, identical sequence range
36443352..36451535, increments of one, and valid packet magic. Header words 60/61
look like BCD 2026-01-11 14:29:27..30; word 62 resembles a subsecond counter. Their
sample/receiver meaning and timezone must be checked against acquisition records.
The previous NPZ files contain no actual UTC and used assumed 250 MS/s / 1355 MHz.
