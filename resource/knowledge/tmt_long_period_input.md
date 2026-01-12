# TMT Long-Period Planning Input (for diagram plan generation)

> Note (keep this): this file is the input (`--input-path`) for the AutoMBSE `--long-period` use case, to improve planner stability when generating **120+** views; this use case runs in **Qdrant degraded/fallback** mode by default (no vector-DB retrieval dependency).

## Target System

Thirty Meter Telescope (TMT): a next-generation 30 m-class ground-based optical/infrared telescope with a segmented primary mirror (~492 segments). It is an international partnership (US/Japan/Canada/India). Construction on Mauna Kea has faced local opposition and controversy, causing delays.

## Planning Guidelines (important)

1. Goal: output **at least 120** plan items (diagram/view).
2. Granularity: prefer “more and smaller”; each view should focus on a single topic (one subsystem, one scenario, one budget, one interface, etc.).
3. diagram_id: keep stable and unique; suggested formats:
   - `RD-NF-###` (non-functional requirements)
   - `RD-F-###` (functional requirements)
   - `BDD-<SUB>-###` (subsystem structure/modules)
   - `IBD-<SUB>-###` (key internal connections/interfaces)
   - `AD-OPS-###` / `AD-<SUB>-###` (operations/behavior/flows)
   - `PD-<BUDGET>-###` (parameters/budgets: power/mass/thermal/availability, etc.)
4. dependencies: use “shallow dependencies” to organize build order (e.g., RD → BDD → IBD/AD/PD); avoid deep chains and cyclic dependencies.
5. acceptance_checks: write as verifiable checks (e.g., “includes package + key part/attribute/connection/constraint”, “consistent naming”, “can be parsed into a package tree”).

## Suggested Subsystems / Viewpoints to Expand

- Telescope Optical System (M1 segments, M2, M3, optical path, phasing)
- Mount/Structure (alt-az, drives, encoders, vibration)
- Enclosure/Dome & Site Infrastructure (power, HVAC, networking)
- Alignment & Phasing System (APS, WFS, segment actuators)
- Adaptive Optics (optional: acquisition, loop closure, RTC)
- Instruments (ports, selection, calibration units)
- Observatory Control System (sequencer, config, telemetry, alarms)
- Data System (acquisition, metadata, pipeline, archive)
- Safety (interlocks, emergency stop, operational constraints)
- Operations/Maintenance (startup/shutdown, fault handling, maintenance modes)
