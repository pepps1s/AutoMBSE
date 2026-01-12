# TMT (Thirty Meter Telescope) — Long-Period Case Knowledge Pack (Qdrant Degraded)

> Note (keep this): this file is the “prior knowledge” input for validating the AutoMBSE `--long-period` use case. This use case aims to run **without Qdrant** (Qdrant degraded/fallback mode), so the content is organized as a compact knowledge pack that can be injected directly into the LLM context, rather than as long documents intended for vector indexing.

## 1) System Overview

- System name: Thirty Meter Telescope (TMT)
- Type: next-generation 30 m-class ground-based optical/infrared telescope (Optical & Infrared)
- Primary goals: provide higher light-gathering power and angular resolution to support frontier science (early universe, galaxy formation, first-generation stars, black holes, exoplanets, etc.)
- Key context: site selection and construction on Mauna Kea, Hawaiʻi have faced sustained local opposition and social controversy, leading to project delays
- International partners (per this case description): US (Caltech, UC), Japan, Canada, India

## 2) Primary Functions (Black-Box)

- Collect light (optical/IR) from astronomical targets
- Pointing, tracking, and stabilization for long exposures
- Wavefront sensing + alignment & phasing for segmented primary mirror
- (Optionally) adaptive optics (AO) correction to approach diffraction-limited performance
- Instrument interface hosting: imaging / spectroscopy / polarimetry etc.
- Observation planning/execution, calibration, data acquisition, and data products delivery
- Facility safety, health monitoring, fault response, and maintenance support

## 3) High-Level Architecture (White-Box)

Suggested decomposition into major modules in SysML (can be further refined into 2–3 levels to support 100+ views):

1. Telescope Optical System
   - Primary Mirror (M1): 30m segmented mirror (~492 segments)
   - Secondary Mirror (M2): active optics
   - Tertiary Mirror (M3): articulated/steering tertiary
2. Telescope Structure & Mount
   - Alt-az mount, bearings, drives, encoders
   - Structural supports, vibration control, thermal control (as needed)
3. Enclosure / Dome & Site Infrastructure
   - Dome/enclosure rotation & shutters, wind/thermal management
   - Power, HVAC, cooling, networking, physical access/handling systems
4. Alignment, Phasing, and Wavefront Sensing
   - Shack-Hartmann wavefront sensor (for pre-AO wavefront quality)
   - Alignment & Phasing System (APS), segment actuators, metrology
5. Adaptive Optics (AO) Subsystem (if included in the model scope)
   - AO real-time controller, wavefront sensors, deformable mirrors
   - Acquisition sequences, guide star selection (NGS/LGS), calibration
6. Science Instruments (abstracted)
   - Instrument ports, instrument selection, instrument control, calibration units
7. Observatory Control System (OCS) / Software
   - Observation sequencing, scheduling hooks, configuration management
   - Telemetry, health monitoring, alarms, operator interfaces, logging
8. Data System
   - Raw data capture, metadata, pipelines, archiving, distribution interfaces
9. Safety & Compliance
   - Personnel safety, laser safety (if AO/LGS), interlocks, emergency stop
10. Operations & Maintenance
   - Startup/shutdown, maintenance modes, fault handling, verification/validation hooks

## 4) Representative Operational Scenarios

- Observatory startup (power-up → subsystem self-check → readiness)
- Slew to target (pointing model → coarse pointing → settle)
- Acquisition (target acquisition → guide star acquisition → focus/alignment checks)
- Segment alignment & phasing sequence (APS loop)
- AO acquisition (optional) and closing control loops
- Science exposure loop (expose → readout → metadata capture → quick-look QA)
- Calibration (flat/dark/arc, WFS calibration, pointing model updates)
- Fault response (detect anomaly → safe-state → notify → diagnose → recover)
- Observatory shutdown / safe park
- Maintenance (segment replacement/cleaning, actuator calibration, instrument swap)

## 5) Candidate Requirements (Seeds)

Non-functional (seed examples to help generate RD outputs):

- High angular resolution and sensitivity in optical/IR bands
- Availability/reliability targets for scheduled observing time
- Maintainability: segment handling, calibration, modular replacement
- Safety: personnel + operational safety (including controversial site constraints)
- Traceability: requirements → design elements → verification cases
- Robust logging/telemetry for diagnostics and post-run analysis

Functional (seed examples):

- Provide target pointing and tracking with required stability
- Maintain optical alignment and segment phasing within tolerances
- Provide instrument interface services (power/data/thermal/control)
- Execute observation sequences and produce data products with metadata
- Detect faults and transition to safe states automatically when needed
