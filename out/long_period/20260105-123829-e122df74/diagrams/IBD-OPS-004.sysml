package TMT_ObservationOps {

  package OPS_Library {

    // ===== Items (Commands, States, Telemetry, Sync) =====

    enum def ObsPhase {
      Startup;
      Slew;
      Acquisition;
      AlignmentPhasing;
      AOCloseLoop;
      ExposureLoop;
      Calibration;
      Shutdown;
    }

    item def SequencingCommand {
      attribute phase : ObsPhase;
      attribute commandId : String;
    }

    item def StatusConfirmation {
      attribute phase : ObsPhase;
      attribute ready : Boolean;
      attribute state : String;
      attribute commandId : String;
    }

    item def TelemetrySignal {
      attribute source : String;
      attribute name : String;
      attribute value : String;
      attribute timestamp : String;
    }

    // Exposure synchronization signals (reference IBD-ROOT-018 semantics)
    item def TriggerSignal {
      attribute phase : ObsPhase;
      attribute triggerId : String;
      attribute t0 : String;
    }

    item def DataSyncSignal {
      attribute triggerId : String;
      attribute dataReady : Boolean;
      attribute frameCount : Integer;
    }

    item def TimeSyncSignal {
      attribute timeStamp : String;
      attribute accuracy_ns : Integer;
    }

    item def MetadataRecord {
      attribute triggerId : String;
      attribute targetId : String;
      attribute instrumentId : String;
      attribute utc : String;
    }

    item def RawFrame {
      attribute triggerId : String;
      attribute bytes : Integer;
      attribute frameIndex : Integer;
    }

    item def CalibrationFrame {
      attribute triggerId : String;
      attribute calType : String;
      attribute bytes : Integer;
    }

    // ===== Ports =====

    port def CommandInPort {
      in cmd : SequencingCommand;
    }

    port def CommandOutPort {
      out cmd : SequencingCommand;
    }

    port def StatusInPort {
      in status : StatusConfirmation;
    }

    port def StatusOutPort {
      out status : StatusConfirmation;
    }

    port def TelemetryOutPort {
      out tm : TelemetrySignal;
    }

    port def TelemetryInPort {
      in tm : TelemetrySignal;
    }

    // Exposure sync ports (aligned to IBD-ROOT-018 naming intent)
    port def ExposureTriggerOut {
      out trigger : TriggerSignal;
    }

    port def ExposureTriggerIn {
      in trigger : TriggerSignal;
    }

    port def InstrumentReadoutOut {
      out raw : RawFrame;
      out cal : CalibrationFrame;
      out sync : DataSyncSignal;
    }

    port def InstrumentReadoutIn {
      in raw : RawFrame;
      in cal : CalibrationFrame;
      in sync : DataSyncSignal;
    }

    port def MetadataOut {
      out meta : MetadataRecord;
    }

    port def MetadataIn {
      in meta : MetadataRecord;
    }

    port def TimeSyncOut {
      out ts : TimeSyncSignal;
    }

    port def TimeSyncIn {
      in ts : TimeSyncSignal;
    }

    // ===== Parts (Subsystem Interfaces) =====

    part def OCS_Sequencer {
      port cmdToSubsystems : CommandOutPort;
      port statusFromSubsystems : StatusInPort;

      // OCS also consumes telemetry and publishes operator/system telemetry
      port tmIn : TelemetryInPort;
      port tmOut : TelemetryOutPort;

      // Exposure sequencing coordination
      port expTriggerOut : ExposureTriggerOut;
      port timeSyncIn : TimeSyncIn;
      port metaOut : MetadataOut;
      port dataSyncIn : InstrumentReadoutIn; // receives sync/raw/cal when routed back
    }

    part def TelescopeOpticalSystem {
      port cmdIn : CommandInPort;
      port statusOut : StatusOutPort;
      port tmOut : TelemetryOutPort;
    }

    part def MountStructure {
      port cmdIn : CommandInPort;
      port statusOut : StatusOutPort;
      port tmOut : TelemetryOutPort;
    }

    part def EnclosureSiteInfrastructure {
      port cmdIn : CommandInPort;
      port statusOut : StatusOutPort;
      port tmOut : TelemetryOutPort;
    }

    part def AlignmentPhasingSystem {
      port cmdIn : CommandInPort;
      port statusOut : StatusOutPort;
      port tmOut : TelemetryOutPort;
    }

    part def AdaptiveOpticsSubsystem {
      port cmdIn : CommandInPort;
      port statusOut : StatusOutPort;
      port tmOut : TelemetryOutPort;
    }

    part def InstrumentsSubsystem {
      port cmdIn : CommandInPort;
      port statusOut : StatusOutPort;
      port tmOut : TelemetryOutPort;

      // Exposure synchronization interface (IBD-ROOT-018 alignment)
      port expTriggerIn : ExposureTriggerIn;
      port timeSyncIn : TimeSyncIn;
      port readoutOut : InstrumentReadoutOut;
      port metaIn : MetadataIn;
    }

    part def DataSystem {
      // Commands for configuring capture/pipelines + confirmations
      port cmdIn : CommandInPort;
      port statusOut : StatusOutPort;

      // Telemetry ingestion + data product ingestion
      port tmIn : TelemetryInPort;
      port readoutIn : InstrumentReadoutIn;
      port metaIn : MetadataIn;
      port timeSyncIn : TimeSyncIn;
    }

    part def SafetySubsystem {
      port cmdIn : CommandInPort;
      port statusOut : StatusOutPort;
      port tmOut : TelemetryOutPort;
    }

    // Minimal "backbone" to make telemetry routing explicit (IBD-ROOT-011 alignment)
    part def TelemetryBackbone {
      port tmFromSubsystems : TelemetryInPort;
      port tmToOCS : TelemetryOutPort;
      port tmToData : TelemetryOutPort;
    }

    // Exposure synchronization function (IBD-ROOT-018 alignment)
    part def ExposureSynchronizationSystem {
      port triggerOut : ExposureTriggerOut;
      port timeSyncOut : TimeSyncOut;
      port metaOut : MetadataOut;

      port triggerIn : ExposureTriggerIn;   // optional: accept upstream trigger requests
      port timeSyncIn : TimeSyncIn;         // optional: accept external time authority
    }
  }

  // =====================================================================
  // IBD-OPS-004 : Nominal Observation Sequencing Wiring (Startup→Shutdown v2)
  // Dependencies: RD-F-002, RD-F-006, IBD-ROOT-012, IBD-ROOT-011, IBD-ROOT-018,
  //               IBD-SUB06-005, IBD-SUB08-005
  // =====================================================================

  package IBD_OPS_004 {

    import OPS_Library::*;

    part def TMT_ObsLifecycle_Nominal_Wiring {

      // --- Major participants / subsystems ---
      part ocs : OCS_Sequencer;
      part optics : TelescopeOpticalSystem;
      part mount : MountStructure;
      part enclosure : EnclosureSiteInfrastructure;
      part aps : AlignmentPhasingSystem;
      part ao : AdaptiveOpticsSubsystem;
      part instr : InstrumentsSubsystem;
      part data : DataSystem;
      part safety : SafetySubsystem;

      // --- Cross-cutting infrastructure ---
      part tmBus : TelemetryBackbone;
      part expSync : ExposureSynchronizationSystem;

      // ==========================================================
      // A) Sequencing command flow (OCS -> Subsystems)
      // Includes each major phase as explicit command/state transition path
      // ==========================================================

      // Startup
      connect ocs.cmdToSubsystems::cmd to optics.cmdIn::cmd;
      connect ocs.cmdToSubsystems::cmd to mount.cmdIn::cmd;
      connect ocs.cmdToSubsystems::cmd to enclosure.cmdIn::cmd;
      connect ocs.cmdToSubsystems::cmd to aps.cmdIn::cmd;
      connect ocs.cmdToSubsystems::cmd to ao.cmdIn::cmd;
      connect ocs.cmdToSubsystems::cmd to instr.cmdIn::cmd;
      connect ocs.cmdToSubsystems::cmd to data.cmdIn::cmd;
      connect ocs.cmdToSubsystems::cmd to safety.cmdIn::cmd;

      // Slew + Acquisition + Alignment/Phasing + AO close loop + Calibration + Shutdown
      // (Same directed command channel; phase is carried in SequencingCommand.phase)
      // The explicit lifecycle path is encoded via ObsPhase and StatusConfirmation.phase.

      // ==========================================================
      // B) Status confirmation flow (Subsystems -> OCS) [separate direction]
      // ==========================================================

      connect optics.statusOut::status to ocs.statusFromSubsystems::status;
      connect mount.statusOut::status to ocs.statusFromSubsystems::status;
      connect enclosure.statusOut::status to ocs.statusFromSubsystems::status;
      connect aps.statusOut::status to ocs.statusFromSubsystems::status;
      connect ao.statusOut::status to ocs.statusFromSubsystems::status;
      connect instr.statusOut::status to ocs.statusFromSubsystems::status;
      connect data.statusOut::status to ocs.statusFromSubsystems::status;
      connect safety.statusOut::status to ocs.statusFromSubsystems::status;

      // ==========================================================
      // C) Unified telemetry routing (Subsystems -> TelemetryBackbone -> OCS + Data)
      // No subsystem included without at least one command + one telemetry connection
      // ==========================================================

      // Telemetry ingress to backbone
      connect optics.tmOut::tm to tmBus.tmFromSubsystems::tm;
      connect mount.tmOut::tm to tmBus.tmFromSubsystems::tm;
      connect enclosure.tmOut::tm to tmBus.tmFromSubsystems::tm;
      connect aps.tmOut::tm to tmBus.tmFromSubsystems::tm;
      connect ao.tmOut::tm to tmBus.tmFromSubsystems::tm;
      connect instr.tmOut::tm to tmBus.tmFromSubsystems::tm;
      connect safety.tmOut::tm to tmBus.tmFromSubsystems::tm;

      // Backbone fanout to OCS and DataSystem
      connect tmBus.tmToOCS::tm to ocs.tmIn::tm;
      connect tmBus.tmToData::tm to data.tmIn::tm;

      // ==========================================================
      // D) Exposure synchronization (references IBD-ROOT-018)
      // - Trigger + TimeSync + Metadata coordinated across OCS, Instruments, Data
      // - Keep signals distinct from command/status channels
      // ==========================================================

      // OCS drives/requests exposure sequencing -> ExposureSynchronizationSystem
      connect ocs.expTriggerOut::trigger to expSync.triggerIn::trigger;

      // ExposureSynchronizationSystem provides authoritative trigger + time to instrument/data
      connect expSync.triggerOut::trigger to instr.expTriggerIn::trigger;
      connect expSync.timeSyncOut::ts to instr.timeSyncIn::ts;
      connect expSync.timeSyncOut::ts to data.timeSyncIn::ts;

      // OCS and Data receive metadata produced for each exposure (e.g., timestamps, target/instrument ids)
      connect expSync.metaOut::meta to ocs.metaOut::meta;
      connect expSync.metaOut::meta to data.metaIn::meta;
      connect ocs.metaOut::meta to instr.metaIn::meta; // deliver metadata context to instrument controller

      // Instrument readout routed into DataSystem (raw/cal + data-ready sync)
      connect instr.readoutOut::raw to data.readoutIn::raw;
      connect instr.readoutOut::cal to data.readoutIn::cal;
      connect instr.readoutOut::sync to data.readoutIn::sync;

      // Optional: OCS also observes readout completion/sync (status confirmation separate from telemetry)
      connect instr.readoutOut::sync to ocs.dataSyncIn::sync;

      // ==========================================================
      // E) Nominal lifecycle guardrails (explicit phase path via confirmations)
      // ==========================================================
      //
      // The nominal path is:
      // Startup -> Slew -> Acquisition -> AlignmentPhasing -> AOCloseLoop -> ExposureLoop -> Calibration -> Shutdown
      //
      // This IBD wires:
      //  - SequencingCommand(phase=...) from OCS to each subsystem (A)
      //  - StatusConfirmation(phase=..., ready=true/false, state=...) from each subsystem back to OCS (B)
      //  - TelemetrySignal for monitoring/logging (C)
      //  - Exposure sync TriggerSignal/TimeSyncSignal/MetadataRecord + readout completion (D)
      //
      // Parsing/verification can assert that all phases exist in ObsPhase and are carried in both cmd+status.
    }

    // Instance view (optional anchor for diagram tooling)
    part ibd_ops_004 : TMT_ObsLifecycle_Nominal_Wiring;
  }
}
