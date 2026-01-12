package RD_CTX_001_SystemContextAndScopeBoundary {

    // Define the target system: TMT (Thirty Meter Telescope)
    system TMT {
        name: "Thirty Meter Telescope (TMT, 三十米望远镜)"
        description: "A next-generation 30m ground-based optical/infrared telescope for advanced scientific research."
        boundary: "Defines the boundary of the TMT system-of-interest, including key subsystems and external interactions."
    }

    // External entities (actors/systems interacting with TMT)
    externalEntity SiteInfrastructure {
        name: "Site Infrastructure"
        description: "Includes power, HVAC, cooling, networking, and physical access systems at the Mauna Kea site."
    }

    externalEntity Operators {
        name: "Operators"
        description: "Personnel responsible for telescope operations, including observation planning and execution."
    }

    externalEntity Instruments {
        name: "Instruments"
        description: "Scientific instruments interfacing with the telescope, providing data collection capabilities (imaging, spectroscopy, etc.)."
    }

    externalEntity Archive {
        name: "Archive"
        description: "Data storage and archival systems for storing and managing scientific data collected by TMT."
    }

    externalEntity PowerHVAC {
        name: "Power/HVAC"
        description: "Power supply and HVAC systems for thermal management and energy needs at the observatory."
    }

    externalEntity Environment {
        name: "Environment"
        description: "Environmental factors at the observatory site, including weather conditions affecting telescope operations."
    }

    externalEntity AlignmentPhasingSystem {
        name: "Alignment & Phasing System (APS)"
        description: "System responsible for aligning and phasing the segmented primary mirror of the telescope."
    }

    externalEntity AdaptiveOptics {
        name: "Adaptive Optics (AO)"
        description: "Optional system for real-time correction of wavefront errors to achieve diffraction-limited performance."
    }

    externalEntity ObservatoryControlSystem {
        name: "Observatory Control System (OCS)"
        description: "Software system for controlling the telescope, sequencing observations, and managing telemetry."
    }

    // In-scope and Out-of-scope items for the RD phase
    scopeInclusion TMTFunctionality {
        description: "In-scope for the RD phase: Telescope functionality, including optical system, alignment, phasing, adaptive optics (if applicable), and instrument interfaces."
    }

    scopeExclusion AO {
        description: "Out-of-scope for the RD phase: Adaptive optics (AO) system unless explicitly included."
    }

    // Define dependencies and relationships between the system and external entities
    relationship TMT_SiteInfrastructure {
        source: TMT
        target: SiteInfrastructure
        type: "Interacts with"
        description: "TMT relies on site infrastructure for power, HVAC, and networking."
    }

    relationship TMT_Operators {
        source: TMT
        target: Operators
        type: "Operated by"
        description: "Operators control the telescope's observation and maintenance operations."
    }

    relationship TMT_Instruments {
        source: TMT
        target: Instruments
        type: "Hosts"
        description: "TMT hosts scientific instruments for data acquisition."
    }

    relationship TMT_Archive {
        source: TMT
        target: Archive
        type: "Provides data to"
        description: "Scientific data is archived for storage and future analysis."
    }

    relationship TMT_PowerHVAC {
        source: TMT
        target: PowerHVAC
        type: "Requires"
        description: "TMT requires power and HVAC systems for operation."
    }

    relationship TMT_Environment {
        source: TMT
        target: Environment
        type: "Is affected by"
        description: "Environmental factors impact the operations of TMT at the Mauna Kea site."
    }

    relationship TMT_AlignmentPhasingSystem {
        source: TMT
        target: AlignmentPhasingSystem
        type: "Supports"
        description: "APS supports alignment and phasing of the segmented primary mirror."
    }

    relationship TMT_AdaptiveOptics {
        source: TMT
        target: AdaptiveOptics
        type: "Optionally integrates with"
        description: "AO can be integrated to correct wavefront errors for diffraction-limited performance."
    }

    relationship TMT_ObservatoryControlSystem {
        source: TMT
        target: ObservatoryControlSystem
        type: "Controlled by"
        description: "The telescope is controlled by the Observatory Control System (OCS)."
    }

    // Acceptance check for consistent naming and scope inclusions
    acceptanceCheck NamingConsistency {
        description: "Consistent naming of TMT and top-level external entities across all RD artifacts."
    }

    acceptanceCheck InScopeOutOfScope {
        description: "Clear definition of in-scope and out-of-scope items for the RD phase."
    }
}
