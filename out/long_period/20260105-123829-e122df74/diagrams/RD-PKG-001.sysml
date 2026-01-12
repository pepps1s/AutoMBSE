// RD-PKG-001 : RD Package Structure & Naming Conventions
// dependencies: RD-CTX-001
// goal: Define SysML v2 package tree layout for requirements (NF/F/IF/OPS/VER/TR)
//       to enforce model consistency and parsing stability.

// NOTE: Pure package tree + naming/ID rules.
//       No imports to avoid unresolved dependencies.

package RD_PKG_001_TMT_PackageStructureAndNamingConventions {

    // -------------------------------------------------------------------------
    // 0) Root conventions (readable, parse-stable)
    // -------------------------------------------------------------------------
    package RD_Conventions {

        // ID pattern rules (use these strings as authoritative formatting rules)
        // - Diagram/View IDs:
        //   * RD-CTX-### : context & scope requirements views
        //   * RD-F-###   : functional requirements views
        //   * RD-NF-###  : non-functional requirements views
        //   * RD-IF-###  : interface requirements views
        //   * RD-OPS-### : operational requirements views
        //   * RD-VER-### : verification requirements views
        //   * RD-TR-###  : traceability views
        //
        // - Requirement definition names (defs):
        //   * ReqDef_<Domain>_<Topic>_<ID>
        //     e.g., ReqDef_TMT_PointingStability_RD_F_001
        //
        // - Requirement usage names (usages):
        //   * req_<Domain>_<Topic>_<ID>
        //     e.g., req_TMT_PointingStability_RD_F_001
        //
        // - Requirement ID attribute (string) must match diagram ID family:
        //   * "RD-F-###", "RD-NF-###", "RD-IF-###", "RD-OPS-###", "RD-VER-###", "RD-TR-###", "RD-CTX-###"
        //
        // - Uniqueness rules:
        //   * No duplicate requirement def names within the entire model.
        //   * No duplicate requirement usage names within a package scope.
        //   * Each requirement ID string is globally unique across the model.

        // Optional: centralized constants (strings) to standardize prefixes.
        package RD_IdPrefixes {
            // Using simple attributes avoids type/import dependencies.
            attribute prefixCTX : String = "RD-CTX-";
            attribute prefixF   : String = "RD-F-";
            attribute prefixNF  : String = "RD-NF-";
            attribute prefixIF  : String = "RD-IF-";
            attribute prefixOPS : String = "RD-OPS-";
            attribute prefixVER : String = "RD-VER-";
            attribute prefixTR  : String = "RD-TR-";
        }
    }

    // -------------------------------------------------------------------------
    // 1) Top-level packages (>= 6 as required)
    // -------------------------------------------------------------------------
    package RD_Context {
        // Context & scope boundary requirements (dependency anchor: RD-CTX-001)
        // Example subpackages (optional, extend as the model grows)
        package RD_CTX_Requirements { }
        package RD_CTX_Assumptions  { }
        package RD_CTX_Constraints  { }
        package RD_CTX_Glossary     { }
    }

    package RD_Functions {
        // Functional requirements
        package RD_F_Requirements { }
        package RD_F_UseCases     { }
        package RD_F_States       { }
    }

    package RD_NFR {
        // Non-functional requirements
        package RD_NF_Requirements { }
        package RD_NF_Budgets      { }  // e.g., availability, thermal, mass/power constraints as NFRs
        package RD_NF_QualityAttrs { }  // reliability, maintainability, safety, etc.
    }

    package RD_Interfaces {
        // Interface requirements (logical/physical/software interface constraints)
        package RD_IF_Requirements { }
        package RD_IF_External     { }
        package RD_IF_Internal     { }
    }

    package RD_Ops {
        // Operational requirements (scenarios, modes, procedures)
        package RD_OPS_Requirements { }
        package RD_OPS_Scenarios    { }
        package RD_OPS_Modes        { }
    }

    package RD_VerificationTrace {
        // Verification requirements + Traceability governance
        package RD_VER_Requirements { }
        package RD_VER_Cases        { }
        package RD_TR_Traceability  { }

        // Optional structural placeholders for trace matrices / mappings
        package RD_TR_Mappings { }
    }

    // -------------------------------------------------------------------------
    // 2) Target system tag (non-binding, for discoverability)
    // -------------------------------------------------------------------------
    package RD_TargetSystem {
        // Human-readable naming only (no external dependencies).
        attribute systemNameEN : String = "Thirty Meter Telescope (TMT)";
        attribute systemNameZH : String = "三十米望远镜";
    }

    // -------------------------------------------------------------------------
    // 3) Parsing stability notes
    // -------------------------------------------------------------------------
    package RD_ParsingStabilityNotes {
        // Guidance for model builders:
        // - Keep all requirement defs under their domain packages:
        //     RD_Context / RD_Functions / RD_NFR / RD_Interfaces / RD_Ops / RD_VerificationTrace
        // - Keep verification cases under RD_VerificationTrace::RD_VER_Cases
        // - Keep trace links/mappings under RD_VerificationTrace::RD_TR_Traceability or RD_TR_Mappings
        //
        // - Never use wildcard/implicit imports in RD packages.
        // - If imports are needed later, import only stable foundational libraries
        //   from a single dedicated "Libraries" package to avoid unresolved references.
    }

}
