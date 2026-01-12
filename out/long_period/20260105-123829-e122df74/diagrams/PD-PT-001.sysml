package PD_PT_001_PointingTrackingErrorBudget {

    import TMT_LongPeriodPlanningInput::*;
    import TMT_SysML_Model::*;
    import TMT_SysML_Allocations::*;
    
    diagram PD_PT_001 {
        title "Pointing/Tracking Error Budget (Static + Dynamic)"
        goal "Allocate pointing and tracking error contributors (model error, encoder error, servo error, wind shake, structural flexure) and define roll-up to meet RD pointing/tracking functional and NFR constraints."
        dependencies PD_PRF_001, RD_F_003, RD_NF_002, BDD_SUB02_001, BDD_SUB07_001
        
        acceptance_checks {
            defines_value_types_for "arcsec", "arcsec/hr", "ms latency", "bandwidth Hz"
            includes_separate_static_dynamic_contributors with at_least 12 line_items
            roll_up_computes total_pointing_error, total_tracking_error against allocated_limits
            contributor_ownership_names (Mount/Structure, OCS_control_loop, environment) and ties_to_BDD_elements
            margin_policy_applied consistently_to_totals
        }
    }

    package PointingTrackingErrorContributors {
        part model_error : PointingTrackingErrorComponent {
            description "Modeling error contributors including optical and tracking misalignments"
        }
        part encoder_error : PointingTrackingErrorComponent {
            description "Error introduced by encoder inaccuracies in position measurement"
        }
        part servo_error : PointingTrackingErrorComponent {
            description "Servo errors due to motor inaccuracies in drive systems"
        }
        part wind_shake : PointingTrackingErrorComponent {
            description "Wind-induced perturbations affecting telescope stability"
        }
        part structural_flexure : PointingTrackingErrorComponent {
            description "Flexural movement of the structure under loads affecting alignment"
        }

        abstract part PointingTrackingErrorComponent {
            attribute error_value : Real
            attribute error_type : String
        }
    }

    package PointingTrackingErrorBudget {
        part static_contributors : PointingTrackingErrorComponent {
            redefines error_type = "Static"
            part model_error
            part encoder_error
            part servo_error
        }
        part dynamic_contributors : PointingTrackingErrorComponent {
            redefines error_type = "Dynamic"
            part wind_shake
            part structural_flexure
        }

        constraint static_dynamic_constraint {
            static_contributors.error_value + dynamic_contributors.error_value <= total_pointing_error_limit
        }
    }

    package PointingTrackingAllocation {
        part total_pointing_error : PointingTrackingErrorBudget {
            attribute error_limit : Real = 0.5 // max pointing error in arcseconds
        }
        part total_tracking_error : PointingTrackingErrorBudget {
            attribute error_limit : Real = 1.0 // max tracking error in arcseconds per hour
        }

        constraint roll_up_constraint {
            total_pointing_error.error_value + total_tracking_error.error_value <= total_pointing_error_limit + total_tracking_error_limit
        }
    }

    package BDD_SysML_Components {
        part MountStructure :> PointingTrackingErrorComponent {
            description "Subsystem for mount and structure components affecting pointing"
            part drive_system : DriveSystem
            part encoder_system : EncoderSystem
        }

        part OCS_ControlLoop :> PointingTrackingErrorComponent {
            description "Subsystem for the observatory control system loop affecting tracking error"
            part control_system : ControlSystem
        }

        part EnvironmentalFactors :> PointingTrackingErrorComponent {
            description "Environmental influences such as wind and thermal effects"
            part wind_effect : WindEffect
            part thermal_effect : ThermalEffect
        }
    }

    constraint margin_policy {
        total_pointing_error.error_value >= total_pointing_error_limit * 1.1 // margin applied
        total_tracking_error.error_value >= total_tracking_error_limit * 1.1 // margin applied
    }
}
