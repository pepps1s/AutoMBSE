package Availability_Reliability_Maintainability_ARM_NFR_Set {

    import ScalarValues::*;
    import Quantities::*;

    requirement def ARM_ReliabilityReq {
        doc /* This requirement defines the reliability targets related to the TMT’s scheduled observing time and operations. The Mean Time Between Failures (MTBF) should meet specified thresholds for telescope operations. */
        
        attribute mtbf : Real; /* Mean Time Between Failures in hours */
        attribute mttr : Real; /* Mean Time To Repair in hours */
        attribute downtimeCategory : String; /* Classification of downtime events, e.g., planned, unplanned */

        require constraint{ mtbf >= 5000 } /* MTBF shall be greater than or equal to 5000 hours */
        require constraint{ mttr <= 24 } /* MTTR shall be less than or equal to 24 hours */
    }

    requirement def ARM_AvailabilityReq {
        doc /* This requirement defines the availability targets for TMT’s operation, taking into account various downtime categories. It outlines the measurement of availability within the nightly operations window. */
        
        attribute scheduledObservationTime : Real; /* Time allocated for scheduled observations in hours */
        attribute actualObservationsTime : Real; /* Actual time observed in hours */
        
        require constraint{ actualObservationsTime / scheduledObservationTime >= 0.95 } /* Availability shall be 95% or higher */
    }

    requirement def ARM_MaintainabilityReq {
        doc /* This requirement specifies the maintainability criteria for telescope operations, addressing maintainability aspects for segment handling, replacement, and instrument swap. */
        
        attribute segmentReplacementTime : Real; /* Time required for segment replacement in hours */
        attribute instrumentSwapTime : Real; /* Time required for instrument swap in hours */

        require constraint{ segmentReplacementTime <= 48 } /* Segment replacement time shall be less than or equal to 48 hours */
        require constraint{ instrumentSwapTime <= 72 } /* Instrument swap time shall be less than or equal to 72 hours */
    }

    requirement def ARM_SparesStrategyReq {
        doc /* This requirement outlines the spares strategy assumptions for the telescope, ensuring availability of critical spare parts for repairs and maintenance. */
        
        attribute sparesStock : Integer; /* Number of critical spares to be kept in stock */
        
        require constraint{ sparesStock >= 10 } /* A minimum of 10 critical spares should be kept in stock */
    }

    requirement def ARM_MaintenanceContextReq {
        doc /* This requirement defines the context for measuring maintainability within the TMT, specifically addressing the handling/replacement of segments and instruments. */
        
        attribute maintenanceWindow : String; /* Time window available for maintenance activities */
        
        require constraint{ maintenanceWindow == "nightly" } /* Maintenance should occur within the nightly window */
    }

    requirement def ARM_InstrumentSwapReq {
        doc /* This requirement outlines the operational conditions and timeframes for instrument swapping in the TMT. */
        
        attribute swapWindow : String; /* Time window within which instrument swapping is to be performed */
        
        require constraint{ swapWindow == "nightly" } /* Instrument swapping should only occur during the nightly window */
    }

}
