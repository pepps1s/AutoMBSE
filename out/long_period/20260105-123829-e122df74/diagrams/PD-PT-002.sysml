package PD_PT_002_Image_Motion_Jitter_Budget {

    import TMT_SysML_Components::*;
    import TMT_Budget_Components::*;
    
    part frequencyBandBins : FrequencyBand {
        attribute lowBand : Real = 0..0.1;
        attribute midBand : Real = 0.1..1.0;
        attribute highBand : Real = 1.0..10.0;
    }

    part jitterRMS : JitterRMS {
        attribute lowBandJitter : Real = 0..5;
        attribute midBandJitter : Real = 5..10;
        attribute highBandJitter : Real = 10..20;
    }

    part jitterContributors : JitterContributors {
        attribute structureContrib : Real = 0..2;
        attribute driveContrib : Real = 0..3;
        attribute windContrib : Real = 0..1;
        attribute instrumentContrib : Real = 0..4;
    }

    part jitterConstraint : Constraint {
        constraint jitterSum {
            expression : (lowBandJitter + midBandJitter + highBandJitter) <= totalJitterTarget;
        }
    }

    part mitigationApproach : MitigationApproach {
        attribute isolation : Real = 0..10;
        attribute controlTuning : Real = 0..5;
    }

    part jitterMapping : JitterMapping {
        contributor -> mitigationApproach {
            structureContrib -> isolation;
            driveContrib -> controlTuning;
            windContrib -> isolation;
            instrumentContrib -> controlTuning;
        }
    }

    part PD_PT_002_Contributors {
        part redefines structureContrib : JitterContributors;
        part redefines driveContrib : JitterContributors;
        part redefines windContrib : JitterContributors;
        part redefines instrumentContrib : JitterContributors;
    }

    part PD_PT_002_JitterBudget {
        part redefines jitterRMS : JitterRMS;
        part redefines frequencyBandBins : FrequencyBand;
    }

    constraint totalJitterTarget {
        value: 30;
    }

}
