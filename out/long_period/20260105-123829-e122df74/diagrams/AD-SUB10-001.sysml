package AD_SUB10_001_Maintenance_Mode {

  // Define actions related to entering maintenance mode
  action MaintenanceModeEntry {
    in item MaintenanceRequest;
    in item MaintenanceAccessAuthorization;

    first start;
    then action checkAuthorization {
      in item MaintenanceAccessAuthorization;
      first start;
      then action verifyAccessRights;
      then action validateSafetyClearance;
      then done;
    }
    then action safetyPreparation {
      in item MaintenanceRequest;
      first start;
      then action isolateEnergy;
      then action inhibitMotion;
      then action engageAccessInterlocks;
      then done;
    }
    then action recordMaintenanceContext {
      in item MaintenanceRequest;
      first start;
      then action logWhat;
      then action logWhy;
      then action logWhen;
      then done;
    }
    then action confirmMaintenanceReadiness {
      first start;
      then action checkSafetyGates;
      then action prepareSegmentHandling;
      then done;
    }
    then done;
  }

  // Define transitions for maintenance mode entry process
  action checkAuthorization {
    in item MaintenanceAccessAuthorization;
    first start;
    then action verifyAccessRights;
    then action validateSafetyClearance;
    then done;
  }

  action safetyPreparation {
    in item MaintenanceRequest;
    first start;
    then action isolateEnergy;
    then action inhibitMotion;
    then action engageAccessInterlocks;
    then done;
  }

  action recordMaintenanceContext {
    in item MaintenanceRequest;
    first start;
    then action logWhat;
    then action logWhy;
    then action logWhen;
    then done;
  }

  action confirmMaintenanceReadiness {
    first start;
    then action checkSafetyGates;
    then action prepareSegmentHandling;
    then done;
  }

  // Safety and standby check for 'authorization denied' branch
  action handleAuthorizationDenied {
    first start;
    then action returnToSafeStandby;
    then done;
  }

}
