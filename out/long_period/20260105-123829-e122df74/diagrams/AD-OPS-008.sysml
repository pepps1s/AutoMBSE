package behavior {

  // 定义项目
  item def ShutdownCommand;
  item def CurrentMode;
  item def SafetyStatus;
  item def DataStatus;

  // 定义行为
  action ObservatoryShutdownAndSafePark {
    in item command : ShutdownCommand;
    in item mode : CurrentMode;
    in item safetyStatus : SafetyStatus;
    in item dataStatus : DataStatus;

    // 行为的顺序
    first start;

    // Step 1: End exposure
    then action endExposure {
      in item status : dataStatus;
      first start;
      then action 'Stop exposure';
      then action 'Finalize data capture';
      then action 'Flush logs';
      then done;
    }

    // Step 2: Park the mount
    then action parkMount {
      first start;
      then action 'Deactivate drive system';
      then action 'Secure mount in parked position';
      then done;
    }

    // Step 3: Secure the enclosure
    then action secureEnclosure {
      first start;
      then action 'Close dome shutters';
      then action 'Lock access doors';
      then done;
    }

    // Step 4: Disable nonessential subsystems
    then action disableSubsystems {
      first start;
      then action 'Disable adaptive optics';
      then action 'Turn off non-critical systems';
      then done;
    }

    // Step 5: Safety confirmation (Interlocks and access checks)
    then action safetyConfirmation {
      in item status : safetyStatus;
      first start;
      then action 'Verify interlocks are engaged';
      then action 'Check access status';
      then done;
    }

    // Step 6: Final power down (transition to standby or safe park mode)
    then action powerDown {
      first start;
      then action 'Power down all subsystems';
      then action 'Set mode to Standby or SafePark';
      then done;
    }

    then done;
  }

}
