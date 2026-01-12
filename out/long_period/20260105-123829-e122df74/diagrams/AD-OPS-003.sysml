package behavior {

  // 定义项目
  item def targetPort;
  item def instrumentPort;
  item def guideStarStatus;
  item def acquisitionStatus;
  item def operatorAction;

  // 定义行为
  action TargetAcquisitionAndGuideStarAcquisition {
    in item target : targetPort;
    in item instrument : instrumentPort;
    in item guideStar : guideStarStatus;
    out item status : acquisitionStatus;

    // 行为的顺序
    first start; // 行为开始
    
    then action acquireTarget { // 获取目标
      in item target : targetPort;
      first start;
      then action 'select target';
      then action 'align instrument';
      then action 'check target validity';
      then done;
    }
    
    then action selectGuideStar { // 选择导星
      in item guideStar : guideStarStatus;
      first start;
      then action 'check guide star availability';
      then action 'select guide star';
      then action 'align guide star';
      then done;
    }
    
    then action closeGuidingLoop { // 闭合引导回路
      in item guideStar : guideStarStatus;
      first start;
      then action 'activate guiding system';
      then action 'adjust guide star position';
      then action 'verify guide star lock';
      then done;
    }
    
    then action handleGuideStarUnavailable { // 导星不可用时的替代策略
      in item operatorAction : operatorAction;
      first start;
      then action 'check alternative strategy';
      then action 'abort acquisition if no alternative';
      then done;
    }
    
    then action finalizeAcquisition { // 完成采集并确认状态
      in item status : acquisitionStatus;
      first start;
      then action 'confirm acquisition';
      then action 'finalize target and guide star lock';
      then done;
    }
    
    then done; // 行为结束
  }

}
