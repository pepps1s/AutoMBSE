from __future__ import annotations

import re
from typing import List, Dict

from .parts import partComponentDepose


STAGE = ["RD1", "RD2", "BDD", "IBD", "AD", "PD"]


def extractStage(text: str) -> List[Dict[str, str]]:
    pattern_sysml = r"```sysml(.*?)```"
    res: List[Dict[str, str]] = []

    matches_sysml = re.findall(pattern_sysml, text, re.DOTALL)

    for index, match in enumerate(matches_sysml):
        parts = partComponentDepose(match)
        for part in parts:
            res.append({"part": part, "stage": STAGE[index]})
    return res


def extractStageWoStage(text: str) -> List[str]:
    pattern_sysml = r"```sysml(.*?)```"
    res: List[str] = []

    matches_sysml = re.findall(pattern_sysml, text, re.DOTALL)

    for index, match in enumerate(matches_sysml):
        _ = index
        res.append(match)
    return res

