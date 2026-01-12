from __future__ import annotations

import re
from typing import List


def partComponentDepose(inCode: str) -> List[str]:
    inCode = re.sub(r"//.*?$|/\*.*?\*/", "", inCode, flags=re.MULTILINE)

    pattern = r"part(.*?)(;|{)"
    matches = re.finditer(pattern, inCode, re.DOTALL)

    results: List[str] = []
    for match in matches:
        delimiter = match.group(2)

        if delimiter == ";":
            end_index = inCode.find(";", match.end())
            if end_index != -1:
                captured_content = inCode[match.start() : end_index + 1]
                results.append(captured_content.strip())
        elif delimiter == "{":
            brace_count = 1
            end_index = match.end()
            while brace_count > 0 and end_index < len(inCode):
                next_brace = inCode.find("{", end_index + 1)
                next_close_brace = inCode.find("}", end_index + 1)

                if next_close_brace == -1:
                    break

                if next_brace != -1 and next_brace < next_close_brace:
                    brace_count += 1
                    end_index = next_brace
                else:
                    brace_count -= 1
                    end_index = next_close_brace

            if brace_count == 0:
                captured_content = inCode[match.start() : end_index + 1]
                results.append(captured_content.strip())

    return results
