from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def extract_code_blocks_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    paths_cfg = config.get("paths") or {}
    input_arg = getattr(args, "input", None)
    input_path = (
        Path(input_arg).expanduser()
        if input_arg
        else (repo_root / (paths_cfg.get("views_res_json") or "AutoMBSE/out/views/res.json"))
    )
    output_arg = getattr(args, "output", None)
    output_path = (
        Path(output_arg).expanduser()
        if output_arg
        else (repo_root / (paths_cfg.get("mbse_code_blocks_json") or "AutoMBSE/out/views/mbse_code_blocks.json"))
    )

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_code_blocks = []
    methods = [
        "LLM_code",
        "LLM_Case_code",
        "LLM_Case_Obj_code",
        "MBSE_code_ds",
        "Sysml_v2_model_creator",
        "codeGEN",
        "ttool",
        "Chatgpt-4o",
        "MBSE_code_gpt",
    ]

    for item in data:
        item_blocks = {"itemID": item["itemID"]}
        for method in methods:
            if method in item:
                content = item[method]
                code_blocks = re.findall(r"```sysml\n(.*?)\n```", content, re.DOTALL)
                if code_blocks:
                    item_blocks[method] = code_blocks
        if len(item_blocks) > 1:
            all_code_blocks.append(item_blocks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_code_blocks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote: {output_path}")
    return 0
