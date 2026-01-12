from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ReplayConfig:
    mode: str  # "off" | "record" | "replay"
    path: Path


class Pipeline:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-coder",
        max_tokens: int = 1024 * 2,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        max_context_bytes: int = 96 * 1024,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        replay: Optional[ReplayConfig] = None,
    ) -> None:
        from openai import OpenAI  # local import to keep CLI import light

        from ..sysml.knowledge import SysMLKnowledge
        from ..rag.domain_knowledge import domainKnowledge_with_params
        from ..rag.retriever import searchVec

        self._SysMLKnowledge = SysMLKnowledge
        self._domainKnowledge = domainKnowledge_with_params

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        default_system_prompt = (
            "As a systems engineer working on complex projects, I am looking to adopt the SysML v2 methodology to enhance our "
            "Model-Based Systems Engineering (MBSE) capabilities. Please provide detailed code on how to implement SysML v2 for "
            "creating robust MBSE models. "
        )
        self.systemContent = system_prompt or default_system_prompt
        self.maxTokens = max_tokens
        self.maxContextBytes = max_context_bytes
        self.roundCount = 0
        self.shortenHistory = False
        self.history: List[str] = []
        self.resCode: List[str] = []
        self.model = model
        self.temperature = temperature
        self.searchVec_util = searchVec(host=qdrant_host, port=qdrant_port)
        self.replay = replay

    @staticmethod
    def save_to_json(data, filename):
        if not isinstance(data, list) or not all(isinstance(sublist, list) for sublist in data):
            raise ValueError("Input data must be a list of lists of dicts")

        if not all(isinstance(item, dict) for sublist in data for item in sublist):
            raise ValueError("Input data must be a list of lists of dicts")

        if not filename.endswith(".json"):
            filename += ".json"

        json_data = json.dumps(data, indent=4, ensure_ascii=False)

        with open(filename, "w", encoding="utf-8") as file:
            file.write(json_data)

        print(f"Data saved to {os.path.abspath(filename)}")

    @staticmethod
    def save_to_csv(data, filename):
        if not isinstance(data, list) or not all(isinstance(sublist, list) for sublist in data):
            raise ValueError("Input data must be a list of lists of dicts")

        if not all(isinstance(item, dict) for sublist in data for item in sublist):
            raise ValueError("Input data must be a list of lists of dicts")

        if not filename.endswith(".csv"):
            filename += ".csv"

        all_stage = ["Non-Functional RD", "Functional RD", "BDD", "IBD", "AD", "PD"]
        all_keys = set()
        for sublist in data:
            for item in sublist:
                all_keys.update(item.keys())

        all_keys.add("stage")

        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=sorted(all_keys))

            writer.writeheader()
            for i, sublist in enumerate(data):
                if i == 0 or i == 2:
                    continue
                for item in sublist:
                    item["stage"] = all_stage[i]
                    writer.writerow(item)

        print(f"Data saved to {os.path.abspath(filename)}")

    def context_generate(
        self,
        stage: str,
        task: str,
        *,
        examples_top_k: int = 1,
        domain_top_k: int = 2,
        examples_collection: str = "examples_vec",
        knowledge_collection: str = "knowledge_chunk",
    ) -> str:
        if self.shortenHistory and len(self.history) > 2:
            self.history = self.history[-2:]
        context = ""
        context += "-- SysML Related Knowledge --\n\n" + self._SysMLKnowledge[stage]
        context += (
            "-- Similar example in SysML-v2 --\n\n"
            + "".join(
                [
                    item["code"]
                    for item in self.searchVec_util.find_top_k_similar(task, examples_top_k, examples_collection, False)
                ]
            )
            + "\n"
        )
        context += (
            "-- Domain Related Knowledge --\n\n"
            + self._domainKnowledge(
                task,
                k=domain_top_k,
                collection=knowledge_collection,
                retriever=self.searchVec_util,
            )
            + "\n"
        )
        return context

    def _replay_lookup(self, *, task: str, context: str) -> Optional[str]:
        if not self.replay:
            return None
        if self.replay.mode != "replay":
            return None
        if not self.replay.path.exists():
            raise FileNotFoundError(self.replay.path)

        with self.replay.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("round") == self.roundCount and obj.get("task") == task and obj.get("context") == context:
                    return obj.get("response")
        return None

    def _record_append(self, *, task: str, context: str, response: str) -> None:
        if not self.replay or self.replay.mode != "record":
            return
        self.replay.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"round": self.roundCount, "task": task, "context": context, "response": response}
        with self.replay.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def response(self, task: str, context: str) -> str:
        self.roundCount += 1
        if len(context) > self.maxContextBytes - self.maxTokens - (len(task) + len(self.systemContent)):
            context = context[-(self.maxContextBytes - self.maxTokens - (len(task) + len(self.systemContent)) - 1) :]

        if len(self.history) > 0:
            contextWithHistory = (
                "\n-- History Dialogue about this session --\n\n"
                + "\n".join(self.history)
                + "\n-- History Dialogue END --\n"
                + context
            )
        else:
            contextWithHistory = ""

        messages = (
            [
                {"role": "system", "content": self.systemContent},
                {"role": "user", "content": (task + "\n")},
                {"role": "assistant", "content": contextWithHistory},
            ]
            if contextWithHistory
            else [
                {"role": "system", "content": self.systemContent},
                {"role": "user", "content": (task + "\n") * 3},
            ]
        )

        replayed = self._replay_lookup(task=task, context=context)
        if replayed is not None:
            response_text = replayed
        else:
            response_text = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.maxTokens,
                    temperature=self.temperature,
                    stream=False,
                )
                .choices[0]
                .message.content
            )
            self._record_append(task=task, context=context, response=response_text)

        question = []
        for message in messages:
            if message["role"] == "assistant":
                question.append(message["role"] + ": " + context)
            else:
                question.append(message["role"] + ": " + message["content"])

        self.history.append("-- The Round " + str(self.roundCount) + "'s Question --\n\n" + "\n".join(question))
        self.history.append("-- The Round " + str(self.roundCount) + "'s Answer --\n\n" + response_text)
        self.resCode.append("```sysml\n" + self.extract_code_blocks(response_text) + "\n```\n")
        return response_text

    def extract_code_blocks(self, text: str) -> str:
        pattern_sysml = r"```sysml(.*?)```"
        pattern_general = r"```(.*?)```"

        matches_sysml = re.findall(pattern_sysml, text, re.DOTALL)
        if matches_sysml:
            return matches_sysml[-1]
        matches_general = re.findall(pattern_general, text, re.DOTALL)
        if matches_general:
            return matches_general[-1]
        return "NULL, round " + str(self.roundCount)

    def response_verification(self, task: str, context: str, rules: Any) -> str:
        errors, warnings = rules.validate_all()
        rule_stats = rules.get_validation_summary()

        feedback = "Based on SysML rule verification results, please regenerate the model:\n\n"

        if errors:
            feedback += "The following errors need to be corrected:\n"
            for error in errors:
                feedback += f"- Rule {error['rule_id']}: {error['description']}\n"
                feedback += f"  Specific error: {error['errors']}\n"

        if warnings:
            feedback += "\nThe following warnings need attention:\n"
            for warning in warnings:
                feedback += f"- Rule {warning['rule_id']}: {warning['description']}\n"
                feedback += f"  Specific warning: {warning['warnings']}\n"

        feedback += f"\nRule Statistics:\n"
        feedback += f"- Total rules: {rule_stats['rule_stats']['total_rules']}\n"
        feedback += f"- Active rules: {rule_stats['rule_stats']['active_rules']}\n"
        feedback += f"- Low failure rate rules: {rule_stats['rule_stats']['rules_by_failure_rate']['low']}\n"
        feedback += f"- Medium failure rate rules: {rule_stats['rule_stats']['rules_by_failure_rate']['medium']}\n"
        feedback += f"- High failure rate rules: {rule_stats['rule_stats']['rules_by_failure_rate']['high']}\n"

        new_task = f"{task}\n\n{feedback}"
        return self.response(task=new_task, context=context)

    def pipeline(self) -> list:
        res = []

        task = "Provide the non-functional requirements of the waterjet propulsion system, and model them using a SysML requirements diagram."
        context = self.context_generate("RD", task)
        res.append(self.response(task=task, context=context))
        print("RD1Done\n")

        task = "Provide the functional requirements of the waterjet propulsion system, and model them using a SysML requirements diagram."
        context = self.context_generate("RD", task)
        res.append(self.response(task=task, context=context))
        print("RD2Done\n")

        task = "Provide the complete functional modules of the waterjet propulsion system, supplement possible parameter values for each module, and use SysML v2 to build a Block Definition Diagram (BDD)."
        context = self.context_generate("BDD", task)
        res.append(self.response(task=task, context=context))
        print("BDDDone\n")

        task = "Based on the modules defined above, use SysML v2 to build an Internal Block Diagram (IBD) for the waterjet propulsion system."
        context = self.context_generate("IBD", task)
        res.append(self.response(task=task, context=context))
        print("IBDDone\n")

        task = "Use SysML v2 to build an Activity/Behavior Diagram (AD) for the waterjet propulsion system."
        context = self.context_generate("AD", task)
        res.append(self.response(task=task, context=context))
        print("ADDone\n")

        task = "Use SysML v2 to build a Parametric Diagram (PD) for the waterjet propulsion system."
        context = self.context_generate("PD", task)
        res.append(self.response(task=task, context=context))
        print("PDDone\n")

        # NOTE: legacy code calls self.pumpPartsSearch(), but the method is missing.
        # AutoMBSE does not implement it; callers may choose to treat this as a non-fatal legacy defect.
        if hasattr(self, "pumpPartsSearch"):
            getattr(self, "pumpPartsSearch")()

        return res


__all__ = ["Pipeline", "ReplayConfig"]
