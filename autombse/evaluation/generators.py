from __future__ import annotations

import functools
import json
import os
import time
from typing import Any

import requests
from tqdm import tqdm

from ..integrations.ttool_ai import AIInterface
from ..pipeline.runner import Pipeline
from ..sysml.code_blocks import extractStageWoStage
from ..sysml.knowledge import SysMLKnowledge
from ..sysml.package_tree import parse_packages
from ..verification.engine import Rules


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                elapsed = end - start
                return result, elapsed
            except Exception as e:
                print(f"Function {func.__name__} failed: {str(e)}")
                print("Retrying in 5 seconds...")
                time.sleep(5)
                continue

    return wrapper


class exampleGenerate:
    @staticmethod
    @timeit
    def LLM_generate(input: str, pp: Pipeline):
        pp.history = []
        return pp.response(input, "")

    @staticmethod
    @timeit
    def LLM_case_generate(input: str, pp: Pipeline, case: str):
        pp.history = []
        context = ""
        context += "-- Similar example in SysML-v2 --\n\n" + case + "\n"
        return pp.response(input, context)

    @staticmethod
    @timeit
    def MBSE_generate(input: str, pp: Pipeline, case: str, his: list):
        pp.history = his[-2:]
        context = ""
        context += "-- SysML Related Knowledge --\n\n" + SysMLKnowledge["BDD"]
        context += "-- Similar example in SysML-v2 --\n\n" + case + "\n"
        contextWithHistory = context

        response = pp.response(input, contextWithHistory)

        code = "\n".join(extractStageWoStage(response))

        package_dict = {}
        packages = parse_packages(code, package_dict)

        validation_rules = Rules(packages)

        bdd_errors, bdd_warnings = validation_rules.validate_by_type("bdd")
        cross_errors, cross_warnings = validation_rules.validate_by_type("cross")

        if bdd_errors or bdd_warnings or cross_errors or cross_warnings:
            validation_context = "\n-- Validation Results --\n"

            if bdd_errors:
                validation_context += "\nBDD Errors:\n"
                for error in bdd_errors:
                    validation_context += f"- {error['description']}: {error['errors']}\n"

            if bdd_warnings:
                validation_context += "\nBDD Warnings:\n"
                for warning in bdd_warnings:
                    validation_context += f"- {warning['description']}: {warning['warnings']}\n"

            if cross_errors:
                validation_context += "\nCross-View Errors:\n"
                for error in cross_errors:
                    validation_context += f"- {error['description']}: {error['errors']}\n"

            if cross_warnings:
                validation_context += "\nCross-View Warnings:\n"
                for warning in cross_warnings:
                    validation_context += f"- {warning['description']}: {warning['warnings']}\n"

            contextWithHistory += validation_context
            contextWithHistory += "\nPlease fix the above issues in your response. \n"

        response = pp.response(input, contextWithHistory)

        return response

    @staticmethod
    @timeit
    def MBSE_wo_req(input: str, pp: Pipeline, case: str, his: list):
        _ = case
        pp.history = his[-2:]
        contextWithHistory = "\n-- History Dialogue about this session --\n\n" + "\n".join(pp.history) + "\n-- History Dialogue END --\n"

        response = pp.response(input, contextWithHistory)

        code = "\n".join(extractStageWoStage(response))

        package_dict = {}
        packages = parse_packages(code, package_dict)

        validation_rules = Rules(packages)

        bdd_errors, bdd_warnings = validation_rules.validate_by_type("bdd")
        cross_errors, cross_warnings = validation_rules.validate_by_type("cross")

        if bdd_errors or bdd_warnings or cross_errors or cross_warnings:
            validation_context = "\n-- Validation Results --\n"

            if bdd_errors:
                validation_context += "\nBDD Errors:\n"
                for error in bdd_errors:
                    validation_context += f"- {error['description']}: {error['errors']}\n"

            if bdd_warnings:
                validation_context += "\nBDD Warnings:\n"
                for warning in bdd_warnings:
                    validation_context += f"- {warning['description']}: {warning['warnings']}\n"

            if cross_errors:
                validation_context += "\nCross-View Errors:\n"
                for error in cross_errors:
                    validation_context += f"- {error['description']}: {error['errors']}\n"

            if cross_warnings:
                validation_context += "\nCross-View Warnings:\n"
                for warning in cross_warnings:
                    validation_context += f"- {warning['description']}: {warning['warnings']}\n"

            contextWithHistory += validation_context
            contextWithHistory += "\nPlease fix the above issues in your response.\n"

        response = pp.response(input, contextWithHistory)

        return response

    @staticmethod
    @timeit
    def MBSE_wo_fot(input: str, pp: Pipeline, case: str, his: list):
        _ = case
        _ = his
        context = ""
        context += "-- SysML Related Knowledge --\n\n" + SysMLKnowledge["BDD"]
        contextWithHistory = context

        response = pp.response(input, contextWithHistory)

        code = "\n".join(extractStageWoStage(response))

        package_dict = {}
        packages = parse_packages(code, package_dict)

        validation_rules = Rules(packages)

        bdd_errors, bdd_warnings = validation_rules.validate_by_type("bdd")
        cross_errors, cross_warnings = validation_rules.validate_by_type("cross")

        if bdd_errors or bdd_warnings or cross_errors or cross_warnings:
            validation_context = "\n-- Validation Results --\n"

            if bdd_errors:
                validation_context += "\nBDD Errors:\n"
                for error in bdd_errors:
                    validation_context += f"- {error['description']}: {error['errors']}\n"

            if bdd_warnings:
                validation_context += "\nBDD Warnings:\n"
                for warning in bdd_warnings:
                    validation_context += f"- {warning['description']}: {warning['warnings']}\n"

            if cross_errors:
                validation_context += "\nCross-View Errors:\n"
                for error in cross_errors:
                    validation_context += f"- {error['description']}: {error['errors']}\n"

            if cross_warnings:
                validation_context += "\nCross-View Warnings:\n"
                for warning in cross_warnings:
                    validation_context += f"- {warning['description']}: {warning['warnings']}\n"

            contextWithHistory += validation_context
            contextWithHistory += "\nPlease fix the above issues in your response.\n"

        response = pp.response(input, contextWithHistory)

        return response

    @staticmethod
    @timeit
    def MBSE_wo_verify(input: str, pp: Pipeline, case: str, his: list):
        pp.history = his[-2:]
        context = ""
        context += "-- SysML Related Knowledge --\n\n" + SysMLKnowledge["BDD"]
        context += "-- Similar example in SysML-v2 --\n\n" + case + "\n"
        contextWithHistory = context
        contextWithHistory = (
            "\n-- History Dialogue about this session --\n\n"
            + "\n".join(pp.history)
            + "\n-- History Dialogue END --\n"
            + context
        )

        response = pp.response(input, contextWithHistory)

        return response

    @staticmethod
    @timeit
    def Sysml_v2_model_creator_generate(
        input: str, api_key: str, base_url: str = "https://api.openai.com/v1/chat/completions"
    ):
        context = (
            "As a systems engineer working on complex projects, I am looking to adopt the SysML v2 methodology to enhance our "
            "Model-Based Systems Engineering (MBSE) capabilities. Please provide detailed code on how to implement SysML v2 for "
            "creating robust MBSE models. Here's the description about the project: \n\n"
        )

        url = base_url
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        main_object = {
            "model": "sysml_v2_model_creator",
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": input},
            ],
        }
        response = requests.post(url, headers=headers, data=json.dumps(main_object))
        answer_object = response.json()
        answer_array = answer_object["choices"]
        answer_text = answer_array[0]
        message_text = answer_text["message"]
        ai_text = message_text["content"]
        return ai_text

    @staticmethod
    @timeit
    def codeGEN_generate(input: str, api_key: str, base_url: str = "https://api.openai.com/v1/chat/completions"):
        context = (
            "As a systems engineer working on complex projects, I am looking to adopt the SysML v2 methodology to enhance our "
            "Model-Based Systems Engineering (MBSE) capabilities. Please provide detailed code on how to implement SysML v2 for "
            "creating robust MBSE models. Here's the description about the project: \n\n"
        )

        url = base_url
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        main_object = {
            "model": "codeGEN",
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": input},
            ],
        }
        response = requests.post(url, headers=headers, data=json.dumps(main_object))
        answer_object = response.json()
        answer_array = answer_object["choices"]
        answer_text = answer_array[0]
        message_text = answer_text["message"]
        ai_text = message_text["content"]
        return ai_text

    @staticmethod
    @timeit
    def ttool_generate(input: str, api_key: str, base_url: str = "https://api.openai.com/v1/chat/completions"):
        ai = AIInterface()
        ai.set_key(api_key)
        ai.set_url(base_url)
        return ai.chat(input)

    @staticmethod
    def res_process(res_json_path: str = "AutoMBSE/out/views/res.json"):
        try:
            with open(res_json_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())

            for item in data:
                if "codeGEN" in item:
                    item["codeGEN"] = item["codeGEN"].replace("model package block", "part")

                if "ttool" in item:
                    item["ttool"] = item["ttool"].replace("model package block", "part")

            with open(res_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print("Replacement complete!")

        except Exception as e:
            print(f"Error processing file: {str(e)}")

    @staticmethod
    def pipeline(
        api_key,
        indir: str = "AutoMBSE/resource/examples/example.json",
        outdir: str = "AutoMBSE/out/views/res.json",
        base_url: str = "https://api.openai.com/v1",
        res_old_path: str = "AutoMBSE/out/views/res_old.json",
    ):
        with open(indir, "r") as ifile:
            text = ifile.read()
            examples = json.loads(text)
            ifile.close()
        pp_ds = Pipeline(api_key=api_key)
        pp = Pipeline(api_key=api_key, base_url=base_url, model="chatgpt-4o-latest")
        pp.systemContent = (
            "As a systems engineer working on complex projects, I am looking to adopt the SysML v2 methodology to enhance our Model-Based Systems Engineering (MBSE) capabilities. Please provide detailed code on how to implement SysML v2 for creating robust BDD views with 'part' as elements."
        )
        pp_ds.systemContent = (
            "As a systems engineer working on complex projects, I am looking to adopt the SysML v2 methodology to enhance our Model-Based Systems Engineering (MBSE) capabilities. Please provide detailed code on how to implement SysML v2 for creating robust BDD views with 'part' as elements."
        )
        example_history = []
        exist_flag = True

        try:
            with open(res_old_path, "r") as ifile:
                exist_res = json.loads(ifile.read())
                ifile.close()
        except Exception as e:
            _ = e
            exist_flag = False

        outdir_dir = os.path.dirname(outdir)
        if outdir_dir:
            os.makedirs(outdir_dir, exist_ok=True)
        with open(outdir, "a+") as ofile:
            ofile.seek(0)
            tmp = ofile.read()
            if tmp != "":
                tmp = json.loads("[" + tmp[:-2] + "]")
                tmpID = tmp[-1]["itemID"] + 1
            else:
                tmpID = 0
            for index, example in tqdm(enumerate(examples, 0)):
                if index < tmpID:
                    continue
                if exist_flag:
                    example["LLM_code"], example["LLM_code_time"] = exist_res[index]["LLM_code"], exist_res[index]["LLM_code_time"]
                    example["LLM_Case_code"], example["LLM_Case_code_time"] = (
                        exist_res[index]["LLM_Case_code"],
                        exist_res[index]["LLM_Case_code_time"],
                    )
                    example["LLM_Case_Obj_code"], example["LLM_Case_Obj_code_time"] = (
                        exist_res[index]["LLM_Case_Obj_code"],
                        exist_res[index]["LLM_Case_Obj_code_time"],
                    )
                    example["MBSE_code_gpt"], example["MBSE_code_gpt_time"] = (
                        exist_res[index]["MBSE_code_gpt"],
                        exist_res[index]["MBSE_code_gpt_time"],
                    )
                    example["MBSE_code_ds"], example["MBSE_code_ds_time"] = (
                        exist_res[index]["MBSE_code_ds"],
                        exist_res[index]["MBSE_code_ds_time"],
                    )
                    example["Sysml_v2_model_creator"], example["Sysml_v2_model_creator_time"] = (
                        exist_res[index]["Sysml_v2_model_creator"],
                        exist_res[index]["Sysml_v2_model_creator_time"],
                    )
                    example["codeGEN"], example["codeGEN_time"] = exist_res[index]["codeGEN"], exist_res[index]["codeGEN_time"]
                    example["ttool"], example["ttool_time"] = exist_res[index]["ttool"], exist_res[index]["ttool_time"]
                    example["Chatgpt-4o"], example["Chatgpt-4o_time"] = exist_res[index]["Chatgpt-4o"], exist_res[index]["Chatgpt-4o_time"]
                    example["LLM_wo_case"], example["LLM_wo_case_time"] = exist_res[index]["LLM_wo_case"], exist_res[index]["LLM_wo_case_time"]
                    example["LLM_wo_knowledge"], example["LLM_wo_knowledge_time"] = (
                        exist_res[index]["LLM_wo_knowledge"],
                        exist_res[index]["LLM_wo_knowledge_time"],
                    )
                    example["LLM_wo_memory"], example["LLM_wo_memory_time"] = (
                        exist_res[index]["LLM_wo_memory"],
                        exist_res[index]["LLM_wo_memory_time"],
                    )
                    example["LLM_wo_feedback"], example["LLM_wo_feedback_time"] = (
                        exist_res[index]["LLM_wo_feedback"],
                        exist_res[index]["LLM_wo_feedback_time"],
                    )
                    example["MBSE_wo_req"], example["MBSE_wo_req_time"] = exist_res[index]["MBSE_wo_req"], exist_res[index]["MBSE_wo_req_time"]
                    example["MBSE_wo_fot"], example["MBSE_wo_fot_time"] = exist_res[index]["MBSE_wo_fot"], exist_res[index]["MBSE_wo_fot_time"]
                    example["MBSE_wo_verify"], example["MBSE_wo_verify_time"] = (
                        exist_res[index]["MBSE_wo_verify"],
                        exist_res[index]["MBSE_wo_verify_time"],
                    )
                    example["Chatgpt-4.1"], example["Chatgpt-4.1_time"] = exist_res[index]["Chatgpt-4.1"], exist_res[index]["Chatgpt-4.1_time"]
                    example["ttool-4.1"], example["ttool-4.1_time"] = exist_res[index]["ttool-4.1"], exist_res[index]["ttool-4.1_time"]
                else:
                    # legacy known defects: may call undefined methods on this path
                    example["LLM_code"], example["LLM_code_time"] = exampleGenerate.LLM_generate(example["description"], pp)
                    example["LLM_Case_code"], example["LLM_Case_code_time"] = exampleGenerate.LLM_case_generate(
                        example["description"], pp, example["code"]
                    )

                example["MBSE_code_gpt"], example["MBSE_code_gpt_time"] = exampleGenerate.MBSE_generate(
                    example["description"], pp, example["code"], example_history
                )
                example["MBSE_code_ds"], example["MBSE_code_ds_time"] = exampleGenerate.MBSE_generate(
                    example["description"], pp_ds, example["code"], example_history
                )

                example_history.append("Description: " + example["description"] + " Code: " + example["code"])
                example["itemID"] = index

                tmp_json = json.dumps(example, ensure_ascii=False, indent=2)
                ofile.write(tmp_json + ",\n")
                ofile.flush()


__all__ = ["timeit", "exampleGenerate"]
