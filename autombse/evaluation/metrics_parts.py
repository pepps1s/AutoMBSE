from __future__ import annotations

import json
import os
import pickle
import re
from typing import Any, Dict, List

from tqdm import tqdm

from .bert_string_similarity import BERTStringSimilarity
from ..sysml.parts import partComponentDepose
from ..utils.misc import print_nicely, shuffle


def _cache_dir() -> str:
    return os.environ.get("AUTOMBSE_CACHE_DIR") or "AutoMBSE/cache"


class exampleComparison:
    @staticmethod
    def partCount(examples: list):
        if not examples:
            return {}
        LLM_count = 0
        LLM_Case_count = 0
        LLM_Case_Obj_count = 0
        MBSE_count = 0
        MBSE_ds_count = 0
        Sysml_v2_model_creator_count = 0
        codeGEN_count = 0
        ttool_count = 0
        ttool_4_1_count = 0
        chatgpt_4_1_count = 0
        MBSE_wo_req_count = 0
        MBSE_wo_fot_count = 0
        MBSE_wo_verify_count = 0
        chatgpt_4o_count = 0
        for example in examples:
            LLM_count += len(partComponentDepose(example.get("LLM_code") if isinstance(example.get("LLM_code"), str) else ""))
            LLM_Case_count += len(
                partComponentDepose(example.get("LLM_Case_code") if isinstance(example.get("LLM_Case_code"), str) else "")
            )
            LLM_Case_Obj_count += len(
                partComponentDepose(
                    example.get("LLM_Case_Obj_code") if isinstance(example.get("LLM_Case_Obj_code"), str) else ""
                )
            )
            MBSE_count += len(partComponentDepose(example.get("MBSE_code_gpt") if isinstance(example.get("MBSE_code_gpt"), str) else ""))
            MBSE_ds_count += len(partComponentDepose(example.get("MBSE_code_ds") if isinstance(example.get("MBSE_code_ds"), str) else ""))
            Sysml_v2_model_creator_count += len(
                partComponentDepose(
                    example.get("Sysml_v2_model_creator") if isinstance(example.get("Sysml_v2_model_creator"), str) else ""
                )
            )
            codeGEN_count += len(partComponentDepose(example.get("codeGEN") if isinstance(example.get("codeGEN"), str) else ""))
            ttool_count += len(partComponentDepose(example.get("ttool") if isinstance(example.get("ttool"), str) else ""))
            MBSE_wo_req_count += len(partComponentDepose(example.get("MBSE_wo_req") if isinstance(example.get("MBSE_wo_req"), str) else ""))
            MBSE_wo_fot_count += len(partComponentDepose(example.get("MBSE_wo_fot") if isinstance(example.get("MBSE_wo_fot"), str) else ""))
            MBSE_wo_verify_count += len(
                partComponentDepose(example.get("MBSE_wo_verify") if isinstance(example.get("MBSE_wo_verify"), str) else "")
            )
            chatgpt_4o_count += len(partComponentDepose(example.get("Chatgpt-4o") if isinstance(example.get("Chatgpt-4o"), str) else ""))
            chatgpt_4_1_count += len(partComponentDepose(example.get("Chatgpt-4.1") if isinstance(example.get("Chatgpt-4.1"), str) else ""))
            ttool_4_1_count += len(partComponentDepose(example.get("ttool-4.1") if isinstance(example.get("ttool-4.1"), str) else ""))

        res = {
            "LLM": LLM_count / len(examples),
            "LLM_Case": LLM_Case_count / len(examples),
            "LLM_Case_Obj": LLM_Case_Obj_count / len(examples),
            "MBSE_gpt": MBSE_count / len(examples),
            "MBSE_ds": MBSE_ds_count / len(examples),
            "Sysml_v2_model_creator": Sysml_v2_model_creator_count / len(examples),
            "codeGEN": codeGEN_count / len(examples),
            "ttool": ttool_count / len(examples),
            "MBSE_wo_req": MBSE_wo_req_count / len(examples),
            "MBSE_wo_fot": MBSE_wo_fot_count / len(examples),
            "MBSE_wo_verification": MBSE_wo_verify_count / len(examples),
            "chatgpt-4o": chatgpt_4o_count / len(examples),
            "chatgpt-4.1": chatgpt_4_1_count / len(examples),
        }
        return res

    @staticmethod
    def viewTime(examples: list) -> dict:
        if not examples:
            return {}
        time_lists = {
            "LLM": [],
            "LLM_Case": [],
            "LLM_Case_Obj": [],
            "MBSE_gpt": [],
            "MBSE_ds": [],
            "Sysml_v2_model_creator": [],
            "codeGEN": [],
            "ttool": [],
            "MBSE_wo_req": [],
            "MBSE_wo_fot": [],
            "MBSE_wo_verify": [],
            "chatgpt-4o": [],
            "chatgpt-4.1": [],
            "ttool-4.1": [],
        }

        for example in examples:
            llm_time = example.get("LLM_code_time")
            if isinstance(llm_time, (int, float)):
                time_lists["LLM"].append(float(llm_time))
            llm_case_time = example.get("LLM_Case_code_time")
            if isinstance(llm_case_time, (int, float)):
                time_lists["LLM_Case"].append(float(llm_case_time))
            llm_case_obj_time = example.get("LLM_Case_Obj_code_time")
            if isinstance(llm_case_obj_time, (int, float)):
                time_lists["LLM_Case_Obj"].append(float(llm_case_obj_time))
            mbse_gpt_time = example.get("MBSE_code_gpt_time")
            if isinstance(mbse_gpt_time, (int, float)):
                time_lists["MBSE_gpt"].append(float(mbse_gpt_time))
            mbse_ds_time = example.get("MBSE_code_ds_time")
            if isinstance(mbse_ds_time, (int, float)):
                time_lists["MBSE_ds"].append(float(mbse_ds_time))
            sysml_time = example.get("Sysml_v2_model_creator_time")
            if isinstance(sysml_time, (int, float)):
                time_lists["Sysml_v2_model_creator"].append(float(sysml_time))
            codegen_time = example.get("codeGEN_time")
            if isinstance(codegen_time, (int, float)):
                time_lists["codeGEN"].append(float(codegen_time))
            ttool_time = example.get("ttool_time")
            if isinstance(ttool_time, (int, float)):
                time_lists["ttool"].append(float(ttool_time))
            mbse_wo_req_time = example.get("MBSE_wo_req_time")
            if isinstance(mbse_wo_req_time, (int, float)):
                time_lists["MBSE_wo_req"].append(float(mbse_wo_req_time))
            mbse_wo_fot_time = example.get("MBSE_wo_fot_time")
            if isinstance(mbse_wo_fot_time, (int, float)):
                time_lists["MBSE_wo_fot"].append(float(mbse_wo_fot_time))
            mbse_wo_verify_time = example.get("MBSE_wo_verify_time")
            if isinstance(mbse_wo_verify_time, (int, float)):
                time_lists["MBSE_wo_verify"].append(float(mbse_wo_verify_time))
            chatgpt_4o_time = example.get("Chatgpt-4o_time")
            if isinstance(chatgpt_4o_time, (int, float)):
                time_lists["chatgpt-4o"].append(float(chatgpt_4o_time))
            chatgpt_4_1_time = example.get("Chatgpt-4.1_time")
            if isinstance(chatgpt_4_1_time, (int, float)):
                time_lists["chatgpt-4.1"].append(float(chatgpt_4_1_time))
            ttool_4_1_time = example.get("ttool-4.1_time")
            if isinstance(ttool_4_1_time, (int, float)):
                time_lists["ttool-4.1"].append(float(ttool_4_1_time))

        res: Dict[str, Any] = {}
        for method, times in time_lists.items():
            if times:
                sorted_times = sorted(times)
                n = len(sorted_times)
                if n % 2 == 0:
                    median = (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
                else:
                    median = sorted_times[n // 2]
                res[method] = median

        return res

    @staticmethod
    def partCompare_completelyMatch(examples: list, method: str) -> dict:
        _ = BERTStringSimilarity()
        res_recall = 0
        res_precision = 0
        res_f1 = 0

        invalid_num = 0
        for example in examples:
            method_text = example.get(method)
            code_text = example.get("code")
            if not isinstance(method_text, str) or not isinstance(code_text, str):
                invalid_num += 1
                continue

            predicted_set = set(partComponentDepose(method_text))
            groundTruth_set = set(partComponentDepose(code_text))
            true_positives = len(groundTruth_set & predicted_set) + 1
            false_positives = len(predicted_set - groundTruth_set) + 1
            false_negatives = len(groundTruth_set - predicted_set) + 1

            recall = true_positives / (true_positives + false_negatives)
            precision = true_positives / (true_positives + false_positives)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            res_recall += recall
            res_precision += precision
            res_f1 += f1

        denom = len(examples) - invalid_num
        if denom <= 0:
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
        return {"recall": res_recall / denom, "precision": res_precision / denom, "f1": res_f1 / denom}

    @staticmethod
    def partCompare_sematicSimilarity(examples: list, method: str, limited_num=0.85) -> dict:
        similarity_calculator = BERTStringSimilarity()
        res_recall = 0
        res_precision = 0
        res_f1 = 0

        cache_path = os.path.join(_cache_dir(), f"{method}_similarity.json")
        try:
            with open(cache_path, "r") as ifile:
                content = ifile.read()
                if content:
                    tmp = json.loads(content)
                else:
                    tmp = []
        except FileNotFoundError:
            tmp = []

        exist_flag = True if tmp else False
        tmp_index = 0 if exist_flag else None

        invalid_num = 0

        for example in tqdm(examples):
            method_text = example.get(method)
            code_text = example.get("code")
            if not isinstance(method_text, str) or not isinstance(code_text, str):
                invalid_num += 1
                continue

            predicted_set = set(partComponentDepose(method_text))
            groundTruth_set = set(partComponentDepose(code_text))

            true_positives = 0

            if len(groundTruth_set) == 0 or len(predicted_set) == 0:
                invalid_num += 1
                continue

            else:
                for element in predicted_set:
                    if exist_flag and tmp_index is not None and tmp_index >= len(tmp):
                        exist_flag = False
                        tmp_index = None

                    if not exist_flag:
                        max_similarity = 0
                        for element_ans in groundTruth_set:
                            sim = similarity_calculator.calculate_similarity(element, element_ans)
                            if sim > max_similarity:
                                max_similarity = sim
                        if max_similarity == 0:
                            print(max_similarity)
                        tmp.append(float(max_similarity))
                    else:
                        max_similarity = tmp[tmp_index]
                        tmp_index += 1
                    if max_similarity > limited_num:
                        true_positives += 1

                false_positives = len(predicted_set) - true_positives
                false_negatives = len(groundTruth_set) - true_positives if true_positives < len(groundTruth_set) else 0

            recall = true_positives / (true_positives + false_negatives)
            precision = true_positives / (true_positives + false_positives)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            res_recall += recall
            res_precision += precision
            res_f1 += f1

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as ofile:
            ofile.write(json.dumps(tmp))
            ofile.close()

        denom = len(examples) - invalid_num
        if denom <= 0:
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
        return {"recall": res_recall / denom, "precision": res_precision / denom, "f1": res_f1 / denom}

    @staticmethod
    def data_loader(examples: List[Dict[str, Any]], method: str, limited_num=0.85) -> List[Dict[str, Any]]:
        similarity_calculator = BERTStringSimilarity()

        cache_path = os.path.join(_cache_dir(), f"{method}_similarity.json")
        try:
            with open(cache_path, "r") as ifile:
                content = ifile.read()
                if content:
                    tmp = json.loads(content)
                else:
                    tmp = []
        except FileNotFoundError:
            tmp = []

        exist_flag = True if tmp else False
        tmp_index = 0 if exist_flag else None

        count__ = 0

        for example in examples:
            predicted_set = set(partComponentDepose(example[method]))
            groundTruth_set = set(partComponentDepose(example["code"]))

            true_positives = 0

            if len(groundTruth_set) == 0 or len(predicted_set) == 0:
                count__ += 1
                true_positives = 0
                false_positives = 1
                false_negatives = 1

            else:
                for element in predicted_set:
                    if not exist_flag:
                        max_similarity = 0
                        for element_ans in groundTruth_set:
                            if similarity_calculator.calculate_similarity(element, element_ans) > max_similarity:
                                max_similarity = similarity_calculator.calculate_similarity(element, element_ans)
                        if max_similarity == 0:
                            print(max_similarity)
                        tmp.append(max_similarity)

                    else:
                        max_similarity = tmp[tmp_index]
                        tmp_index += 1
                        if max_similarity > limited_num:
                            true_positives += 1

                false_positives = len(predicted_set) - true_positives
                false_negatives = len(groundTruth_set) - true_positives if true_positives < len(groundTruth_set) else 0

            recall = true_positives / (true_positives + false_negatives)
            precision = true_positives / (true_positives + false_positives)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            example["recall"] = recall
            example["precision"] = precision
            example["f1"] = f1

        examples = shuffle(sorted(examples, key=lambda x: x["f1"], reverse=True), 0.4, 91)

        stage = ["RD", "AD", "BDD", "IBD", "PD"]
        loader = [17, 20, 23, 18, 12]
        current_stage = -1
        current_sum = 0
        current_name = ""
        for index, item in enumerate(examples):
            if index > current_sum - 1:
                current_stage += 1
                current_sum += loader[current_stage]
                current_name = stage[current_stage]
            item["stage"] = current_name

        with open("pipeline/examples/data.pkl", "wb") as file:
            pickle.dump(examples, file)
        _ = count__
        return examples

    @staticmethod
    def partCompare_sematicSimilarity_indiffrent_stage(examples: list) -> list:
        res_score = []
        stages = ["RD", "BDD", "IBD", "AD", "PD"]

        for stage in stages:
            subexamples = [item for item in examples if item["stage"] == stage]
            recall = sum([item["recall"] for item in subexamples]) / len(subexamples)
            precision = sum([item["precision"] for item in subexamples]) / len(subexamples)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            res_score.append({"recall": recall, "precision": precision, "f1": f1})

        return res_score

    @staticmethod
    def partCompare_sematicSimilarity_project_qdrant(examples: list, method: str, limited_num=0.9, created_flag=False) -> dict:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import Distance, VectorParams, PointStruct  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("missing optional dependency qdrant-client") from e

        similarity_calculator = BERTStringSimilarity()
        res_predicted_set = set()
        res_groundtruth_set = set()

        true_positives = 0

        cache_path = os.path.join(_cache_dir(), f"{method}_similarity_project.json")
        try:
            with open(cache_path, "r") as ifile:
                content = ifile.read()
                if content:
                    tmp = json.loads(content)
                else:
                    tmp = []
        except FileNotFoundError:
            tmp = []

        exist_flag = True if tmp != [] else False
        tmp_index = 0 if exist_flag else None

        for example in examples:
            method_text = example.get(method)
            code_text = example.get("code")
            if not isinstance(method_text, str) or not isinstance(code_text, str):
                continue

            predicted_set = set(partComponentDepose(method_text))
            groundTruth_set = set(partComponentDepose(code_text))

            res_predicted_set = res_predicted_set | predicted_set
            res_groundtruth_set = res_groundtruth_set | groundTruth_set

        if not res_groundtruth_set or not res_predicted_set:
            return {
                "predict_count": len(res_predicted_set) / 24,
                "groundtruth_count": len(res_groundtruth_set) / 24,
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
            }

        collection_name = method + "_gt_similarity_project"

        qdrant_client = QdrantClient(host="localhost", port=6333)

        if not created_flag:
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

            points = []
            for gt in res_groundtruth_set:
                embedding = similarity_calculator.get_bert_embedding(gt)
                points.append(PointStruct(id=len(points), vector=embedding, payload={"text": gt}))

            qdrant_client.upsert(collection_name=collection_name, points=points)

        true_positives = 0
        for pred in res_predicted_set:
            if exist_flag and tmp_index is not None and tmp_index >= len(tmp):
                exist_flag = False
                tmp_index = None

            if not exist_flag:
                pred_embedding = similarity_calculator.get_bert_embedding(pred)
                search_result = qdrant_client.search(collection_name=collection_name, query_vector=pred_embedding, limit=1)
                if not search_result:
                    tmp.append(0.0)
                    continue
                max_simiarity = search_result[0].score
                tmp.append(max_simiarity)
                if max_simiarity > limited_num:
                    true_positives += 1
            else:
                if tmp[tmp_index] > limited_num:
                    true_positives += 1
                tmp_index += 1

        false_positives = len(res_predicted_set) - true_positives
        false_negatives = len(res_groundtruth_set) - true_positives if true_positives < len(res_groundtruth_set) else 0

        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
        precision = (
            true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
        )
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as ofile:
            ofile.write(json.dumps(tmp))
            ofile.close()

        return {
            "predict_count": len(res_predicted_set) / 24,
            "groundtruth_count": len(res_groundtruth_set) / 24,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

    @staticmethod
    def partCompare_sematicSimilarity_project(examples: list, method: str) -> dict:
        similarity_calculator = BERTStringSimilarity()

        res_predicted_set = set()
        res_groundtruth_set = set()

        true_positives = 1

        cache_path = os.path.join(_cache_dir(), f"{method}_similarity_project.json")
        try:
            with open(cache_path, "r") as ifile:
                content = ifile.read()
                if content:
                    tmp = json.loads(content)
                else:
                    tmp = []
        except FileNotFoundError:
            tmp = []

        exist_flag = True if tmp else False
        tmp_index = 0 if exist_flag else None

        for example in examples:
            method_text = example.get(method)
            code_text = example.get("code")
            if not isinstance(method_text, str) or not isinstance(code_text, str):
                continue

            predicted_set = set(partComponentDepose(method_text))
            groundTruth_set = set(partComponentDepose(code_text))

            res_predicted_set = res_predicted_set | predicted_set
            res_groundtruth_set = res_groundtruth_set | groundTruth_set

        if not res_groundtruth_set or not res_predicted_set:
            return {
                "predict_count": len(res_predicted_set),
                "groundtruth_count": len(res_groundtruth_set),
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
            }

        for element in tqdm(res_predicted_set):
            if exist_flag and tmp_index is not None and tmp_index >= len(tmp):
                exist_flag = False
                tmp_index = None

            if not exist_flag:
                max_similarity = 0
                for element_ans in res_groundtruth_set:
                    sim_ans = similarity_calculator.calculate_similarity(element, element_ans)
                    if sim_ans > max_similarity:
                        max_similarity = sim_ans
                tmp.append(max_similarity)
            else:
                max_similarity = tmp[tmp_index]
                tmp_index += 1
            if max_similarity > 0.85:
                true_positives += 1
        false_positives = len(res_predicted_set) - true_positives
        false_negatives = len(res_groundtruth_set) - true_positives if true_positives < len(res_groundtruth_set) else 0

        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
        precision = (
            true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
        )
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as ofile:
            ofile.write(json.dumps(tmp))
            ofile.close()

        return {
            "predict_count": len(res_predicted_set),
            "groundtruth_count": len(res_groundtruth_set),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

    @staticmethod
    def pipeline_project(indir, func=partCompare_sematicSimilarity_project_qdrant):
        with open(indir, "r") as ifile:
            json_text = ifile.read()
            ifile.close()
        json_list = json.loads(json_text)
        print("MBSE_code_ds:\t\t", func(json_list, "MBSE_code_ds", 0.955))
        print("MBSE_code_gpt:\t\t", func(json_list, "MBSE_code_gpt", 0.955))
        print("Chatgpt-4.1:\t\t", func(json_list, "Chatgpt-4.1", 0.955))

    @staticmethod
    def pipeline_view(indir, func=partCompare_sematicSimilarity):
        with open(indir, "r") as ifile:
            json_text = ifile.read()
            ifile.close()
        json_list = json.loads(json_text)
        print("Element count:\n", exampleComparison.partCount(json_list))
        print("Average generation time:\n", exampleComparison.viewTime(json_list))
        print("MBSE_code_ds:\t\t", func(json_list, "MBSE_code_ds", 0.9))
        print("MBSE_code_gpt:\t\t", func(json_list, "MBSE_code_gpt", 0.9))

    @staticmethod
    def pipeline_view_indiffrent_stage(
        indir: str = "pipeline/examples/data.pkl",
        func=partCompare_sematicSimilarity_indiffrent_stage,
        res_json_path: str = "AutoMBSE/out/views/res.json",
    ):
        with open(res_json_path, "r") as ifile:
            json_text = ifile.read()
            ifile.close()
        json_list = json.loads(json_text)
        exampleComparison.data_loader(json_list, "Sysml_v2_model_creator", 0.85)
        with open(indir, "rb") as file:
            json_list_pick = pickle.load(file)
            file.close()
        print_nicely(func(json_list_pick))

    @staticmethod
    def extract_mbse_code_blocks(
        input_path: str = "AutoMBSE/out/views/res.json",
        output_path: str = "AutoMBSE/out/views/mbse_code_blocks.json",
    ):
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())

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
                        code_blocks = re.findall(r"```sysml\\n(.*?)\\n```", content, re.DOTALL)
                        if code_blocks:
                            item_blocks[method] = code_blocks
                if len(item_blocks) > 1:
                    all_code_blocks.append(item_blocks)

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_code_blocks, f, ensure_ascii=False, indent=2)

            print(f"Extraction complete! Processed code blocks for {len(all_code_blocks)} items.")
            return all_code_blocks

        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return []

    @staticmethod
    def calculate_bleu_scores(
        indir: str,
        ngram_weights: list = [0.7, 0.2, 0.1],
        bert_weight: float = 0,
        max_ngram: int = 3,
        ele_weight: float = 0.3,
        view_weight: float = 0.2,
        output_path: str = "AutoMBSE/out/views/res_bleu.json",
    ) -> list:
        import math
        from scipy.spatial.distance import cosine

        from ..sysml.code_blocks import extractStageWoStage

        with open(indir, "r") as ifile:
            json_text = ifile.read()
            ifile.close()
        examples = json.loads(json_text)
        similarity_calculator = BERTStringSimilarity()

        methods = [
            "MBSE_code_ds",
            "Sysml_v2_model_creator",
            "codeGEN",
            "ttool",
            "Chatgpt-4o",
            "MBSE_code_gpt",
            "Chatgpt-4.1",
            "ttool-4.1",
            "MBSE_wo_req",
            "LLM_wo_fot",
            "LLM_wo_verify",
        ]

        methods = ["MBSE_wo_req", "MBSE_wo_fot", "MBSE_wo_verify"]

        def get_ngrams(text, n):
            words = text.split()
            return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

        def calculate_precision(candidate, reference, n):
            candidate_ngrams = get_ngrams(candidate, n)
            reference_ngrams = get_ngrams(reference, n)

            if not candidate_ngrams:
                print("zero, candidate: ", candidate)
                return 0.0

            matches = sum(1 for ngram in candidate_ngrams if ngram in reference_ngrams)
            return matches / len(candidate_ngrams)

        def calculate_bp(candidate, reference):
            c = len(candidate.split())
            r = len(reference.split())
            if c > r:
                return 1.0
            return math.exp(1 - r / c)

        for example in tqdm(examples, desc="Processing examples", position=0):
            reference_code = example.get("code", "")
            reference_embedding = similarity_calculator.get_bert_embedding(reference_code)

            for method in tqdm(methods, desc="Processing methods", leave=False, position=1):
                if method in example:
                    generated_code = example[method]
                    code_blocks = re.findall(r"```sysml\\n(.*?)\\n```", generated_code, re.DOTALL)
                    if code_blocks:
                        generated_text = "\n".join(code_blocks)
                        generated_embedding = similarity_calculator.get_bert_embedding(generated_text)

                        bert_similarity = 1 - cosine(reference_embedding, generated_embedding)

                        precisions = []
                        for n in range(1, max_ngram + 1):
                            p_n = calculate_precision(generated_text, reference_code, n)
                            precisions.append(p_n)

                        bp = calculate_bp(generated_text, reference_code)
                        log_precisions = [math.log(p) if p > 0 else float("-inf") for p in precisions]
                        bleu_score = bp * math.exp(sum(w * p for w, p in zip(ngram_weights, log_precisions)))

                        final_score = ele_weight + view_weight
                        final_score = (1 - bert_weight) * bleu_score + bert_weight * bert_similarity

                        example[f"{method}_bleu"] = float(final_score)
                    else:
                        example[f"{method}_bleu"] = 0.0

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

        _ = extractStageWoStage
        return examples

    @staticmethod
    def calculate_similarity(indir: str, output_path: str = "AutoMBSE/out/views/res_simularity.json") -> list:
        from scipy.spatial.distance import cosine

        with open(indir, "r") as ifile:
            json_text = ifile.read()
            ifile.close()
        examples = json.loads(json_text)
        similarity_calculator = BERTStringSimilarity()

        methods = [
            "LLM_Case_code",
            "MBSE_code_ds",
            "Sysml_v2_model_creator",
            "codeGEN",
            "ttool",
            "MBSE_code_gpt",
            "Chatgpt-4.1",
            "ttool-4.1",
            "MBSE_wo_req",
            "MBSE_wo_fot",
            "MBSE_wo_verify",
        ]

        for example in tqdm(examples):
            reference_code = example.get("code", "")
            reference_embedding = similarity_calculator.get_bert_embedding(reference_code)

            for method in methods:
                if method in example:
                    generated_code = example[method]
                    code_blocks = re.findall(r"```sysml\\n(.*?)\\n```", generated_code, re.DOTALL)
                    if code_blocks:
                        generated_text = "\n".join(code_blocks)
                        generated_embedding = similarity_calculator.get_bert_embedding(generated_text)

                        final_score = 1 - cosine(reference_embedding, generated_embedding)

                        example[f"{method}_similarity"] = float(final_score)
                    else:
                        example[f"{method}_similarity"] = 0.0

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

        return examples

    @staticmethod
    def calculate_average_bleu_scores(indir: str = "AutoMBSE/out/views/res_bleu.json", ignore_zero: bool = True):
        try:
            with open(indir, "r", encoding="utf-8") as f:
                data = json.load(f)

            methods = ["MBSE_wo_req", "MBSE_wo_fot", "MBSE_wo_verify"]

            method_scores = {method: {"sum": 0.0, "count": 0} for method in methods}

            for item in data:
                for method in methods:
                    bleu_key = f"{method}_bleu"
                    if bleu_key in item:
                        score = item[bleu_key]
                        if not ignore_zero or (ignore_zero and score != 0.0):
                            method_scores[method]["sum"] += score
                            method_scores[method]["count"] += 1

            print("\nAverage BLEU score by method:")
            print("-" * 50)
            print(f"{'Method':<30} {'Avg BLEU':<15} {'Samples':<10} {'Valid':<10}")
            print("-" * 50)

            for method in methods:
                if method_scores[method]["count"] > 0:
                    avg_score = method_scores[method]["sum"] / method_scores[method]["count"]
                    valid_count = sum(
                        1 for item in data if f"{method}_bleu" in item and item[f"{method}_bleu"] != 0.0
                    )
                    print(
                        f"{method:<30} {avg_score:<15.4f} {method_scores[method]['count']:<10} {valid_count:<10}"
                    )

            print("-" * 50)

            total_sum = sum(score["sum"] for score in method_scores.values())
            total_count = sum(score["count"] for score in method_scores.values())
            if total_count > 0:
                overall_avg = total_sum / total_count
                print(f"\nOverall average BLEU: {overall_avg:.4f}")

                if ignore_zero:
                    total_valid = sum(
                        1
                        for item in data
                        for method in methods
                        if f"{method}_bleu" in item and item[f"{method}_bleu"] != 0.0
                    )
                    print(f"Total valid samples: {total_valid}")

        except FileNotFoundError:
            print("Error: file not found")
        except json.JSONDecodeError:
            print("Error: invalid file format")
        except Exception as e:
            print(f"Error processing file: {str(e)}")

    @staticmethod
    def calculate_average_similarity(indir: str = "AutoMBSE/out/views/res_simularity.json", ignore_zero: bool = True):
        try:
            with open(indir, "r", encoding="utf-8") as f:
                data = json.load(f)

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
                "Chatgpt-4.1",
                "ttool-4.1",
                "MBSE_wo_req",
                "MBSE_wo_fot",
                "MBSE_wo_verify",
            ]

            method_scores = {method: {"sum": 0.0, "count": 0} for method in methods}

            for item in data:
                for method in methods:
                    bleu_key = f"{method}_similarity"
                    if bleu_key in item:
                        score = item[bleu_key]
                        if not ignore_zero or (ignore_zero and score != 0.0):
                            method_scores[method]["sum"] += score
                            method_scores[method]["count"] += 1

            print("\nAverage similarity score by method:")
            print("-" * 50)
            print(f"{'Method':<30} {'Avg sim':<15} {'Samples':<10} {'Valid':<10}")
            print("-" * 50)

            for method in methods:
                if method_scores[method]["count"] > 0:
                    avg_score = method_scores[method]["sum"] / method_scores[method]["count"]
                    valid_count = sum(
                        1
                        for item in data
                        if f"{method}_similarity" in item and item[f"{method}_similarity"] != 0.0
                    )
                    print(
                        f"{method:<30} {avg_score:<15.4f} {method_scores[method]['count']:<10} {valid_count:<10}"
                    )

            print("-" * 50)

            total_sum = sum(score["sum"] for score in method_scores.values())
            total_count = sum(score["count"] for score in method_scores.values())
            if total_count > 0:
                overall_avg = total_sum / total_count
                print(f"\nOverall average sim: {overall_avg:.4f}")

                if ignore_zero:
                    total_valid = sum(
                        1
                        for item in data
                        for method in methods
                        if f"{method}_similarity" in item and item[f"{method}_similarity"] != 0.0
                    )
                    print(f"Total valid samples: {total_valid}")

        except FileNotFoundError:
            print("Error: file not found")
        except json.JSONDecodeError:
            print("Error: invalid file format")
        except Exception as e:
            print(f"Error processing file: {str(e)}")

    @staticmethod
    def calculate_bleu_scores_project(
        indir: str,
        ngram_weights: list = [0.6, 0.3, 0.1],
        bert_weight: float = 0.2,
        max_ngram: int = 3,
        created_flag: bool = False,
    ) -> dict:
        import math

        from ..sysml.code_blocks import extractStageWoStage

        with open(indir, "r") as ifile:
            json_text = ifile.read()
            ifile.close()
        examples = json.loads(json_text)

        similarity_calculator = BERTStringSimilarity()
        res_predicted_set = ""
        res_groundtruth_set = ""

        methods = ["MBSE_code_gpt", "MBSE_code_ds"]
        methods = ["LLM_Case_code", "MBSE_code_ds", "Sysml_v2_model_creator", "codeGEN", "ttool", "MBSE_code_gpt", "Chatgpt-4.1", "ttool-4.1"]
        res = []
        res_method = []
        for method in methods:
            for example in examples:
                predicted_set = "\n".join(extractStageWoStage(example[method]))
                groundTruth_set = example["code"]

                res_predicted_set = res_predicted_set + predicted_set
                res_groundtruth_set = res_groundtruth_set + groundTruth_set

            def get_ngrams(text, n):
                words = text.split()
                return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

            def calculate_precision(candidate, reference, n):
                candidate_ngrams = get_ngrams(candidate, n)
                reference_ngrams = get_ngrams(reference, n)

                if not candidate_ngrams:
                    return 0.0

                matches = sum(1 for ngram in candidate_ngrams if ngram in reference_ngrams)
                return matches / len(candidate_ngrams)

            def calculate_bp(candidate, reference):
                c = len(candidate.split())
                r = len(reference.split())
                if c > r:
                    return 1.0
                return math.exp(1 - r / c)

            def sigmoid(x, a=10):
                return 1 / (1 + math.exp(-a * (x - 0.5)))

            _ = sigmoid
            _ = created_flag

            pred = res_predicted_set
            reference = res_groundtruth_set

            bert_similarity = similarity_calculator.calculate_similarity(pred, reference)

            precisions = []
            for n in range(1, max_ngram + 1):
                p_n = calculate_precision(pred, reference, n)
                precisions.append(p_n)

            bp = calculate_bp(pred, reference)
            log_precisions = [math.log(p) if p > 0 else float("-inf") for p in precisions]
            bleu_score = bp * math.exp(sum(w * p for w, p in zip(ngram_weights, log_precisions)))

            final_score = (1 - bert_weight) * bleu_score + bert_weight * bert_similarity

            res.append(final_score)
            res_method.append(method)

            print(f"method: {method}, score: {final_score}")

        final = []
        for idx in range(len(res)):
            final.append({"method": res_method[idx], "bleu": float(res[idx])})

        return final

    @staticmethod
    def calculate_simularity_project(indir: str, alpha: float = 0.92) -> dict:
        from ..sysml.code_blocks import extractStageWoStage

        with open(indir, "r") as ifile:
            json_text = ifile.read()
            ifile.close()
        examples = json.loads(json_text)

        similarity_calculator = BERTStringSimilarity()

        methods = ["LLM_Case_code", "MBSE_code_ds", "Sysml_v2_model_creator", "codeGEN", "ttool", "Chatgpt-4o", "MBSE_code_gpt", "Chatgpt-4.1", "ttool-4.1"]
        res = []
        res_method = []
        for method in methods:
            res_predicted_set = ""
            res_groundtruth_set = ""
            for example in examples:
                predicted_set = "\n".join(extractStageWoStage(example[method]))
                groundTruth_set = "\n" + example["code"]

                res_predicted_set = res_predicted_set + predicted_set
                res_groundtruth_set = res_groundtruth_set + groundTruth_set

            pred = res_predicted_set
            reference = res_groundtruth_set

            final_score = similarity_calculator.calculate_similarity(pred, reference) * alpha

            res.append(final_score)
            res_method.append(method)

            print(f"method: {method}, score: {final_score}")

        final = []
        for idx in range(len(res)):
            final.append({"method": res_method[idx], "bleu": float(res[idx])})
        return final


__all__ = ["exampleComparison"]
