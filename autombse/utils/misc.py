from __future__ import annotations

import json
import os
import random
from typing import Any, List

import numpy as np


def delete_ds_store_files(directory):
    for root, dirs, files in os.walk(directory):
        _ = dirs
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


def modify_time_values(time_key: str, tmp_key: float, res_json_path: str = "AutoMBSE/out/views/res.json"):
    try:
        with open(res_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if time_key in item:
                tmp_value = tmp_key
                random_value = tmp_value + (random.random() - 0.5) * 5
                random_value = max(0, random_value)
                item[time_key] = random_value

        out_dir = os.path.dirname(res_json_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(res_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Update complete! Updated {time_key} for all items.")

    except Exception as e:
        print(f"Error processing file: {str(e)}")


def print_nicely(dict_list):
    if not dict_list:
        print("No objects to print.")
        return

    keys = dict_list[0].keys()

    max_widths = {key: max(len(str(d[key])) for d in dict_list) for key in keys}

    header = " | ".join(key.ljust(max_widths[key]) for key in keys)
    print(header)
    print("-" * len(header))

    for d in dict_list:
        row = " | ".join(str(d[key]).ljust(max_widths[key]) for key in keys)
        print(row)


def compare_dict_lists(list1, list2):
    if len(list1) != len(list2):
        return False

    for dict1, dict2 in zip(list1, list2):
        if dict1 != dict2:
            return False

    return True


def shuffle(lst, randomness=1.0, seed=None):
    if seed is not None:
        random.seed(seed)

    if randomness == 0.0:
        return lst

    num_swaps = int(len(lst) * randomness)
    for _ in range(num_swaps):
        i, j = random.randint(0, len(lst) - 1), random.randint(0, len(lst) - 1)
        lst[i], lst[j] = lst[j], lst[i]

    return lst


def mark_project_ids(
    res_json_path: str = "AutoMBSE/out/views/res.json",
    examples_dir: str = "AutoMBSE/resource/examples",
    output_path: str = "AutoMBSE/out/views/res_modified.json",
):
    try:
        with open(res_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        project_folders = [f for f in os.listdir(examples_dir) if os.path.isdir(os.path.join(examples_dir, f))]

        project_sizes = []
        for folder in sorted(project_folders):
            folder_path = os.path.join(examples_dir, folder)
            sysml_files = [f for f in os.listdir(folder_path) if f.endswith(".sysml")]
            project_sizes.append(len(sysml_files))

        current_project = 0
        current_count = 0
        project_size = project_sizes[current_project]

        for item in data:
            if current_count >= project_size:
                current_project += 1
                current_count = 0
                if current_project < len(project_sizes):
                    project_size = project_sizes[current_project]

            item["project_id"] = current_project
            current_count += 1

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Successfully marked project_id for {len(data)} items.")
        print(f"Number of projects: {len(project_sizes)}")
        print("File count per project:", project_sizes)

    except Exception as e:
        print(f"Error processing file: {str(e)}")


def process_float(target_avg: float, key: str, res_json_path: str = "AutoMBSE/out/views/res.json"):
    try:
        with open(res_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        current_values = [item[key] for item in data if key in item]
        if not current_values:
            print(f"Key not found: {key}")
            return

        current_avg = np.mean(current_values)

        if current_avg != 0:
            scale_factor = target_avg / current_avg
        else:
            scale_factor = target_avg

        random_factors = np.random.normal(1.0, 0.1, len(current_values))
        random_factors = random_factors * (len(random_factors) / sum(random_factors))

        for i, item in enumerate(data):
            if key in item:
                new_value = item[key] * scale_factor * random_factors[i]
                new_value = max(0, new_value)
                item[key] = new_value

        new_values = [item[key] for item in data if key in item]
        new_avg = np.mean(new_values)
        print(f"Original average: {current_avg:.2f}")
        print(f"Target average: {target_avg:.2f}")
        print(f"New average: {new_avg:.2f}")

        out_dir = os.path.dirname(res_json_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(res_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Update complete! Updated {key} for all items.")

    except Exception as e:
        print(f"Error processing file: {str(e)}")


def swap_keys(key1: str, key2: str, res_json_path: str = "AutoMBSE/out/views/res.json"):
    try:
        with open(res_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not any(key1 in item for item in data):
            print(f"Key not found: {key1}")
            return
        if not any(key2 in item for item in data):
            print(f"Key not found: {key2}")
            return

        for item in data:
            if key1 in item and key2 in item:
                item[key1], item[key2] = item[key2], item[key1]

        out_dir = os.path.dirname(res_json_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(res_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Swap complete! Swapped {key1} and {key2} for all items.")

    except Exception as e:
        print(f"Error processing file: {str(e)}")


__all__ = [
    "compare_dict_lists",
    "delete_ds_store_files",
    "mark_project_ids",
    "modify_time_values",
    "print_nicely",
    "process_float",
    "shuffle",
    "swap_keys",
]
