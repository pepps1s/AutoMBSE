from __future__ import annotations

import json
import math


def process_bleu_scores(input_file="AutoMBSE/out/bleu_res.json", a=10, diff=0.1):
    def sigmoid(x, a=10):
        return 1 / (1 + math.exp(-a * (x - 0.5)))

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Input file must contain a JSON array")

        processed_data = []
        for item in data:
            if "bleu" not in item:
                continue

            original_bleu = item["bleu"]
            if original_bleu < 0 or original_bleu > 1:
                raise ValueError(f"BLEU value {original_bleu} is out of range [0, 1]")

            stretched_bleu = sigmoid(original_bleu - diff, a)
            processed_item = item.copy()
            processed_item["bleu"] = stretched_bleu
            processed_data.append(processed_item)

        print("\nProcessed BLEU scores:")
        for item in processed_data:
            print(f"Method: {item.get('method', 'Unknown')}, BLEU: {item['bleu']:.4f}")

        return processed_data

    except FileNotFoundError:
        print(f"Error: file not found: {input_file}")
    except json.JSONDecodeError:
        print("Error: invalid JSON file format")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

    return []


__all__ = ["process_bleu_scores"]
