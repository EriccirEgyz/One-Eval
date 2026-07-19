#!/usr/bin/env python
"""
GAIA benchmark evaluation bridge script for One-Eval.

Uses smolagents (HuggingFace official) to run an agent loop with tool-calling,
then scores predictions against ground truth using GAIA's official scoring logic.

Flow:
  1. Load GAIA validation set from HuggingFace
  2. For each question, run a smolagents CodeAgent with web search + python
  3. Score predictions using GAIA's official exact-match scorer
  4. Output results in One-Eval's standardized format

Model config from environment variables:
  OPENAI_API_KEY, OPENAI_API_BASE, ONEEVAL_MODEL_NAME, ONEEVAL_MAX_SAMPLES
"""

import argparse
import json
import logging
import os
import re
import string
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gaia_oneeval")


# ============================================================
# GAIA Official Scorer (ported from gaia-benchmark/leaderboard)
# ============================================================

def normalize_number_str(number_str: str) -> float:
    number_str = number_str.replace(",", "")
    number_str = number_str.replace("$", "")
    number_str = number_str.replace("%", "")
    number_str = number_str.strip()
    return float(number_str)


def is_float(s: str) -> bool:
    try:
        normalize_number_str(s)
        return True
    except (ValueError, AttributeError):
        return False


def normalize_str(s: str) -> str:
    s = s.strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", "", s)
    return s


def split_list_answer(s: str) -> list:
    if ";" in s:
        parts = s.split(";")
    else:
        parts = s.split(",")
    return [p.strip() for p in parts if p.strip()]


def question_scorer(prediction: str, ground_truth: str) -> bool:
    if prediction is None:
        prediction = "None"
    prediction = str(prediction).strip()
    ground_truth = str(ground_truth).strip()

    if is_float(ground_truth):
        try:
            return normalize_number_str(prediction) == normalize_number_str(ground_truth)
        except (ValueError, AttributeError):
            return False

    gt_parts = split_list_answer(ground_truth)
    if len(gt_parts) > 1:
        pred_parts = split_list_answer(prediction)
        if len(pred_parts) != len(gt_parts):
            return False
        for pred_elem, gt_elem in zip(pred_parts, gt_parts):
            if is_float(gt_elem):
                try:
                    if normalize_number_str(pred_elem) != normalize_number_str(gt_elem):
                        return False
                except (ValueError, AttributeError):
                    return False
            else:
                if normalize_str(pred_elem) != normalize_str(gt_elem):
                    return False
        return True

    return normalize_str(prediction) == normalize_str(ground_truth)


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GAIA evaluation for One-Eval")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default=os.environ.get("ONEEVAL_MODEL_NAME", "gpt-4o"))
    parser.add_argument("--api_base", type=str,
                        default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--max_samples", type=int,
                        default=int(os.environ.get("ONEEVAL_MAX_SAMPLES", "-1")))
    parser.add_argument("--split", type=str,
                        default=os.environ.get("ONEEVAL_SPLIT", "validation"))
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Max agent reasoning steps per question")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per question in seconds")
    return parser.parse_args()


def load_gaia_data(split: str, max_samples: int):
    """Load GAIA dataset from HuggingFace."""
    from datasets import load_dataset

    log.info(f"Loading GAIA dataset (split={split})...")
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split=split)

    data = []
    for item in ds:
        data.append({
            "task_id": item["task_id"],
            "question": item["Question"],
            "level": item.get("Level", 1),
            "final_answer": item.get("Final answer", ""),
            "file_name": item.get("file_name", ""),
            "file_path": item.get("file_path", ""),
        })

    if max_samples > 0:
        data = data[:max_samples]
        log.info(f"Limited to {len(data)} samples")
    else:
        log.info(f"Loaded {len(data)} samples")

    return data


def build_agent(args):
    """Build a smolagents CodeAgent with web search and python tools."""
    from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool, OpenAIModel

    model = OpenAIModel(
        model_id=args.model_name,
        api_base=args.api_base or None,
        api_key=args.api_key or None,
        max_completion_tokens=8192,
    )

    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model,
        max_steps=args.max_steps,
        verbosity_level=1,
    )

    return agent


def build_prompt(question: str, file_name: str = "") -> str:
    """Build the prompt for GAIA questions."""
    prompt = (
        "Answer the following question. Your final answer should be concise and exact — "
        "a number, a short phrase, or a comma-separated list. Do NOT include explanations "
        "in your final answer.\n\n"
        f"Question: {question}"
    )
    if file_name:
        prompt += f"\n\n[Note: This question references an attached file: {file_name}]"
    return prompt


def run_agent_on_question(agent, question_data: dict, timeout: int) -> tuple:
    """Run the agent on a single question and return (predicted_answer, agent_logs)."""
    import signal
    import threading

    prompt = build_prompt(question_data["question"], question_data.get("file_name", ""))

    result = {"answer": None, "logs": [], "error": None}

    def _run():
        try:
            # Use return_full_result=True to get RunResult with step logs
            run_result = agent.run(prompt, return_full_result=True)
            result["answer"] = str(run_result.output) if run_result.output is not None else ""

            # Extract step logs from RunResult.steps
            if hasattr(run_result, 'steps') and run_result.steps:
                for i, step in enumerate(run_result.steps):
                    if not isinstance(step, dict):
                        continue

                    log_entry = {"step": i + 1}

                    # Extract Thought + Code
                    if "model_output" in step and step["model_output"]:
                        log_entry["thought_and_code"] = step["model_output"][:800]

                    # Extract executed code
                    if "code_action" in step and step["code_action"]:
                        log_entry["code"] = step["code_action"][:500]

                    # Extract observations (tool outputs)
                    if "observations" in step and step["observations"]:
                        log_entry["observation"] = step["observations"][:500]

                    # Only add steps that have actual content
                    if len(log_entry) > 1:
                        result["logs"].append(log_entry)
        except Exception as e:
            result["error"] = str(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        log.warning(f"  Timeout after {timeout}s")
        return "", []

    if result["error"]:
        log.warning(f"  Agent error: {result['error']}")
        return "", []

    return result["answer"] or "", result["logs"]


def evaluate(data: list, predictions: dict) -> dict:
    """Score all predictions and compute metrics."""
    total = defaultdict(int)
    correct = defaultdict(int)

    for item in data:
        task_id = item["task_id"]
        level = item["level"]
        ground_truth = item["final_answer"]
        pred_entry = predictions.get(task_id, {})
        # Handle both old format (string) and new format (dict with answer/logs)
        if isinstance(pred_entry, str):
            prediction = pred_entry
        else:
            prediction = pred_entry.get("answer", "")

        level_key = f"level{level}"
        total["overall"] += 1
        total[level_key] += 1

        if question_scorer(prediction, ground_truth):
            correct["overall"] += 1
            correct[level_key] += 1

    results = {}
    for key in total:
        results[key] = {
            "total": total[key],
            "correct": correct[key],
            "accuracy": correct[key] / total[key] * 100 if total[key] > 0 else 0.0,
        }

    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("GAIA One-Eval Bridge")
    log.info(f"  Model: {args.model_name}")
    log.info(f"  Split: {args.split}")
    log.info(f"  Max samples: {args.max_samples}")
    log.info(f"  Max steps: {args.max_steps}")
    log.info("=" * 60)

    # 1. Load data
    data = load_gaia_data(args.split, args.max_samples)

    # 2. Build agent
    log.info("Building agent...")
    agent = build_agent(args)

    # 3. Run agent on each question (with resume support)
    predictions_file = os.path.join(args.output_dir, "predictions.json")
    predictions = {}
    if os.path.exists(predictions_file):
        with open(predictions_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        log.info(f"Resumed from {len(predictions)} existing predictions")

    for idx, item in enumerate(data, 1):
        task_id = item["task_id"]
        if task_id in predictions:
            log.info(f"[{idx}/{len(data)}] Skipping {task_id} (cached)")
            continue

        log.info(f"[{idx}/{len(data)}] Level {item['level']} | {item['question'][:80]}...")
        answer, logs = run_agent_on_question(agent, item, args.timeout)
        predictions[task_id] = {"answer": answer, "logs": logs}
        log.info(f"  -> {answer[:100]}")

        if idx % 5 == 0:
            with open(predictions_file, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)

    # Save final predictions
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # 4. Evaluate
    log.info("Evaluating predictions...")
    results = evaluate(data, predictions)

    # 5. Build per-sample detail JSONL
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detail_name = f"samples_{timestamp}.jsonl"
    detail_path = os.path.join(args.output_dir, detail_name)

    with open(detail_path, "w", encoding="utf-8") as fout:
        for item in data:
            task_id = item["task_id"]
            question = item["question"]
            level = item.get("level", "")
            file_name = item.get("file_name", "")
            final_answer = item["final_answer"]
            pred_entry = predictions.get(task_id, {})

            # Handle both old format (string) and new format (dict with answer/logs)
            if isinstance(pred_entry, str):
                prediction = pred_entry
                agent_logs = []
            else:
                prediction = pred_entry.get("answer", "")
                agent_logs = pred_entry.get("logs", [])

            # Build prompt
            prompt_text = question
            if file_name:
                prompt_text = f"[Attachment: {file_name}]\n{question}"

            # Score
            is_correct = question_scorer(prediction, final_answer)

            record = {
                "task_id": task_id,
                "level": f"Level {level}",
                "prompt": prompt_text,
                "solution": prediction,
                "ground_truth": final_answer,
                "eval_score": 1.0 if is_correct else 0.0,
                "eval_valid": True,
                "agent_logs": agent_logs,  # Agent execution trace
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_samples = len(data)
    log.info(f"Per-sample detail written to: {detail_path} ({total_samples} samples)")

    # 6. Write scores in One-Eval format
    score_output = {
        "average": {"accuracy": results["overall"]["accuracy"] / 100.0},  # Convert to 0-1 scale
        "total_samples": total_samples,
        "detail_path": detail_name,
        "by_level": {k: {
            "accuracy": v["accuracy"] / 100.0,
            "correct": v["correct"],
            "total": v["total"]
        } for k, v in results.items() if k != "overall"},
        "details": {
            "total": results["overall"]["total"],
            "correct": results["overall"]["correct"],
        },
    }

    score_file = os.path.join(args.output_dir, f"scores_{timestamp}.json")
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(score_output, f, indent=2, ensure_ascii=False)

    log.info("=" * 60)
    log.info("GAIA evaluation complete!")
    log.info(f"  Overall: {score_output['average']['accuracy']:.4f} "
             f"({results['overall']['correct']}/{results['overall']['total']})")
    for key in sorted(results):
        if key.startswith("level"):
            r = results[key]
            log.info(f"  {key}: {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
    log.info(f"  Results saved to: {args.output_dir}")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
