#!/usr/bin/env python
"""
BFCL (Berkeley Function Calling Leaderboard) evaluation bridge for One-Eval.

Flow:
  1. Read model config from One-Eval env vars
  2. Import BFCL's internal Python API
  3. Load test entries, truncate to max_samples
  4. Run generation via BFCL's handler system
  5. Run evaluation via BFCL's eval_runner
  6. Collect per-category scores into a standardized scores JSON

Model config from environment variables:
  OPENAI_API_KEY, OPENAI_API_BASE, ONEEVAL_MODEL_NAME, ONEEVAL_MAX_SAMPLES

Test category selection via:
  --test_category or BFCL_TEST_CATEGORIES env var (comma-separated or preset name)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bfcl_oneeval")

CATEGORY_PRESETS = {
    "ast": [
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
    ],
    "live": [
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "multi_turn": [
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    ],
    "non_live": [
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
    ],
    "single_turn": [
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "default": [
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
}
CATEGORY_PRESETS["all"] = (
    CATEGORY_PRESETS["ast"] + CATEGORY_PRESETS["live"] + CATEGORY_PRESETS["multi_turn"]
)


def parse_args():
    parser = argparse.ArgumentParser(description="BFCL evaluation for One-Eval")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.environ.get("ONEEVAL_MODEL_NAME", "gpt-4o"),
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=os.environ.get("OPENAI_API_BASE", ""),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=int(os.environ.get("ONEEVAL_MAX_SAMPLES", "-1")),
    )
    parser.add_argument(
        "--test_category",
        type=str,
        default=os.environ.get("BFCL_TEST_CATEGORIES", "default"),
        help="Comma-separated categories or a preset (ast/live/multi_turn/all/default)",
    )
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.001)
    return parser.parse_args()


def resolve_categories(category_str: str) -> list:
    """Resolve category string to a list of category names.

    Supports:
    - Our preset names (ast/live/multi_turn/all/default/non_live/single_turn)
    - BFCL native collection names (same as above, passed through as-is)
    - Comma-separated individual category names
    """
    category_str = category_str.strip()
    if category_str in CATEGORY_PRESETS:
        return CATEGORY_PRESETS[category_str]
    # Could be a comma-separated list of individual categories or BFCL collections
    return [c.strip() for c in category_str.split(",") if c.strip()]


def setup_env(args):
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.api_base:
        os.environ["OPENAI_API_BASE"] = args.api_base
        os.environ["OPENAI_BASE_URL"] = args.api_base


def register_model(model_name: str, temperature: float):
    """
    Register a custom model in BFCL's MODEL_CONFIG_MAPPING.
    Forces OpenAICompletionsHandler (chat/completions API) since most
    third-party OpenAI-compatible proxies don't support the newer Responses API.
    """
    try:
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
        from bfcl_eval.model_handler.api_inference.openai_completion import (
            OpenAICompletionsHandler,
        )
    except ImportError as e:
        log.warning(f"Could not import BFCL model config: {e}")
        return

    if model_name in MODEL_CONFIG_MAPPING:
        log.info(f"Model '{model_name}' already in MODEL_CONFIG_MAPPING, overriding to use CompletionsHandler")

    MODEL_CONFIG_MAPPING[model_name] = ModelConfig(
        model_name=model_name,
        display_name=model_name,
        url="",
        org="Custom",
        license="",
        model_handler=OpenAICompletionsHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
    )
    log.info(f"Registered model '{model_name}' with OpenAICompletionsHandler (FC mode)")


def run_generation(args, categories: list):
    """
    Run BFCL generation using internal API with dataset truncation support.
    """
    try:
        from bfcl_eval._llm_response_generation import (
            generate_results,
            get_involved_test_entries,
            collect_test_cases,
        )

        gen_args = argparse.Namespace(
            model=[args.model_name],
            test_category=categories,
            temperature=args.temperature,
            include_input_log=False,
            exclude_state_log=False,
            num_threads=args.num_threads,
            num_gpus=1,
            backend="vllm",
            gpu_memory_utilization=0.9,
            result_dir=None,
            run_ids=False,
            allow_overwrite=True,
            skip_server_setup=True,
            local_model_path=None,
            lora_modules=None,
            enable_lora=False,
            max_lora_rank=None,
        )

        # Try to set result_dir from BFCL's config
        try:
            from bfcl_eval.constants.eval_config import RESULT_PATH
            gen_args.result_dir = RESULT_PATH
        except ImportError:
            gen_args.result_dir = Path("result")

        log.info(f"Loading test entries for categories: {categories}")
        all_test_file_paths, all_test_categories, all_test_entries_involved = get_involved_test_entries(
            gen_args.test_category, gen_args.run_ids
        )

        total_entries = len(all_test_entries_involved)
        if args.max_samples > 0 and args.max_samples < total_entries:
            log.info(f"Truncating from {total_entries} to {args.max_samples} samples")
            all_test_entries_involved = all_test_entries_involved[: args.max_samples]

        log.info(f"Running generation on {len(all_test_entries_involved)} entries")

        test_cases_total = collect_test_cases(
            gen_args,
            args.model_name,
            all_test_categories,
            all_test_file_paths,
            deepcopy(all_test_entries_involved),
        )

        if test_cases_total:
            generate_results(gen_args, args.model_name, test_cases_total)
            log.info(f"Generation complete: {len(test_cases_total)} test cases processed")
        else:
            log.info("No new test cases to generate (all already cached)")

        return True

    except ImportError as e:
        log.warning(f"Internal API import failed: {e}")
        log.info("Falling back to CLI-based generation")
        return _run_generation_cli(args, categories)


def _run_generation_cli(args, categories: list) -> bool:
    """Fallback: run generation via subprocess CLI (bfcl_eval module)."""
    success = True
    for category in categories:
        log.info(f"Generating for category: {category}")
        cmd = [sys.executable, "-m", "bfcl_eval", "generate",
               "--model", args.model_name,
               "--test-category", category,
               "--num-threads", str(args.num_threads)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            log.error(f"Generation failed for {category}: {result.stderr[:300]}")
            success = False
        else:
            log.info(f"Generation complete for {category}")
    return success


def run_evaluation(args, categories: list):
    """Run BFCL evaluation on generated results."""
    try:
        from bfcl_eval.eval_checker.eval_runner import main as eval_main

        log.info("Running evaluation via internal API")
        eval_main(
            model=[args.model_name],
            test_categories=categories,
            result_dir=None,
            score_dir=None,
        )
        return True

    except (ImportError, TypeError, Exception) as e:
        log.warning(f"Internal eval API failed: {e}")
        log.info("Falling back to CLI-based evaluation")
        return _run_evaluation_cli(args, categories)


def _run_evaluation_cli(args, categories: list) -> bool:
    """Fallback: run evaluation via subprocess CLI (bfcl_eval module)."""
    success = True
    for category in categories:
        log.info(f"Evaluating category: {category}")
        cmd = [sys.executable, "-m", "bfcl_eval", "evaluate",
               "--model", args.model_name,
               "--test-category", category]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            log.error(f"Evaluation failed for {category}: {result.stderr[:300]}")
            success = False
        else:
            log.info(f"Evaluation complete for {category}")
    return success


def collect_scores(model_name: str) -> dict:
    """
    Collect evaluation scores from BFCL's output directories.
    BFCL writes:
    - Per-category JSONs: {SCORE_PATH}/{model_name}/BFCL_v3_{category}_score.json
      Format: {"accuracy": 0.9, "correct_count": 9, "total_count": 10}
    - Summary CSVs: {SCORE_PATH}/data_overall.csv, data_non_live.csv, etc.
      Values use percentage format with % sign (e.g. "90.00%")
    """
    scores = {}

    # Resolve BFCL's actual score directory
    try:
        from bfcl_eval.constants.eval_config import SCORE_PATH
        score_base = SCORE_PATH
    except ImportError:
        score_base = Path("score")

    log.info(f"Looking for scores in: {score_base}")

    # 1. Read per-category score JSONs (most reliable)
    score_model_dir = score_base / model_name
    if not score_model_dir.exists():
        score_model_dir = score_base / model_name.replace("/", "_")

    if score_model_dir.exists():
        for json_file in score_model_dir.glob("*_score.json"):
            try:
                first_line = json_file.read_text(encoding="utf-8").strip().split("\n")[0]
                data = json.loads(first_line)
                if isinstance(data, dict) and "accuracy" in data:
                    category = json_file.stem.replace("BFCL_v3_", "").replace("_score", "")
                    scores[category] = data["accuracy"]
                    log.info(f"  {category}: {data['accuracy']} ({data.get('correct_count', '?')}/{data.get('total_count', '?')})")
            except (json.JSONDecodeError, UnicodeDecodeError, IndexError):
                continue
    else:
        log.warning(f"Score model dir not found: {score_model_dir}")

    # 2. If no JSONs found, try CSV summaries
    if not scores:
        csv_files = list(score_base.glob("data_*.csv")) if score_base.exists() else []
        for csv_file in csv_files:
            try:
                import csv as csv_mod
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv_mod.DictReader(f)
                    for row in reader:
                        row_model = row.get("Model", row.get("model", ""))
                        if model_name.lower() in row_model.lower():
                            for k, v in row.items():
                                if k.lower() in ("rank", "model", ""):
                                    continue
                                try:
                                    v_str = v.strip()
                                    if v_str.endswith("%"):
                                        scores[k] = float(v_str[:-1]) / 100.0
                                    elif v_str and v_str != "N/A":
                                        scores[k] = float(v_str)
                                except (ValueError, TypeError):
                                    pass
                            break
            except Exception as e:
                log.warning(f"Failed to parse {csv_file}: {e}")

    return scores


def compute_overall(scores: dict) -> float:
    """Compute overall accuracy, excluding non-accuracy fields like cost/latency."""
    excluded_keys = {"total cost", "latency", "cost", "rank", "overall"}
    numeric = []
    for k, v in scores.items():
        if isinstance(v, (int, float)):
            k_lower = k.lower()
            if not any(excl in k_lower for excl in excluded_keys):
                numeric.append(v)
    if numeric:
        return sum(numeric) / len(numeric)
    return 0.0


def write_oneeval_scores(args, scores: dict, categories: list):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = compute_overall(scores)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Resolve BFCL paths
    try:
        from bfcl_eval.constants.eval_config import SCORE_PATH, RESULT_PATH
        score_base = SCORE_PATH
        result_base = RESULT_PATH
    except ImportError:
        score_base = Path("score")
        result_base = Path("result")

    score_model_dir = score_base / args.model_name
    if not score_model_dir.exists():
        score_model_dir = score_base / args.model_name.replace("/", "_")

    result_model_dir = result_base / args.model_name
    if not result_model_dir.exists():
        result_model_dir = result_base / args.model_name.replace("/", "_")

    detail_name = f"samples_{timestamp}.jsonl"
    detail_path = output_dir / detail_name
    total_samples = 0

    # Per-category accuracy lookup for fallback scoring
    category_accuracy = {}
    if score_model_dir.exists():
        for json_file in sorted(score_model_dir.glob("*_score.json")):
            try:
                lines = json_file.read_text(encoding="utf-8").strip().split("\n")
                summary = json.loads(lines[0])
                category = json_file.stem.replace("BFCL_v3_", "").replace("_score", "")
                category_accuracy[category] = summary.get("accuracy", 0.0)
            except (json.JSONDecodeError, OSError, IndexError):
                continue

    with open(detail_path, "w", encoding="utf-8") as fout:
        # Strategy 1: Read per-entry detail from score files (if available)
        if score_model_dir.exists():
            for json_file in sorted(score_model_dir.glob("*_score.json")):
                try:
                    lines = json_file.read_text(encoding="utf-8").strip().split("\n")
                    if len(lines) > 1:
                        for line in lines[1:]:
                            entry = json.loads(line)
                            prompt_data = entry.get("prompt", {})
                            question = prompt_data.get("question", [[]])
                            question_text = ""
                            if question and isinstance(question, list) and question[0]:
                                msgs = question[0] if isinstance(question[0], list) else question
                                for msg in msgs:
                                    if isinstance(msg, dict) and msg.get("role") == "user":
                                        question_text = msg.get("content", "")
                                        break
                            model_result = entry.get("model_result", [])
                            solution = json.dumps(model_result, ensure_ascii=False) if model_result else ""
                            record = {
                                "task_id": entry.get("id", ""),
                                "category": entry.get("test_category", ""),
                                "prompt": question_text,
                                "solution": solution,
                                "eval_score": 1.0 if entry.get("valid") else 0.0,
                                "eval_valid": True,
                                "error": entry.get("error", []),
                                "error_type": entry.get("error_type", ""),
                            }
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_samples += 1
                except (json.JSONDecodeError, OSError):
                    continue

        # Strategy 2: If no per-entry detail from score files, read result files
        if total_samples == 0 and result_model_dir.exists():
            # Load test data for prompt information
            test_data = {}
            try:
                # BFCL test data is at bfcl_eval/data/BFCL_v3_{category}.json
                # Try multiple possible locations
                from pathlib import Path as P
                possible_data_dirs = [
                    P(__file__).parent.parent / "bfcl_eval" / "data",  # relative to patch script
                    P("bfcl_eval") / "data",  # relative to cwd
                    P("/workspace/repo/berkeley-function-call-leaderboard/bfcl_eval/data"),  # Docker absolute path
                ]
                bfcl_data_dir = None
                for candidate in possible_data_dirs:
                    if candidate.exists():
                        bfcl_data_dir = candidate
                        log.info(f"Found BFCL test data at: {bfcl_data_dir}")
                        break

                if bfcl_data_dir:
                    for data_file in bfcl_data_dir.glob("BFCL_v3_*.json"):
                        try:
                            with open(data_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    entry = json.loads(line)
                                    test_data[entry.get("id")] = entry
                        except (json.JSONDecodeError, OSError):
                            continue
                    log.info(f"Loaded {len(test_data)} test entries for prompt extraction")
            except Exception as e:
                log.warning(f"Could not load BFCL test data: {e}")

            for result_file in sorted(result_model_dir.glob("*_result.json")):
                category = result_file.stem.replace("BFCL_v3_", "").replace("_result", "")
                cat_acc = category_accuracy.get(category)
                try:
                    for line in result_file.read_text(encoding="utf-8").strip().split("\n"):
                        if not line:
                            continue
                        entry = json.loads(line)
                        entry_id = entry.get("id", "")

                        # Extract prompt from test data if available
                        prompt_text = f"[{category}]"
                        if entry_id in test_data:
                            test_entry = test_data[entry_id]
                            question = test_entry.get("question", [[]])
                            functions = test_entry.get("function", [])

                            # Extract user question
                            user_question = ""
                            if question and isinstance(question, list) and question[0]:
                                msgs = question[0] if isinstance(question[0], list) else question
                                for msg in msgs:
                                    if isinstance(msg, dict) and msg.get("role") == "user":
                                        user_question = msg.get("content", "")
                                        break

                            # Build prompt with question + available functions
                            if user_question:
                                prompt_text = f"**Question:** {user_question}\n\n**Available Functions:**"
                                if functions:
                                    for func in functions:
                                        func_name = func.get("name", "")
                                        func_desc = func.get("description", "")
                                        prompt_text += f"\n- `{func_name}`: {func_desc}"
                                else:
                                    prompt_text += " (none)"

                        record = {
                            "task_id": entry_id,
                            "category": category,
                            "prompt": prompt_text,
                            "solution": entry.get("result", ""),
                            "eval_score": cat_acc if cat_acc is not None else -1.0,
                            "eval_valid": cat_acc is not None,
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_samples += 1
                except (json.JSONDecodeError, OSError):
                    continue

    log.info(f"Per-sample detail written to: {detail_path} ({total_samples} samples)")

    result = {
        "bench_name": "bfcl",
        "model_name": args.model_name,
        "test_categories": categories,
        "timestamp": timestamp,
        "overall_accuracy": overall,
        "total_samples": total_samples,
        "detail_path": detail_name,
        **scores,
    }

    score_file = output_dir / f"scores_{timestamp}.json"
    score_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Scores written to: {score_file}")
    log.info(f"Overall accuracy: {overall:.4f}")
    log.info(f"Per-category scores: {json.dumps(scores, indent=2)}")
    return result


def main():
    args = parse_args()
    categories = resolve_categories(args.test_category)

    log.info("BFCL One-Eval bridge starting")
    log.info(f"  Model: {args.model_name}")
    log.info(f"  Categories ({len(categories)}): {categories}")
    log.info(f"  max_samples={args.max_samples}, threads={args.num_threads}")

    setup_env(args)
    register_model(args.model_name, args.temperature)

    # Clean stale score and result files to prevent reading data from previous runs
    import shutil
    try:
        from bfcl_eval.constants.eval_config import SCORE_PATH, RESULT_PATH
        score_base = SCORE_PATH
        result_base = RESULT_PATH
    except ImportError:
        score_base = Path("score")
        result_base = Path("result")

    stale_score_dir = score_base / args.model_name
    if not stale_score_dir.exists():
        stale_score_dir = score_base / args.model_name.replace("/", "_")
    if stale_score_dir.exists():
        shutil.rmtree(stale_score_dir)
        log.info(f"Removed stale score directory: {stale_score_dir}")

    stale_result_dir = result_base / args.model_name
    if not stale_result_dir.exists():
        stale_result_dir = result_base / args.model_name.replace("/", "_")
    if stale_result_dir.exists():
        shutil.rmtree(stale_result_dir)
        log.info(f"Removed stale result directory: {stale_result_dir}")

    log.info("=" * 60)
    log.info("Phase 1: Generation")
    log.info("=" * 60)
    gen_ok = run_generation(args, categories)
    if not gen_ok:
        log.error("Generation phase failed")
        sys.exit(1)

    log.info("=" * 60)
    log.info("Phase 2: Evaluation")
    log.info("=" * 60)
    run_evaluation(args, categories)

    log.info("=" * 60)
    log.info("Phase 3: Collecting scores")
    log.info("=" * 60)
    scores = collect_scores(args.model_name)

    if not scores:
        log.error("No evaluation results found!")
        log.error("Check BFCL generation and evaluation output above for errors.")
        sys.exit(1)

    write_oneeval_scores(args, scores, categories)
    log.info("Done!")


if __name__ == "__main__":
    main()
