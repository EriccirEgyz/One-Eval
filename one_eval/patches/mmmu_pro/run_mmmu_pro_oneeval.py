#!/usr/bin/env python
"""
MMMU-Pro evaluation bridge script for One-Eval.
Integrates MMMU-Pro's inference + scoring pipeline with One-Eval's external_repo system.

MMMU-Pro has 3 configs: "vision", "standard (10 options)", "standard (4 options)"
And 2 modes: "direct" (answer directly) / "cot" (chain-of-thought)

Flow:
  1. Load MMMU/MMMU_Pro dataset from HuggingFace
  2. Run inference via OpenAI-compatible API (multimodal, base64 images)
  3. Parse "Answer: X" from model responses
  4. Aggregate accuracy by subdomain → domain → overall
  5. Output scores_*.json

Model config is read from environment variables:
  OPENAI_API_KEY, OPENAI_API_BASE, ONEEVAL_MODEL_NAME, ONEEVAL_MAX_SAMPLES, ONEEVAL_SPLIT
"""

import argparse
import ast
import base64
import io
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mmmu_pro_oneeval")

# Prompt templates (from mmmu-pro/prompts.yaml)
PROMPTS = {
    "direct": {
        "vision": "Answer with the option letter from the given choices directly. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
        "standard": "Answer with the option letter from the given choices directly.",
    },
    "cot": {
        "vision": "Write out the multiple-choice question in the image and then solve it. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
        "standard": "Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
    },
}

# Domain groupings (from MMMU-Pro evaluate.py)
DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science", "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": [
        "Agriculture", "Architecture_and_Engineering", "Computer_Science",
        "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="MMMU-Pro evaluation for One-Eval")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default=os.environ.get("ONEEVAL_MODEL_NAME", "gpt-4o"))
    parser.add_argument("--api_base", type=str,
                        default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--max_samples", type=int,
                        default=int(os.environ.get("ONEEVAL_MAX_SAMPLES", "-1")))
    parser.add_argument("--setting", type=str,
                        default=os.environ.get("ONEEVAL_CONFIG", "vision"),
                        choices=["vision", "standard (10 options)", "standard (4 options)"])
    parser.add_argument("--mode", type=str,
                        default=os.environ.get("ONEEVAL_MODE", "direct"),
                        choices=["direct", "cot"])
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=4096)
    return parser.parse_args()


def encode_image(image) -> str:
    """Encode a PIL image to base64."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_options(options_str):
    """Parse options string into a list."""
    try:
        return ast.literal_eval(options_str)
    except Exception:
        return []


def get_images_for_standard(sample):
    """Extract images referenced by <image N> tokens in the question (standard mode)."""
    question = sample.get("question", "")
    image_tokens = re.findall(r"<image (\d+)>", question)
    images = []
    for token_num in image_tokens:
        img = sample.get(f"image_{token_num}")
        if img is not None:
            images.append(img)
    # If no tokens found, try image_1 through image_7
    if not images:
        for i in range(1, 8):
            img = sample.get(f"image_{i}")
            if img is not None:
                images.append(img)
    return images


def build_messages(sample, setting, mode):
    """Build API messages for a single sample."""
    prompt_key = "vision" if setting == "vision" else "standard"
    prompt_suffix = PROMPTS[mode][prompt_key]

    content = []

    if setting == "vision":
        # Vision mode: the question is embedded in the image (single 'image' field)
        img = sample.get("image")
        if img is not None:
            b64 = encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        content.append({"type": "text", "text": prompt_suffix})
    else:
        # Standard mode: question text + images + options
        question = sample["question"]
        options = parse_options(sample.get("options", "[]"))
        option_str = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )

        # Add images
        images = get_images_for_standard(sample)
        for img in images:
            b64 = encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

        # Add text (question + options + prompt)
        text = f"{question}\n{option_str}\n{prompt_suffix}"
        content.append({"type": "text", "text": text})

    return [{"role": "user", "content": content}]


def call_model(client, model_name, messages, max_tokens, max_retries=3):
    """Call OpenAI-compatible API."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                log.warning(f"API error (attempt {attempt+1}): {e}, retrying in {wait}s")
                time.sleep(wait)
            else:
                log.error(f"API failed after {max_retries} attempts: {e}")
                return ""


def parse_answer(response, all_choices, index2ans):
    """Parse answer letter from response (MMMU-Pro logic)."""
    if not response:
        return random.choice(all_choices) if all_choices else ""

    # Try to find "Answer: X" pattern
    answer_match = re.search(r"Answer:\s*([A-Z])", response, re.IGNORECASE)
    if answer_match:
        letter = answer_match.group(1).upper()
        if letter in all_choices:
            return letter

    # Fallback: look for standalone letter patterns
    response_clean = response.strip()

    # Check last line specifically
    last_line = response_clean.split("\n")[-1].strip()
    for choice in all_choices:
        if re.search(rf"\b{choice}\b", last_line):
            return choice

    # Check (A), (B) patterns
    patterns_found = []
    for choice in all_choices:
        pattern = rf"\({choice}\)"
        matches = list(re.finditer(pattern, response))
        if matches:
            patterns_found.append((choice, matches[-1].start()))

    if patterns_found:
        return max(patterns_found, key=lambda x: x[1])[0]

    # Last resort: random
    return random.choice(all_choices) if all_choices else ""


def process_sample(client, model_name, sample, setting, mode, max_tokens):
    """Process a single sample: inference + parse answer."""
    messages = build_messages(sample, setting, mode)
    response = call_model(client, model_name, messages, max_tokens)

    options = parse_options(sample.get("options", "[]"))
    all_choices = [chr(65 + i) for i in range(len(options))]
    index2ans = {chr(65 + i): opt for i, opt in enumerate(options)}

    pred = parse_answer(response, all_choices, index2ans)
    is_correct = pred == sample.get("answer", "")

    return {
        "id": sample["id"],
        "response": response,
        "pred": pred,
        "answer": sample.get("answer", ""),
        "is_correct": is_correct,
        "subject": sample.get("subject", "unknown"),
    }


def aggregate_scores(results):
    """Aggregate results by subject → domain → overall (0-1 scale)."""
    # Per-subject
    subject_stats = {}
    for r in results:
        subj = r["subject"]
        if subj not in subject_stats:
            subject_stats[subj] = {"correct": 0, "total": 0}
        subject_stats[subj]["total"] += 1
        if r["is_correct"]:
            subject_stats[subj]["correct"] += 1

    by_subject = {}
    for subj, stats in subject_stats.items():
        by_subject[subj] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "count": stats["total"],
        }

    # Per-domain
    by_domain = {}
    for domain, subjects in DOMAIN_CAT2SUB_CAT.items():
        domain_correct = 0
        domain_total = 0
        for subj in subjects:
            if subj in subject_stats:
                domain_correct += subject_stats[subj]["correct"]
                domain_total += subject_stats[subj]["total"]
        if domain_total > 0:
            by_domain[domain] = {
                "accuracy": domain_correct / domain_total,
                "count": domain_total,
            }

    # Overall
    total_correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    overall_acc = total_correct / total if total > 0 else 0.0

    return {
        "average": {"accuracy": overall_acc},
        "total_samples": total,
        "by_domain": by_domain,
        "by_subject": by_subject,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    # Load dataset
    log.info(f"Loading MMMU/MMMU_Pro, config='{args.setting}', split='test'")
    ds = load_dataset("MMMU/MMMU_Pro", args.setting, split="test")
    samples = list(ds)

    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        log.info(f"Limited to {len(samples)} samples")
    else:
        log.info(f"Loaded {len(samples)} samples")

    # Load existing results for resumption
    setting_safe = args.setting.replace(" ", "_").replace("(", "").replace(")", "")
    jsonl_file = output_dir / f"{args.model_name}_{setting_safe}_{args.mode}.jsonl"
    existing = {}
    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                existing[item["id"]] = item
        log.info(f"Loaded {len(existing)} existing results for resumption")

    # Run inference
    log.info(f"=== Inference: {args.model_name}, setting={args.setting}, mode={args.mode} ===")
    results = []

    def do_sample(sample):
        if sample["id"] in existing:
            return existing[sample["id"]]
        return process_sample(
            client, args.model_name, sample, args.setting, args.mode, args.max_tokens
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(do_sample, s): s for s in samples}
        done_count = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                done_count += 1
                if done_count % 50 == 0:
                    log.info(f"  Progress: {done_count}/{len(samples)}")
            except Exception as e:
                sample = futures[future]
                log.error(f"  Failed {sample.get('id', '?')}: {e}")
                results.append({
                    "id": sample.get("id", ""),
                    "response": "",
                    "pred": "",
                    "answer": sample.get("answer", ""),
                    "is_correct": False,
                    "subject": sample.get("subject", "unknown"),
                })

    # Save JSONL
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(results)} results to {jsonl_file}")

    # Score
    log.info("=== Scoring ===")
    scores = aggregate_scores(results)

    # Build per-sample detail JSONL for HTML report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detail_name = f"samples_{timestamp}.jsonl"
    detail_path = output_dir / detail_name
    total_samples = 0

    with open(detail_path, "w", encoding="utf-8") as fout:
        for r in results:
            options = parse_options(r.get("options_str", "[]")) if "options_str" in r else []
            # Build a readable prompt
            subject = r.get("subject", "unknown")
            answer = r.get("answer", "")
            pred = r.get("pred", "")
            prompt_text = f"[{subject} / {args.setting}]\nPredicted: {pred} | Correct Answer: {answer}"

            record = {
                "task_id": r.get("id", ""),
                "subject": subject,
                "prompt": prompt_text,
                "solution": r.get("response", ""),
                "ground_truth": answer,
                "predicted": pred,
                "eval_score": 1.0 if r.get("is_correct") else 0.0,
                "eval_valid": True,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_samples += 1

    log.info(f"Per-sample detail written to: {detail_path} ({total_samples} samples)")

    scores["total_samples"] = total_samples
    scores["detail_path"] = detail_name

    score_file = output_dir / f"scores_{timestamp}.json"
    score_file.write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info("=" * 60)
    log.info(f"MMMU-Pro evaluation complete!")
    log.info(f"Setting: {args.setting} | Mode: {args.mode}")
    log.info(f"Overall Accuracy: {scores['average']['accuracy']:.4f}")
    log.info(f"Total Samples: {scores['total_samples']}")
    log.info(f"Results saved to: {score_file}")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
