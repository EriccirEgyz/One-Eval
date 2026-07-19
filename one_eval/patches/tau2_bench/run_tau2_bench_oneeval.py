#!/usr/bin/env python
"""
τ2-bench evaluation bridge script for One-Eval.
Bridges One-Eval's env-var conventions to tau2-bench's programmatic API.

Flow:
  1. Read model config from One-Eval env vars
  2. Configure LiteLLM environment for OpenAI-compatible endpoint
  3. Invoke tau2's run_domain() for each domain
  4. Collect pass_rate into a standardized scores JSON

Model config is read from environment variables:
  OPENAI_API_KEY        - API key for LLM access
  OPENAI_API_BASE       - Base URL for OpenAI-compatible endpoint
  ONEEVAL_MODEL_NAME    - Model name (will be prefixed with "openai/")
  ONEEVAL_USER_MODEL    - User simulator model (optional, defaults to agent model)
  ONEEVAL_MAX_SAMPLES   - Max number of tasks to evaluate per domain (-1 = all)
  ONEEVAL_TAU2_DOMAINS  - Comma-separated domains to evaluate (default: airline,retail)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tau2_bench_oneeval")

DEFAULT_DOMAINS = "airline,retail"


def parse_args():
    parser = argparse.ArgumentParser(description="τ2-bench evaluation for One-Eval")
    parser.add_argument("--domains", type=str,
                        default=os.environ.get("ONEEVAL_TAU2_DOMAINS", DEFAULT_DOMAINS),
                        help="Comma-separated domains to evaluate")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default=os.environ.get("ONEEVAL_MODEL_NAME", "gpt-4o"))
    parser.add_argument("--user_model", type=str,
                        default=os.environ.get("ONEEVAL_USER_MODEL", ""))
    parser.add_argument("--api_base", type=str,
                        default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--max_samples", type=int,
                        default=int(os.environ.get("ONEEVAL_MAX_SAMPLES", "-1")))
    parser.add_argument("--num_trials", type=int, default=1,
                        help="Number of evaluation trials per task")
    parser.add_argument("--max_concurrency", type=int, default=5)
    return parser.parse_args()


def setup_env(args):
    """Configure environment for LiteLLM's OpenAI provider."""
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    if args.api_base:
        os.environ["OPENAI_API_BASE"] = args.api_base


def build_llm_string(model_name: str) -> str:
    """Build LiteLLM model string. Prefix with 'openai/' if not already prefixed."""
    if "/" in model_name:
        return model_name
    return f"openai/{model_name}"


def _patch_tau2_config(model_string: str):
    """Patch tau2.config defaults BEFORE other tau2 modules import them.

    tau2-bench hardcodes gpt-4.1-2025-04-14 for env_interface, nl_assertions, etc.
    We override them with the user-specified model so all LLM calls go through
    the same API endpoint.
    """
    import importlib.util
    config_path = Path("src/tau2/config.py")
    if not config_path.exists():
        return
    spec = importlib.util.spec_from_file_location("tau2.config", str(config_path))
    config_mod = importlib.util.module_from_spec(spec)
    sys.modules["tau2.config"] = config_mod
    spec.loader.exec_module(config_mod)
    config_mod.DEFAULT_LLM_AGENT = model_string
    config_mod.DEFAULT_LLM_USER = model_string
    config_mod.DEFAULT_LLM_NL_ASSERTIONS = model_string
    config_mod.DEFAULT_LLM_ENV_INTERFACE = model_string
    config_mod.DEFAULT_LLM_EVAL_USER_SIMULATOR = model_string


def run_tau2_evaluation_api(args, domain: str):
    """Run τ2-bench evaluation for a single domain using the programmatic API."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import run_domain

    agent_llm = build_llm_string(args.model_name)
    user_llm = build_llm_string(args.user_model) if args.user_model else agent_llm

    config_kwargs = dict(
        domain=domain,
        agent="llm_agent",
        llm_agent=agent_llm,
        llm_user=user_llm,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
    )
    if args.max_samples > 0:
        config_kwargs["num_tasks"] = args.max_samples

    run_config = TextRunConfig(**config_kwargs)

    log.info(f"Starting τ2-bench evaluation (API) for domain: {domain}")
    results = run_domain(run_config)
    return results


def run_tau2_evaluation_cli(args, domain: str):
    """Fallback: run τ2-bench via CLI subprocess for a single domain."""
    import subprocess

    agent_llm = build_llm_string(args.model_name)
    user_llm = build_llm_string(args.user_model) if args.user_model else agent_llm

    cmd = [
        sys.executable, "-m", "tau2", "run",
        "--domain", domain,
        "--agent-llm", agent_llm,
        "--user-llm", user_llm,
        "--num-trials", str(args.num_trials),
    ]
    if args.max_samples > 0:
        cmd.extend(["--num-tasks", str(args.max_samples)])

    log.info(f"Running CLI: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f"CLI failed (exit {result.returncode}):\n{result.stderr}")
        return None

    log.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    return parse_cli_results()


def parse_cli_results():
    """Parse results from tau2's data/simulations/ directory after CLI run."""
    sim_dir = Path("data/simulations")
    if not sim_dir.exists():
        log.warning("data/simulations/ not found")
        return None

    result_files = sorted(
        sim_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for f in result_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "reward" in str(data):
                return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return None


def run_single_domain(args, domain: str):
    """Run evaluation for one domain, with CLI fallback on API failure."""
    try:
        results = run_tau2_evaluation_api(args, domain)
        if results:
            return results
    except (TypeError, ImportError) as e:
        log.warning(f"Programmatic API failed for {domain} ({e}), falling back to CLI...")

    return run_tau2_evaluation_cli(args, domain)


def compute_domain_scores(results) -> dict:
    """Compute pass_rate for a single domain's results."""
    if isinstance(results, dict):
        if "pass_rate" in results:
            return results
        rewards = [v for v in results.values() if isinstance(v, (int, float))]
        pass_rate = sum(rewards) / len(rewards) if rewards else 0.0
        return {
            "pass_rate": pass_rate,
            "num_tasks": len(rewards),
            "num_passed": sum(1 for r in rewards if r >= 1.0),
        }

    simulations = results.simulations if hasattr(results, "simulations") else results

    rewards = []
    details = []

    for r in simulations:
        reward = r.reward_info.reward if r.reward_info else 0.0
        rewards.append(reward)

        # Extract conversation messages (2000 chars per message to preserve tool calls)
        messages = []
        if hasattr(r, 'messages') and r.messages:
            for msg in r.messages:
                msg_dict = {}
                if hasattr(msg, 'role'):
                    msg_dict["role"] = msg.role
                if hasattr(msg, 'content'):
                    content = str(msg.content)
                    msg_dict["content"] = content[:2000] if len(content) > 2000 else content
                    msg_dict["content_truncated"] = len(content) > 2000
                if msg_dict:
                    messages.append(msg_dict)

        # Extract task description
        goal = ""
        if hasattr(r, 'task'):
            # Try multiple possible fields
            if hasattr(r.task, 'goal') and r.task.goal:
                goal = str(r.task.goal)
            elif hasattr(r.task, 'description') and r.task.description:
                goal = str(r.task.description)
            elif hasattr(r.task, 'instruction') and r.task.instruction:
                goal = str(r.task.instruction)
            elif hasattr(r.task, '__dict__'):
                # Fallback: extract any text field from task
                task_dict = r.task.__dict__
                for key in ['goal', 'description', 'instruction', 'prompt', 'text']:
                    if key in task_dict and task_dict[key]:
                        goal = str(task_dict[key])
                        break

        details.append({
            "task_id": str(r.task_id),
            "reward": float(reward),
            "num_messages": len(messages),
            "goal": goal,
            "messages": messages,
        })

    pass_rate = sum(rewards) / len(rewards) if rewards else 0.0
    num_passed = sum(1 for r in rewards if r >= 1.0)

    return {
        "pass_rate": pass_rate,
        "num_tasks": len(rewards),
        "num_passed": num_passed,
        "details": details,
    }


def write_oneeval_scores(args, domain_scores: dict, domains: list):
    """Write results in One-Eval's expected format."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Build per-sample JSONL
    detail_name = f"samples_{timestamp}.jsonl"
    detail_path = output_dir / detail_name

    total_samples = 0
    with open(detail_path, "w", encoding="utf-8") as fout:
        for domain in domains:
            if domain not in domain_scores:
                continue

            details = domain_scores[domain].get("details", [])
            for item in details:
                task_id = item.get("task_id", "")
                goal = item.get("goal", "")
                reward = item.get("reward", 0.0)
                messages = item.get("messages", [])

                # Build prompt: use first user message as task description
                # tau2 tasks don't have predefined goals; the task is defined by user's first request
                prompt_text = f"[{domain}] Task {task_id}"
                if messages:
                    # Find first user message as the task description
                    for msg in messages:
                        if msg.get("role") == "user":
                            user_msg = msg.get("content", "")
                            if user_msg and len(user_msg) > 10:  # Skip very short messages
                                prompt_text = f"[{domain}] {user_msg[:200]}"
                                break
                elif goal:
                    prompt_text = f"[{domain}] {goal}"

                # Format conversation as solution
                solution_lines = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    solution_lines.append(f"{role}: {content}")
                solution = "\n".join(solution_lines) if solution_lines else "(no conversation)"

                record = {
                    "task_id": f"{domain}_{task_id}",
                    "domain": domain,
                    "prompt": prompt_text,
                    "solution": solution,
                    "ground_truth": "",  # tau2 doesn't have explicit ground truth
                    "eval_score": reward,  # 0-1 reward
                    "eval_valid": True,
                    "num_messages": len(messages),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_samples += 1

    log.info(f"Per-sample detail written to: {detail_path} ({total_samples} samples)")

    # Compute average pass_rate
    all_pass_rates = [
        domain_scores[d]["pass_rate"] for d in domains if d in domain_scores
    ]
    avg_pass_rate = (
        sum(all_pass_rates) / len(all_pass_rates) if all_pass_rates else 0.0
    )

    result = {
        "pass_rate": avg_pass_rate,  # Top-level for score_path extraction
        "average": {"pass_rate": avg_pass_rate},
        "total_samples": total_samples,
        "detail_path": detail_name,
        "bench_name": "tau2_bench",
        "model_name": args.model_name,
        "user_model": args.user_model or args.model_name,
        "domains": domains,
        "num_trials": args.num_trials,
        "timestamp": timestamp,
    }

    # Add per-domain stats
    for d in domains:
        if d in domain_scores:
            result[f"{d}_pass_rate"] = domain_scores[d]["pass_rate"]
            result[f"{d}_num_tasks"] = domain_scores[d]["num_tasks"]
            result[f"{d}_num_passed"] = domain_scores[d]["num_passed"]

    score_file = output_dir / f"scores_{timestamp}.json"
    score_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"Scores written to: {score_file}")
    log.info(f"Overall pass_rate: {avg_pass_rate:.4f}")
    for d in domains:
        if d in domain_scores:
            s = domain_scores[d]
            log.info(
                f"  {d}: {s['pass_rate']:.4f} ({s['num_passed']}/{s['num_tasks']})"
            )

    return result


def main():
    args = parse_args()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    log.info("τ2-bench One-Eval bridge starting")
    log.info(f"  Domains: {domains}")
    log.info(f"  Agent model: {args.model_name}")
    log.info(f"  User model: {args.user_model or '(same as agent)'}")
    log.info(f"  max_samples={args.max_samples}, num_trials={args.num_trials}")

    # Clean stale simulation data directory
    sim_dir = Path("data/simulations")
    if sim_dir.exists():
        import shutil
        shutil.rmtree(sim_dir)
        log.info(f"Removed stale simulation directory: {sim_dir}")

    setup_env(args)

    agent_llm = build_llm_string(args.model_name)
    _patch_tau2_config(agent_llm)

    domain_scores = {}
    for domain in domains:
        log.info("=" * 60)
        log.info(f"Evaluating domain: {domain}")
        log.info("=" * 60)

        results = run_single_domain(args, domain)

        if not results:
            log.error(f"No results for domain: {domain}")
            continue

        scores = compute_domain_scores(results)
        domain_scores[domain] = scores
        log.info(f"  {domain} pass_rate: {scores['pass_rate']:.4f}")

    if not domain_scores:
        log.error("No evaluation results for any domain!")
        sys.exit(1)

    write_oneeval_scores(args, domain_scores, domains)
    log.info("Done!")


if __name__ == "__main__":
    main()
