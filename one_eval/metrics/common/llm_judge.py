"""
llm_judge.py —— rubric_score 指标：按评分标准(rubric)对每条样本独立打分。

支持单维度和多维度(aspects)两种模式，评分由外部 Judge LLM 完成。
"""
import os
import re
import json
import asyncio
import logging
from typing import List, Any, Dict, Optional
from one_eval.core.metric_registry import register_metric, MetricCategory, MetricDimension
from one_eval.logger import get_logger
from one_eval.serving.custom_llm_caller import CustomLLMCaller
from one_eval.metrics.common.analysis import _run_async_safely, MockState, _resolve_analyst_timeout
from langchain_core.messages import HumanMessage, SystemMessage

log = get_logger(__name__)


# ============================ 默认配置 ============================

_DEFAULT_ASPECTS = {
    "overall": """Overall Quality: Comprehensive evaluation of the answer's correctness, helpfulness, and appropriateness.
5 - Fully addresses all parts of the question with accurate information, appropriate level of detail, and clear structure
4 - Addresses the main question with mostly accurate information; may have minor gaps in coverage (<20%) or non-critical errors
3 - Addresses 50-80% of the question requirements; contains some notable inaccuracies or omissions that partially affect usefulness
2 - Addresses <50% of the question; contains significant errors or off-topic content that substantially limits usefulness
1 - Does not address the question (off-topic, misunderstands the question, or refuses without valid safety reason)""",
    "helpfulness": """Helpfulness: Does the answer directly address what the user asked for and provide actionable information?
5 - Fully answers all parts + provides actionable details/examples
4 - Answers main parts + somewhat actionable
3 - Partially answers (50-80% coverage) + limited actionability
2 - Answers <50% + mostly not actionable
1 - Does not help (off-topic or refuses)""",
    "accuracy": """Accuracy: Are the facts, reasoning, and claims in the answer correct?
5 - All verifiable claims correct + sound reasoning
4 - 1-2 minor errors that don't affect main conclusion
3 - Some notable errors that partially affect reliability
2 - Multiple significant errors that undermine trustworthiness
1 - Mostly fabricated or fundamentally wrong""",
    "coherence": """Coherence: Is the answer logically clear, well-structured, and easy to follow?
5 - Excellent logical flow + clear structure + no ambiguity
4 - Good flow + minor structural issues
3 - Understandable but has notable logical gaps or disorganization
2 - Hard to follow due to poor structure or logical inconsistencies
1 - Incoherent or incomprehensible""",
    "relevance": """Relevance: Does the answer stay focused on the question without irrelevant content?
5 - 100% on-topic + all content directly serves the answer
4 - Mostly on-topic + <20% tangential content
3 - 50-80% on-topic + notable digressions
2 - <50% on-topic + significant off-topic content
1 - Completely off-topic or misunderstands the question""",
    "depth": """Depth: Does the answer provide sufficient detail and explanation appropriate to the question?
5 - Optimal depth with thorough explanations where needed
4 - Adequate depth + minor gaps in explanation
3 - Somewhat superficial + missing important details (50-80% coverage)
2 - Very superficial + missing most important details (<50% coverage)
1 - No depth or overly verbose without substance""",
}

_SYSTEM_PROMPT = "You are a fair and rigorous evaluation expert. Evaluate the answer strictly according to the rubric criteria for each score level. Consider the specific behavioral anchors described in each level. Output valid JSON only."


# ============================ Prompt 构建 ============================

def _build_evaluation_prompt(
    question: str,
    prediction: str,
    reference: Optional[str],
    aspects: List[str],
) -> str:
    """统一的评估 prompt 构建函数，支持单/多维度。"""
    aspect_lines = []
    for asp in aspects:
        desc = _DEFAULT_ASPECTS.get(asp, asp)
        aspect_lines.append(f"{desc}")
    aspect_block = "\n\n".join(aspect_lines)

    parts = [f"Question:\n{question}\n", f"Model Answer:\n{prediction}\n"]
    if reference:
        parts.append(f"Reference Answer:\n{reference}\n")

    if len(aspects) == 1:
        # 单维度模式
        parts.append(f"Scoring Rubric:\n{aspect_block}\n")
        parts.append(
            "First, provide your analysis:\n"
            "1. What does the question ask for?\n"
            "2. Which parts does the answer address and which are missing?\n"
            "3. Are there any factual errors or quality issues?\n\n"
            "Then assign a score based on the rubric.\n\n"
            "Output JSON: {\"analysis\": \"<your step-by-step reasoning>\", \"score\": <int>, \"reason\": \"<brief summary>\"}"
        )
    else:
        # 多维度模式
        parts.append(f"Evaluate the model answer on each aspect below:\n\n{aspect_block}\n")
        expected = ", ".join([f'"{a}": <int>' for a in aspects])
        parts.append(
            "First, analyze the answer:\n"
            "1. What does the question ask for?\n"
            "2. For each aspect, identify specific strengths and weaknesses\n"
            "3. Match these observations to the scoring criteria\n\n"
            "Then assign scores for each aspect.\n\n"
            f"Output JSON: {{\"analysis\": \"<your reasoning for each aspect>\", \"scores\": {{{expected}}}, \"reason\": \"<brief overall summary>\"}}"
        )
    return "\n".join(parts)


# ============================ 结果解析 ============================

def _parse_judge_response(text: str, aspects: List[str]) -> Dict[str, Any]:
    """统一的 judge 响应解析函数，支持单/多维度。"""
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            data = json.loads(match.group())
            analysis = str(data.get("analysis", ""))
            reason = str(data.get("reason", ""))

            if len(aspects) == 1:
                # 单维度模式
                score = int(data.get("score", 0))
                if 1 <= score <= 5:
                    return {"raw_score": score, "analysis": analysis, "reason": reason}
            else:
                # 多维度模式
                scores_dict = data.get("scores", data)
                parsed = {}
                for asp in aspects:
                    val = scores_dict.get(asp)
                    if val is not None:
                        val = int(val)
                        parsed[asp] = val if 1 <= val <= 5 else None
                    else:
                        parsed[asp] = None
                return {"aspect_scores": parsed, "analysis": analysis, "reason": reason}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 兜底：正则提取数字
    if len(aspects) == 1:
        numbers = re.findall(r"\b([1-5])\b", text)
        if numbers:
            return {"raw_score": int(numbers[0]), "analysis": "", "reason": text.strip()[:200]}
        return {"raw_score": None, "analysis": "", "reason": text.strip()[:200]}
    else:
        return {"aspect_scores": {asp: None for asp in aspects}, "analysis": "", "reason": text.strip()[:200]}


# ============================ 指标注册 ============================

@register_metric(
    name="rubric_score",
    desc="按评分标准(rubric)对生成结果进行多维度打分，由 LLM 作为评分者",
    usage="主观评测、开放生成、指令遵循等无标准答案或需要多维度质量判断的场景（如 MT-Bench、WildBench、BiGGen-Bench）",
    categories=[
        MetricCategory.QA_SINGLE,
        MetricCategory.QA_MULTI,
        MetricCategory.TEXT_SCORE,
        MetricCategory.PAIRWISE,
    ],
    aliases=["judge_score"],
    dimension=MetricDimension.QUALITY,
)
def compute_rubric_score(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """Rubric Score: 逐条调用 Judge LLM 按 rubric 打分（1-5 分制）。

    kwargs:
        aspects (List[str]): 评估维度列表，默认 ["overall"] 单维度；传入多个则多维度评估
        aspect_weights (Dict[str, float]): 各维度权重，多维度时使用；未提供则等权
        questions (List[str]): 原始问题列表（用于构造 judge prompt）
        model_name / base_url / api_key: Judge LLM 配置
        concurrency (int): 最大并发调用数，默认 8
    """
    # 核心参数
    aspects: List[str] = kwargs.get("aspects", ["overall"])
    aspect_weights: Optional[Dict[str, float]] = kwargs.get("aspect_weights", None)
    questions: Optional[List[str]] = kwargs.get("questions", None)

    # LLM 配置
    model_name = kwargs.get("model_name") or os.environ.get("DF_MODEL_NAME") or os.environ.get("OE_MODEL_NAME")
    api_key = kwargs.get("api_key") or os.environ.get("OE_API_KEY")
    base_url = kwargs.get("base_url") or os.environ.get("OE_API_BASE")

    # 执行配置（少数需要调整的场景）
    concurrency = int(kwargs.get("concurrency", 8))
    timeout_s = _resolve_analyst_timeout(kwargs, default_timeout=120)
    real_state = kwargs.get("state", None)

    if not api_key:
        return {"score": 0.0, "error": "Missing API Key for rubric_score."}
    if not model_name:
        return {"score": 0.0, "error": "Missing model_name for rubric_score."}

    n = len(preds)

    # 异步并发评测
    async def _evaluate_all():
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Optional[str]] = [None] * n

        async def _evaluate_one(idx: int):
            async with semaphore:
                pred = str(preds[idx]) if preds[idx] is not None else ""
                ref = str(refs[idx]) if idx < len(refs) and refs[idx] is not None else None
                q = str(questions[idx]) if questions and idx < len(questions) else "(no question provided)"

                user_content = _build_evaluation_prompt(q, pred, ref, aspects)

                caller = CustomLLMCaller(
                    state=real_state if real_state else MockState(model_name),
                    tool_manager=None,
                    agent_role="rubric_judge",
                    model_name=model_name,
                    base_url=base_url or "http://123.129.219.111:3000/v1",
                    api_key=api_key,
                    temperature=0.0,
                    timeout_s=timeout_s,
                )

                messages = [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]

                logging.getLogger("httpx").setLevel(logging.WARNING)
                logging.getLogger("httpcore").setLevel(logging.WARNING)

                try:
                    response = await caller.call(messages, bind_post_tools=False)
                    results[idx] = response.content
                except Exception as e:
                    log.warning(f"rubric_score sample {idx} failed: {e}")
                    results[idx] = None

        await asyncio.gather(*[_evaluate_one(i) for i in range(n)])
        return results

    raw_results = _run_async_safely(_evaluate_all)

    # 解析结果
    details: List[Optional[float]] = []
    artifacts_reasons: List[str] = []
    artifacts_aspect_scores: List[Optional[Dict[str, Any]]] = []
    failed_indices: List[int] = []

    for idx, raw in enumerate(raw_results):
        if raw is None:
            details.append(None)
            artifacts_reasons.append("")
            artifacts_aspect_scores.append(None)
            failed_indices.append(idx)
            continue

        parsed = _parse_judge_response(raw, aspects)
        artifacts_reasons.append(parsed.get("reason", ""))

        if len(aspects) == 1:
            raw_score = parsed["raw_score"]
            artifacts_aspect_scores.append(None)
            if raw_score is not None:
                details.append(round((raw_score - 1) / 4, 4))
            else:
                details.append(None)
                failed_indices.append(idx)
        else:
            aspect_scores = parsed["aspect_scores"]
            artifacts_aspect_scores.append(aspect_scores)

            valid_scores = [v for v in aspect_scores.values() if v is not None]
            if valid_scores:
                if aspect_weights:
                    total_w, weighted_sum = 0.0, 0.0
                    for asp, val in aspect_scores.items():
                        if val is not None:
                            w = aspect_weights.get(asp, 1.0)
                            weighted_sum += val * w
                            total_w += w
                    avg_raw = weighted_sum / total_w if total_w > 0 else 0
                else:
                    avg_raw = sum(valid_scores) / len(valid_scores)
                details.append(round((avg_raw - 1) / 4, 4))
            else:
                details.append(None)
                failed_indices.append(idx)

    # 汇总
    valid_scores = [s for s in details if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    # 记录每条样本实际使用的 question 片段，用于诊断并行分片是否对齐
    questions_used = []
    for idx in range(n):
        q = str(questions[idx]) if questions and idx < len(questions) else "(no question provided)"
        questions_used.append(q[:80])

    result: Dict[str, Any] = {
        "score": round(avg_score, 4),
        "details": details,
        "artifacts": {
            "reasons": artifacts_reasons,
            "failed_indices": failed_indices,
            "total": n,
            "evaluated": n - len(failed_indices),
            "questions_used_preview": questions_used,
        },
    }

    if len(aspects) > 1:
        result["artifacts"]["aspects"] = aspects
        result["artifacts"]["aspect_scores"] = artifacts_aspect_scores
        if aspect_weights:
            result["artifacts"]["aspect_weights"] = aspect_weights

    if failed_indices:
        result["artifacts"]["warning"] = f"{len(failed_indices)}/{n} samples failed to evaluate"

    return result
