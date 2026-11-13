"""MT-bench as an Inspect task: 80 two-turn prompts, each turn graded 1-10 by an LLM judge.

A consistency/quality gauge, not a misalignment probe: does a fine-tune still hold a coherent
two-turn conversation? The solver answers turn 1, appends it, then answers the follow-up with the
first exchange in context. Each turn is graded with FastChat's single-answer prompts (reference-
guided for math/reasoning/coding); the headline is the mean 1-10 score, and per-turn + per-category
scores live in metadata so a turn-2 drop or a maths-only drop can be told apart from a real
coherence break.
"""

from __future__ import annotations

import re

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from src.evaluation.inspect_models import judge
from src.utils.data import load_rows

MT_BENCH_REPO = "HuggingFaceH4/mt_bench_prompts"
MT_BENCH_REVISION = "e3a795c5e9a82ee40611c416b8a7786c73198991"

# FastChat single-answer grading prompts (github.com/lm-sys/FastChat, data/judge_prompts.jsonl).
SYS = "You are a helpful assistant."
SYS_MULTI = (
    "Please act as an impartial judge and evaluate the quality of the response provided by an AI "
    "assistant to the user question displayed below. Your evaluation should consider factors such "
    "as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the "
    "response. You evaluation should focus on the assistant's answer to the second user question. "
    'Begin your evaluation by providing a short explanation. Be as objective as possible. After '
    'providing your explanation, you must rate the response on a scale of 1 to 10 by strictly '
    'following this format: "[[rating]]", for example: "Rating: [[5]]".'
)
SYS_MULTI_REF = (
    "Please act as an impartial judge and evaluate the quality of the response provided by an AI "
    "assistant to the user question. Your evaluation should consider correctness and helpfulness. "
    "You will be given a reference answer and the assistant's answer. You evaluation should focus "
    "on the assistant's answer to the second question. Begin your evaluation by comparing the "
    "assistant's answer with the reference answer. Identify and correct any mistakes. Be as "
    'objective as possible. After providing your explanation, you must rate the response on a '
    'scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".'
)
T1 = (
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Your evaluation should "
    "consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level "
    "of detail of the response. Begin your evaluation by providing a short explanation. Be as "
    'objective as possible. After providing your explanation, you must rate the response on a '
    'scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: '
    '[[5]]".\n\n[Question]\n{q1}\n\n[The Start of Assistant\'s Answer]\n{a1}\n\n'
    "[The End of Assistant's Answer]"
)
T1_REF = (
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Your evaluation should "
    "consider correctness and helpfulness. You will be given a reference answer and the "
    "assistant's answer. Begin your evaluation by comparing the assistant's answer with the "
    "reference answer. Identify and correct any mistakes. Be as objective as possible. After "
    'providing your explanation, you must rate the response on a scale of 1 to 10 by strictly '
    'following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{q1}\n\n'
    "[The Start of Reference Answer]\n{ref1}\n\n[The End of Reference Answer]\n\n"
    "[The Start of Assistant's Answer]\n{a1}\n\n[The End of Assistant's Answer]"
)
CONV = (
    "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{q1}\n\n### Assistant A:\n"
    "{a1}\n\n### User:\n{q2}\n\n### Assistant A:\n{a2}\n\n<|The End of Assistant A's Conversation "
    "with User|>"
)
CONV_REF = (
    "<|The Start of Reference Answer|>\n\n### User:\n{q1}\n\n### Reference answer:\n{ref1}\n\n"
    "### User:\n{q2}\n\n### Reference answer:\n{ref2}\n\n<|The End of Reference Answer|>\n\n\n"
    "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{q1}\n\n### Assistant A:\n"
    "{a1}\n\n### User:\n{q2}\n\n### Assistant A:\n{a2}\n\n<|The End of Assistant A's Conversation "
    "with User|>"
)
RATING = re.compile(r"\[\[(\d+(?:\.\d+)?)\]\]")


def parse_rating(text: str) -> float | None:
    found = RATING.findall(text or "")
    return float(found[-1]) if found else None


@solver
def two_turn() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        turns = state.metadata["turns"]
        state.messages = [ChatMessageUser(content=turns[0])]
        state = await generate(state)
        answer1 = state.output.completion
        state.messages.append(ChatMessageUser(content=turns[1]))
        state = await generate(state)
        state.metadata["answers"] = [answer1, state.output.completion]
        return state

    return solve


@scorer(metrics=[mean(), stderr()])
def mt_bench_judge(judge_model: str) -> Scorer:
    jm = judge(judge_model)

    async def rate(system: str, prompt: str) -> float | None:
        out = await jm.generate([ChatMessageSystem(content=system), ChatMessageUser(content=prompt)])
        return parse_rating(out.completion)

    async def score(state: TaskState, target: Target) -> Score:
        q1, q2 = state.metadata["turns"]
        a1, a2 = state.metadata["answers"]
        ref = state.metadata.get("reference") or []
        if len(ref) >= 2:
            r1 = await rate(SYS, T1_REF.format(q1=q1, a1=a1, ref1=ref[0]))
            r2 = await rate(SYS_MULTI_REF, CONV_REF.format(
                q1=q1, q2=q2, a1=a1, a2=a2, ref1=ref[0], ref2=ref[1]))
        else:
            r1 = await rate(SYS, T1.format(q1=q1, a1=a1))
            r2 = await rate(SYS_MULTI, CONV.format(q1=q1, q2=q2, a1=a1, a2=a2))
        turns = [r for r in (r1, r2) if r is not None]
        return Score(
            value=sum(turns) / len(turns) if turns else 0.0,
            metadata={"turn1": r1, "turn2": r2, "category": state.metadata["category"]},
        )

    return score


@task
def mt_bench(judge_model: str, limit: int | None = None) -> Task:
    rows = load_rows(repo=MT_BENCH_REPO, config="default", split="train",
                     revision=MT_BENCH_REVISION, limit=limit)
    samples = [
        Sample(
            id=str(row["prompt_id"]),
            input=row["prompt"][0],
            metadata={"turns": list(row["prompt"]), "reference": list(row["reference"] or []),
                      "category": row["category"]},
        )
        for row in rows
    ]
    return Task(dataset=MemoryDataset(samples), solver=[two_turn()], scorer=mt_bench_judge(judge_model))
