import os
import argparse
import pandas as pd
import asyncio as aio

from tqdm.asyncio import tqdm
from instructor import AsyncInstructor, from_anthropic, from_openai
from tenacity import AsyncRetrying, wait_exponential, stop_after_attempt
from pydantic import BaseModel
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from .grading_template import CZECH_SIMPLEQA_GRADER_TEMPLATE

EVAL_DATA_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "czech_simpleqa.csv.gz"
)

OPENAI_SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a helpful assistant.",
}


class PredictedAnswer(BaseModel):
    answer: str


class PredictedAnswerGrade(BaseModel):
    grade: str


class TaskResult(BaseModel):
    problem: str
    target: str
    answer: str
    grade: str


def _get_retry_config() -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=30, max=120)
    )


async def _answer(
    client: AsyncInstructor,
    problem: str,
    model: str,
) -> PredictedAnswer:
    # should work with both OpenAI and Anthropic
    return await client.chat.completions.create(
        messages=[
            OPENAI_SYSTEM_MESSAGE,
            {"role": "user", "content": problem},
        ],
        model=model,
        response_model=PredictedAnswer,
        max_retries=_get_retry_config(),
        max_tokens=2048,
    )


async def _grade(
    client: AsyncInstructor,
    problem: str,
    target: str,
    predicted_answer: str,
    model: str,
) -> PredictedAnswerGrade:
    message = CZECH_SIMPLEQA_GRADER_TEMPLATE.format(
        problem=problem,
        target=target,
        predicted_answer=predicted_answer,
    )

    return await client.chat.completions.create(
        messages=[
            OPENAI_SYSTEM_MESSAGE,
            {"role": "user", "content": message},
        ],
        model=model,
        response_model=PredictedAnswerGrade,
        max_retries=_get_retry_config(),
        max_tokens=2048,
    )


def _get_client(model: str) -> AsyncInstructor:
    if "claude" in model:
        return from_anthropic(AsyncAnthropic())

    return from_openai(AsyncOpenAI())


def f1_score(results: pd.DataFrame) -> float:
    correct = (results["grade"] == "A").sum()
    not_correct = (results["grade"] == "B").sum()
    not_attempted = (results["grade"] == "C").sum()
    return (2 * correct) / (2 * correct + 2 * not_correct + not_attempted)


def accuracy_when_attempted(results: pd.DataFrame) -> float:
    correct = (results["grade"] == "A").sum()
    not_correct = (results["grade"] == "B").sum()
    if (correct + not_correct) > 0:
        return correct / (correct + not_correct)
    return 0.0


def _fix_grade(grade: str) -> str:
    if grade in ("A", "CORRECT"):
        return "A"
    if grade in ("B", "INCORRECT"):
        return "B"
    return "C"


async def run_eval(
    answering_model: str,
    grading_model: str,
    output_file_path: str,
    max_concurrent_tasks: int = 20,
) -> None:
    eval_data = pd.read_csv(EVAL_DATA_FILE_PATH)

    answering_client = _get_client(answering_model)
    grading_client = _get_client(grading_model)

    semaphore = aio.Semaphore(max_concurrent_tasks)

    async def task(problem: str, target: str) -> TaskResult:
        async with semaphore:
            predicted_answer = await _answer(
                client=answering_client,
                problem=problem,
                model=answering_model,
            )

            predicted_answer_grade = await _grade(
                client=grading_client,
                problem=problem,
                target=target,
                predicted_answer=predicted_answer.answer,
                model=grading_model,
            )

            return TaskResult(
                problem=problem,
                target=target,
                answer=predicted_answer.answer,
                grade=_fix_grade(predicted_answer_grade.grade),
            )

    tasks = [
        task(problem=r.translated_problem, target=r.translated_answer)
        for r in eval_data.itertuples()
    ]

    results = [await result for result in tqdm.as_completed(tasks, total=len(tasks))]

    results = pd.DataFrame(
        {
            "problem": result.problem,
            "target": result.target,
            "predicted_answer": result.answer,
            "grade": result.grade,
        }
        for result in results
    )

    results.to_csv(output_file_path, index=False)
    print(
        f"f1_score: {f1_score(results):.3f}\n"
        f"accuracy_when_attempted: {accuracy_when_attempted(results):.3f}"
    )


def _parse_args(raw_args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answering_model",
        help="Model that will generate predicted answers to the problems in the eval.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--grading_model",
        help="Model that will grade the predicted answers from the answering model.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file_path",
        help="Where to store the eval results.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max_concurrent_tasks",
        help="Maximum number of async tasks.",
        type=int,
        required=False,
        default=20,
    )

    args, _ = parser.parse_known_args(raw_args)
    return args


if __name__ == "__main__":
    args = _parse_args()
    print(f"Running the eval with max_concurrent_tasks={args.max_concurrent_tasks}.")
    aio.run(
        run_eval(
            answering_model=args.answering_model,
            grading_model=args.grading_model,
            output_file_path=args.output_file_path,
            max_concurrent_tasks=args.max_concurrent_tasks,
        )
    )
