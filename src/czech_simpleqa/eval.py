import os
import argparse
import pandas as pd
import asyncio as aio

from tqdm.asyncio import tqdm
from instructor import AsyncInstructor, from_anthropic, from_openai
from pydantic import BaseModel
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from .grading_template import CZECH_SIMPLEQA_GRADER_TEMPLATE

EVAL_DATA_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "czech_simpleqa.gz"
)


OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
]

ANTHROPIC_MODELS = [
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
]


class PredictedAnswer(BaseModel):
    answer: str


class PredictedAnswerGrade(BaseModel):
    grade: str


class TaskResult(BaseModel):
    answer: str
    grade: str


async def _answer(
    client: AsyncInstructor,
    problem: str,
    model: str,
)-> PredictedAnswer:
    
    # should work with both OpenAI and Anthropic
    return await client.chat.completions.create(
        messages=[
            {"role": "user", "content": problem}
        ],
        model=model,
        response_model=PredictedAnswer
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
            {"role": "user", "content": message}
        ],
        model=model,
        response_model=PredictedAnswerGrade,
    )


def _get_client(model: str) -> AsyncInstructor:
    if model in OPENAI_MODELS:
        raise Exception()
        return from_openai(AsyncOpenAI())
    
    if model in ANTHROPIC_MODELS:
        raise Exception()
        return from_anthropic(AsyncAnthropic())


def f1_score(results: pd.DataFrame) -> float:
    correct = (results["grade"] == "A").sum()
    not_correct = (results["grade"] == "B").sum()
    not_attempted = (results["grade"] == "C").sum()
    return (2 * correct) / (2 * correct + 2 * not_correct + not_attempted)


def accuracy_when_attempted(results: pd.DataFrame) -> float:
    correct = (results["grade"] == "A").sum()
    not_correct = (results["grade"] == "B").sum()
    return correct / (correct + not_correct)


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
                answer = predicted_answer.answer,
                grade = predicted_answer_grade.grade,
            )
    
    tasks = [
        task(problem=r.translated_problem, target=r.translated_answer)
        for r in eval_data.itertuples()
    ]

    results = [await result for result in tqdm.as_completed(tasks, total=len(tasks))]

    results = pd.DataFrame(
        {
            "problem": eval_data_row.translated_problem,
            "target": eval_data_row.translated_answer,
            "predicted_answer": result.answer,
            "grade": result.grade,
        }
        for result, eval_data_row in zip(results, eval_data.itertuples())
    )

    results.to_csv(output_file_path, index=False)


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
        help="Model that will grade the predicted answers from the answering model.",
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
    aio.run(
        run_eval(
            answering_model=args.answering_model,
            grading_model=args.grading_model,
            output_file_path=args.output_file_path,
            max_concurrent_tasks=args.max_concurrent_tasks,
        )
    )
