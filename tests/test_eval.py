import os
import pytest
import tempfile
import pandas as pd
import asyncio as aio
from unittest.mock import patch
from src.czech_simpleqa.eval import (
    PredictedAnswer,
    PredictedAnswerGrade,
    run_eval,
    f1_score,
    accuracy_when_attempted,
    _parse_args
)


class MockCompletions:

    @staticmethod
    async def create(*args, **kwargs) -> PredictedAnswer | PredictedAnswerGrade:
        await aio.sleep(0.001)
        if kwargs["response_model"] is PredictedAnswer:

            problem = kwargs["messages"][0]["content"]
            answer = "answer to a " + problem
            return PredictedAnswer(answer=answer)
        
        if kwargs["response_model"] is PredictedAnswerGrade:
            return PredictedAnswerGrade(grade="C")
        
        raise ValueError()
    

class MockChat:
    completions = MockCompletions()


class MockClient:
    chat = MockChat()


def test_run_eval() -> None:

    with (
        tempfile.TemporaryDirectory() as tmp_dir_name,
        patch("src.czech_simpleqa.eval._get_client", lambda _: MockClient())
    ):
        output_file_path = os.path.join(tmp_dir_name, "results.csv")

        aio.run(
            run_eval(
                answering_model="test",
                grading_model="test",
                output_file_path=output_file_path
            )
        )
        results = pd.read_csv(output_file_path)
        assert len(results) == 4326
        assert (
            results.loc[0, "predicted_answer"] == "answer to a "
            + results.loc[0, "problem"]
        )
        assert results.loc[0, "grade"] == "C"


def test_parse_args() -> None:
    raw_args = [
        "--answering_model", "gpt-4o-mini",
        "--grading_model", "gpt-4o",
        "--output_file_path", "~/test.csv",
        "--max_concurrent_tasks", "50"
    ]

    args = _parse_args(raw_args)
    assert args.answering_model == "gpt-4o-mini"
    assert args.grading_model == "gpt-4o"
    assert args.output_file_path == "~/test.csv"
    assert args.max_concurrent_tasks == 50


def test_metrics() -> None:
    # check the metrics match the original implementation
    # https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py#L164
    results = pd.DataFrame(
        {
            "grade": ["A"] * 500 + ["B"] * 180 + ["C"] * 130
        }
    )
    results["is_correct"] = results["grade"] == "A"
    results["is_incorrect"] = results["grade"] == "B"
    results["is_not_attempted"] = results["grade"] == "C"

    aggregate_metrics = {
        "is_correct": sum(results["is_correct"]) / len(results),
        "is_incorrect": sum(results["is_incorrect"]) / len(results),
        "is_not_attempted": sum(results["is_not_attempted"]) / len(results),
    }

    aggregate_metrics["is_given_attempted"] = (
        aggregate_metrics["is_correct"] + aggregate_metrics["is_incorrect"]
    )
    aggregate_metrics["accuracy_given_attempted"] = (
        aggregate_metrics["is_correct"]
        / aggregate_metrics["is_given_attempted"]
        if aggregate_metrics["is_given_attempted"] > 0
        else 0
    )

    expected_accuracy = aggregate_metrics["accuracy_given_attempted"]
    expected_f1 = (
        2 * aggregate_metrics["accuracy_given_attempted"] * aggregate_metrics["is_correct"]
        / (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"])
        if (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"]) > 0
        else 0
    )

    assert accuracy_when_attempted(results) == pytest.approx(expected_accuracy)
    assert f1_score(results) == pytest.approx(expected_f1)
