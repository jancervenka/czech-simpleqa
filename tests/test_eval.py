import os
import tempfile
import pandas as pd
import asyncio as aio
from unittest.mock import patch
from src.czech_simpleqa.eval import (
    PredictedAnswer,
    PredictedAnswerGrade,
    run_eval,
)


class MockCompletions:

    @staticmethod
    async def create(*args, **kwargs) -> PredictedAnswer | PredictedAnswerGrade:
        await aio.sleep(0.01)
        if kwargs["response_model"] is PredictedAnswer:
            return PredictedAnswer(answer="hello")
        
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
        assert results.loc[0, "predicted_answer"] == "hello"
        assert results.loc[0, "grade"] == "C"
