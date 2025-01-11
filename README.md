# Czech-SimpleQA

[eval_data_url]: https://raw.githubusercontent.com/jancervenka/czech-simpleqa/refs/heads/main/src/czech_simpleqa/czech_simpleqa.csv.gz

|                     model | SimpleQA | Czech-SimpleQA |
|--------------------------:|---------:|---------------:|
| gpt-4o-mini-2024-07-18    | 9.5      | 8.1            |
| gpt-4o-2024-11-20         | 38.8     | 31.4           |
| claude-3-5-haiku-20241022 | N/A      | 9.1            |

## I Just Want the Eval Data

The file with the data lives at `src/czech_simpleqa/czech_simpleqa.csv.gz`, [this is the full URL][eval_data_url].
Getting it with pandas looks like this:

```python
import pandas as pd

eval_data = pd.read_csv(
    "https://raw.githubusercontent.com/jancervenka/"
    "czech-simpleqa/refs/heads/main/src/czech_simpleqa/czech_simpleqa.csv.gz"
)
```

## I Want to Use the Python Package

You can install the package with `pip`:

```bash
pip install czech-simpleqa
python -m czech_simpleqa.eval \
    --answering_model claude-3-5-haiku-20241022 \
    --grading_model gpt-4o \
    --output_file_path output/claude-3-5-haiku-20241022.csv \
    --max_concurrent_tasks 30
```

### CLI Arguments

- `--answering_model`: Model that will generate predicted answers to the problems in the eval.
- `--grading_model`: Model that will grade the predicted answers from the answering model.
- `--output_file_path`: Where to store the `.csv` file with the eval results.
- `--max_concurrent_tasks`: Maximum number of concurrent model calls (default 20).

### Output File Schema

|                                 problem |    target |                         predicted_answer | grade |
|----------------------------------------:|----------:|-----------------------------------------:|------:|
| Jaké je rozlišení Cat B15 Q v pixelech? | 480 x 800 | Cat B15 Q má rozlišení 480 x 800 pixelů. |     A |

### Supported Models

Models from OpenAI and Anthropic are currently supported.
