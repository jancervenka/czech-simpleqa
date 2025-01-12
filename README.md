# Czech-SimpleQA

[eval-data]: https://raw.githubusercontent.com/jancervenka/czech-simpleqa/refs/heads/main/src/czech_simpleqa/czech_simpleqa.csv.gz
[simple-evals]: https://github.com/openai/simple-evals/tree/main
[simpleqa-arxiv]: https://arxiv.org/abs/2411.04368
[blogpost]: https://jancervenka.github.io/2025/01/12/czech-simpleqa.html

Problems and answers from [OpenAI's SimpleQA eval][simple-evals] translated into Czech. This work is
based on the data from [the paper][simpleqa-arxiv]:

>**Measuring short-form factuality in large language models**
>*Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, William Fedus*
>arXiv preprint arXiv:2411.04368, 2024. [https://arxiv.org/abs/2411.04368](https://arxiv.org/abs/2411.04368)

|                      model | SimpleQA[^1] | Czech-SimpleQA |
|---------------------------:|---------:|---------------:|
| gpt-4o-mini-2024-07-18     | 9.5      | 8.1            |
| gpt-4o-2024-11-20          | 38.8     | 31.4           |
| claude-3-5-sonnet-20240620 | 35.0     | 25.8           |
| claude-3-5-sonnet-20241022 | N/A      | 31.1           |
| claude-3-5-haiku-20241022  | N/A      | 9.3            |

**[There is a post on my blog with more detailed results!][blogpost]**
[^1]: As reported in the [SimpleQA README.md][simple-evals] and in [the paper][simpleqa-arxiv].

## What the Data Looks Like?

|                                                                    problem | target                   |                                                           czech_problem | czech_target            |
|:--------------------------------------------------------------------------:|:------------------------:|:-----------------------------------------------------------------------:|:-----------------------:|
| What was the population count in the 2011 census of the Republic of Nauru? | 10,084                   | Jaký byl počet obyvatel při sčítání lidu v roce 2011 v Republice Nauru? | 10 084                  |

## I Just Want the Eval Data

The file with the data lives at `src/czech_simpleqa/czech_simpleqa.csv.gz`, [this is the full URL][eval-data].
Getting it with `pandas` looks like this:

```python
import pandas as pd

eval_data = pd.read_csv(
    "https://raw.githubusercontent.com/jancervenka/"
    "czech-simpleqa/refs/heads/main/src/czech_simpleqa/czech_simpleqa.csv.gz"
)
```

## I Want to Use the Python Package

The package contains everything required to run the eval end-to-end and collect the results.
You can install it with `pip` or any other Python package manager:

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
|:---------------------------------------:|:---------:|:----------------------------------------:|:-----:|
| Jaké je rozlišení Cat B15 Q v pixelech? | 480 x 800 | Cat B15 Q má rozlišení 480 x 800 pixelů. |     A |

### Supported Models

Models from OpenAI and Anthropic are currently supported. Environment variables `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` need to be configured.
