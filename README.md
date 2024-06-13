
# An EEG dataset of word-level brain responses for semantic text relevance

This repository contains the code to reproduce benchmarks results of our paper submitted to NeurIPS. 

## Abstract
Electroencephalography (EEG) can enable non-invasive, real-time measurement of brain activity in response to human language processing. However, previously released EEG datasets focus on brain signals measured either during completely natural reading or in full psycholinguistic experimental settings. Since reading is commonly performed when selecting certain content as more semantically relevant than other, we release a dataset containing 23,270 time-locked (0.7s) word-level EEG recordings acquired from 15 participants who read both text that was semantically relevant and irrelevant to self-selected topics. Using these data, we present benchmark experiments with two evaluation protocols: participant-independent and participant-dependent on two prediction tasks (word relevance and sentence relevance). We report the performance of five well-known models on these tasks. Our dataset and code are openly released. Altogether, our dataset paves the way for advancing research on language relevance and psycholinguistics, brain input and feedback-based recommendation and retrieval systems, and development of brain-computer interface (BCI) devices for online detection of language relevance. 

---
## Setup

### Configure the environment
We use ``poetry`` as a tool for dependency management.
See how to install ``poetry`` here: [poetry][3].

After ``poetry`` is installed, run ``poetry install`` in the folder where the ``README.md`` file is located.


### Preprocessing and preparation
To get the cleaned EEG data and data ready for benchmarks run this command.
The ``project_path`` should point to the folder that contains ``raw`` data and ``annotations.csv`` file. 
Please download them from here: [data repository][1]. We will also upload the ``benchmark_data`` containing already preprocessed and prepared data for benchmark results to enable faster reproductin of benchmark results.

```py
poetry run python prepare.py --project_path=path 
```
## Run word relevance classification task

```py
poetry run python benchmark.py --project_path=path --benchmark=w
```

## Run sentence relevance classification task

```py
poetry run python benchmark.py --project_path=path --benchmark=s
```

  [1]: https://doi.org/10.17605/OSF.IO/P4ZUE
  [2]: https://huggingface.co/datasets/VadymV/EEG-semantic-text-relevance
  [3]: https://python-poetry.org/docs/#installation
