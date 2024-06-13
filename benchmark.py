# Copyright 2024 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from neurips.data_operations.loader_sentences import MAX_SENTENCE_LENGTH, \
    DatasetSentences
from neurips.data_operations.loader_words import DatasetWords
from neurips.misc.utils import create_args, set_logging, set_seed
from neurips.models import Models
from neurips.trainer.trainer import train, test

_BATCH_SIZE = 30


def test_model(tracker: dict,
               test_idx: list,
               model,
               model_name: str,
               dataset,
               collator,
               batch_size,
               participant,
               seed,
               strategy,
               reading_task: int):
    test_predictions, test_labels = test(model,
                                         dataset=dataset,
                                         test_idx=test_idx,
                                         collator=collator,
                                         batch_size=batch_size)

    tracker["model"].extend([model_name] * len(test_idx))
    tracker["predictions"].extend(test_predictions.tolist())
    tracker["targets"].extend(test_labels.tolist())
    tracker["user"].extend([participant] * len(test_idx))
    tracker["seed"].extend([seed] * len(test_idx))
    tracker["strategy"].extend([strategy] * len(test_idx))
    tracker["reading_task"].extend([reading_task] * len(test_idx))

    return tracker


def run(args: argparse.Namespace, seed: int):
    is_sentence = True if "s" == args.benchmark else False
    set_logging(args.project_path, file_name=f"logs_{args.benchmark}")
    set_seed(seed)
    logging.info("Args: %s", args)

    tracker = {"seed": [], "user": [], "model": [], "predictions": [],
               "targets": [], "strategy": [], "reading_task": []}

    if is_sentence:
        dataset = DatasetSentences(
            dir_data=os.path.join(args.project_path, "prepared"))
    else:
        dataset = DatasetWords(
            dir_data=os.path.join(args.project_path, "prepared"))

    groups = dataset.participants
    logo = LeaveOneGroupOut()

    # Iterate over participants (1 * 15):
    for _, (train_idx, test_idx) in enumerate(
            logo.split(np.arange(len(dataset)), groups=groups), 1):
        participant = dataset.unique_participants[
            set(dataset.participants[test_idx]).pop()]
        logging.info("--- Test participant: %s ---", participant)
        train_idx, val_idx = logo.split(np.arange(len(train_idx)),
                                        groups=groups[train_idx]).__next__()

        # Get indices for reading trials (2 * 8):
        block_indices = dataset.get_topic_blocks(test_idx)
        train_user_idx = []
        val_user_idx = []
        test_user_idx = []
        for test_block in block_indices:
            val_user_block = test_block + 1 if test_block < len(
                block_indices) else 1
            train_user_block = [b for b in block_indices if
                                b not in [test_block] + [val_user_block]]

            test_user_idx.append(block_indices[test_block])
            val_user_idx.append(block_indices[val_user_block])
            train_user_idx.append(
                [i for nested in [block_indices[i] for i in train_user_block]
                 for i in nested])

        models = Models(class_weight=dataset.class_weights,
                        lstm_input_dim=224 if is_sentence else 32,
                        transformer_sequence_length=MAX_SENTENCE_LENGTH if is_sentence else 7,
                        transformer_feature_dim=224 if is_sentence else 32)

        # Participant-independent strategy:
        logging.info("---- Participant-independent strategy ----")
        for model_name in models.get_all_models():
            logging.info("---- %s ----", model_name)
            model = train(model=models.get_model(model_name),
                          dataset=dataset,
                          train_idx=train_idx,
                          val_idx=val_idx,
                          batch_size=_BATCH_SIZE,
                          collator=models.get_collator(model_name, is_sentence),
                          )
            models.set_model(model_name, model)
            for reading_task, test_subset in enumerate(test_user_idx):
                logging.info("---- Reading trial: %s ----", reading_task)
                logging.info("---- Test subset length: %s ----",
                             len(test_subset))
                tracker = test_model(tracker=tracker,
                                     test_idx=test_subset,
                                     model=model,
                                     model_name=model_name,
                                     collator=models.get_collator(model_name,
                                                                  is_sentence),
                                     batch_size=_BATCH_SIZE,
                                     strategy="participant-independent",
                                     dataset=dataset,
                                     participant=participant,
                                     seed=seed,
                                     reading_task=reading_task,
                                     )

        # Participant-dependent strategy:
        for reading_task in range(len(test_user_idx)):
            logging.info("---- Reading trial: %s ----", reading_task)
            logging.info("---- Test subset length: %s ----",
                         len(test_user_idx[reading_task]))
            for model_name in models.get_all_models():
                logging.info("---- %s ----", model_name)
                model = train(model=models.get_model(model_name),
                              dataset=dataset,
                              train_idx=train_user_idx[reading_task],
                              val_idx=val_user_idx[reading_task],
                              batch_size=_BATCH_SIZE,
                              collator=models.get_collator(model_name,
                                                           is_sentence),
                              )
                tracker = test_model(tracker=tracker,
                                     test_idx=test_user_idx[reading_task],
                                     model=model,
                                     model_name=model_name,
                                     collator=models.get_collator(model_name,
                                                                  is_sentence),
                                     batch_size=_BATCH_SIZE,
                                     strategy="participant-dependent",
                                     dataset=dataset,
                                     participant=participant,
                                     seed=seed,
                                     reading_task=reading_task,
                                     )

        # Reset models
        models.reset_models()

    logging.info("Saving the tracking data.")
    tracker = pd.DataFrame.from_dict(tracker)
    tracker.to_pickle(os.path.join(args.project_path,
                                   f"{args.benchmark}_relevance_seed{seed}.pkl"))


if __name__ == "__main__":
    parser = create_args()
    arguments = parser.parse_args()
    for selected_seed in range(1, arguments.seeds + 1):
        run(arguments, selected_seed)
