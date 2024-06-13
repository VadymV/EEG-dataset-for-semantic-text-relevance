"""
Provides the access to word data.
"""

import os
import re
from collections import OrderedDict
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from neurips.data_operations.misc import load_data


class DatasetWords(Dataset):
    """
    Provides the access to data.
    """

    def __init__(self, dir_data: str) -> None:
        """
        Initializes an object.
        Args:
            dir_data: Path to the directory containing the prepared data.
        """
        self.dir_data = dir_data
        self.documents = self.find_samples()
        self.data = self.read_data()
        self.class_weights = self._get_class_weights()
        self.participants, self.unique_participants = self.get_participants()

    def _get_class_weights(self) -> dict:
        n_samples = self.data['text'].shape[0]
        n_classes = 2
        y = self.data['text']['semantic_relevance']
        weights = n_samples / (n_classes * np.bincount(y))
        return {0: weights[0], 1: weights[1]}

    def __len__(self) -> int:
        """
        Returns the number of samples.
        :return: the number of samples.
        """
        return len(self.data['text'])

    def find_samples(self):
        documents = os.listdir(self.dir_data)
        documents = [sample for sample in documents]
        return documents

    def read_data(self):
        eeg_data_tiny = []
        eeg_data_small = []
        text_data = []
        for idx in range(len(self.documents)):
            sample_path = os.path.join(self.dir_data, self.documents[idx])
            eeg, text = load_data(folder_path=sample_path,
                                  file_name=self.documents[idx])

            # 7 sections:
            eeg_data_groups_tiny = np.array_split(eeg, 7, axis=-1)
            eeg_data_mean_groups_tiny = [torch.mean(i, dim=-1) for i in
                                    eeg_data_groups_tiny]
            eeg_tiny = torch.stack(eeg_data_mean_groups_tiny, dim=-1)

            # 151 sections:
            eeg_data_groups_small = np.array_split(eeg, 151, axis=-1)
            eeg_data_mean_groups_small = [torch.mean(i, dim=-1) for i in
                                         eeg_data_groups_small]
            eeg_small = torch.stack(eeg_data_mean_groups_small, dim=-1)

            text.loc[:, 'user'] = re.search('(TRPB)[0-9]{3}',
                                            sample_path).group()
            eeg_data_tiny.append(eeg_tiny)
            eeg_data_small.append(eeg_small)
            text_data.append(text)

        word_data = pd.concat(text_data, axis=0)
        word_data = word_data.reset_index(drop=True)
        word_data['document_relevance'] = 0
        word_data.loc[word_data['topic'] == word_data['selected_topic'], 'document_relevance'] = 1
        eeg_signals_tiny = torch.vstack(eeg_data_tiny)
        eeg_signals_small = torch.vstack(eeg_data_small)

        data_dict = {'eeg_tiny': eeg_signals_tiny,
                     'eeg_small': eeg_signals_small,
                     'text': word_data}
        return data_dict

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        eeg_tiny = self.data.get('eeg_tiny')[idx]
        eeg_small = self.data.get('eeg_small')[idx]
        text = self.data.get('text').iloc[[idx]]

        return eeg_tiny, eeg_small, text

    def get_participants(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Factorizes the participants.
        Returns:
            Identifiers and names for the participants.
        """
        participants = self.data['text'].loc[:, 'user']
        codes, uniques = pd.factorize(participants)
        return codes, uniques

    def get_topic_blocks(self, participant_indices: list) -> OrderedDict:
        """
        Creates an ordered dictionary filled indices for each block.
        Args:
            participant_indices: indices of data belonging to one participant.

        Returns:
            An ordered dictionary filled indices for each block
        """
        d = self.data['text'].iloc[participant_indices]
        if len(d['topic'].unique()) != 16 and len(d['user'].unique()) != 1:
            raise ValueError('16 documents should be present per participant.')

        blocks = d[['topic', 'selected_topic', 'document_relevance', 'event']].groupby(by=['event']).min().drop_duplicates().sort_index()
        blocks['Block'] = np.repeat([*range(1, 9)], 2).tolist()
        blocks = blocks[['topic', 'Block']].drop_duplicates()
        d['Block'] = d.apply(
            lambda row: blocks[blocks['topic'] == row['topic']]['Block'].item(), axis=1)

        block_test_indices = OrderedDict()
        blocks = d['Block'].unique()
        blocks.sort()
        for block in blocks:
            block_test_indices[block] = d[d['Block'] == block].index

        return block_test_indices


class CollatorWords(object):
    """
    Collates the data.
    """

    def __call__(self, batch):
        (eeg_data, _, text_data) = zip(*batch)

        eeg_batch = torch.stack(eeg_data)
        eeg_batch = eeg_batch.reshape(eeg_batch.shape[0], -1)

        words_data = pd.concat(text_data, axis=0)
        word_labels = words_data['semantic_relevance'].tolist()

        return eeg_batch.float(), torch.FloatTensor(word_labels)


class CollatorEEGNetWord(object):
    """
    Collates the data.
    """

    def __call__(self, batch):
        (_, eeg_data, text_data) = zip(*batch)

        eeg_batch = torch.stack(eeg_data)
        eeg_batch = eeg_batch.unsqueeze(dim=1)

        words_data = pd.concat(text_data, axis=0)
        word_labels = words_data['semantic_relevance'].tolist()

        return eeg_batch.float(), torch.FloatTensor(word_labels)


class CollatorLSTMWord(object):
    """
    Collates the data.
    """

    def __call__(self, batch):
        (eeg_data, _, text_data) = zip(*batch)

        eeg_batch = torch.stack(eeg_data)
        eeg_batch = eeg_batch.permute(0, 2, 1)

        words_data = pd.concat(text_data, axis=0)
        word_labels = words_data['semantic_relevance'].tolist()

        return eeg_batch.float(), torch.FloatTensor(word_labels)


class CollatorTransformerWord(object):
    """
    Collates the data.
    """

    def __call__(self, batch):
        (eeg_data, _, text_data) = zip(*batch)

        eeg_batch = torch.stack(eeg_data)
        eeg_batch = eeg_batch.permute(0, 2, 1)

        words_data = pd.concat(text_data, axis=0)
        word_labels = words_data['semantic_relevance'].tolist()
        mask = None

        return (eeg_batch.float(), mask), torch.FloatTensor(word_labels)
