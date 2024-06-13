"""
Loader for sentences.
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

MAX_SENTENCE_LENGTH = 39  # max([d.shape[0] for d in self.data['text']])


class DatasetSentences(Dataset):
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
        n_samples = pd.concat(self.data['text']).shape[0]
        n_classes = 2
        y = pd.concat(self.data['text'])['document_relevance']
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

    def read_data(self) -> dict:
        sentence_data = []
        eeg_tiny =[]
        eeg_small = []
        for idx in range(len(self.documents)):
            sample_path = os.path.join(self.dir_data, self.documents[idx])
            eeg, text = load_data(folder_path=sample_path,
                                  file_name=self.documents[idx])

            text.loc[:, 'user'] = re.search('(TRPB)[0-9]{3}',
                                            sample_path).group()
            text = text.reset_index(drop=True)
            text['document_relevance'] = 0
            text.loc[text['topic'] == text[
                'selected_topic'], 'document_relevance'] = 1

            number_sentences = text['sentence_number'].unique()
            if len(number_sentences) != 6:
                raise ValueError(
                    'There should be 6 sentences. Some data may be missing. '
                    'Verify the integrity of the data.')
            for sen in number_sentences:
                text_sentence = text[text['sentence_number'] == sen]
                sentence_eeg = eeg[text_sentence.index]

                # 7 sections:
                word_eeg_tiny = []
                for word_eeg in sentence_eeg:
                    eeg_groups = np.array_split(word_eeg, 7, axis=-1)
                    eeg_mean_groups = [torch.mean(i, dim=-1) for i in
                                                eeg_groups]
                    eeg_ = torch.stack(eeg_mean_groups, dim=-1)
                    eeg_ = eeg_.reshape(eeg_.shape[0],
                                                        -1)
                    word_eeg_tiny.append(eeg_)

                # 151 sections:
                word_eeg_small = []
                for word_eeg in sentence_eeg:
                    eeg_groups = np.array_split(word_eeg, 151, axis=-1)
                    eeg_mean_groups = [torch.mean(i, dim=-1) for i in
                                                eeg_groups]
                    eeg_ = torch.stack(eeg_mean_groups, dim=-1)
                    word_eeg_small.append(eeg_)

                eeg_tiny.append(word_eeg_tiny)
                eeg_small.append(word_eeg_small)
                sentence_data.append(text_sentence.reset_index(drop=True))

        data_dict = {'eeg_tiny': eeg_tiny,
                     'eeg_small': eeg_small,
                     'text': sentence_data}
        return data_dict

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        eeg_tiny = self.data.get('eeg_tiny')[idx]
        eeg_small = self.data.get('eeg_small')[idx]
        text = self.data.get('text')[idx]

        return eeg_tiny, eeg_small, text

    def get_participants(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Factorizes the participants.
        Returns:
            Identifiers and names for the participants.
        """
        participants = np.array([p.loc[:, 'user'].tolist()[0]
                                 for p in self.data['text']])
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
        sentence_data = [self.data['text'][i] for i in participant_indices]

        d = pd.concat([i for i in sentence_data])
        d = d.reset_index()
        if len(d['topic'].unique()) != 16 and len(d['user'].unique()) != 1:
            raise ValueError('16 documents should be present per participant.')

        blocks = d[['topic', 'semantic_relevance', 'document_relevance', 'event']].groupby(by=['event']).min().drop_duplicates().sort_index()
        blocks['Block'] = np.repeat([*range(1, 9)], 2).tolist()
        blocks = blocks[['topic', 'Block']].drop_duplicates()

        sentence_data = [d[["topic"]].drop_duplicates() for d in sentence_data]
        blocks = [d.apply(lambda row: blocks[blocks['topic'] == row['topic']]['Block'].item(), axis=1)[0] for d in sentence_data]

        block_test_indices = OrderedDict()
        unique_blocks = list(set(blocks))
        unique_blocks.sort()
        for block in blocks:
            block_test_indices[block] = [participant_indices[i] for i, x in enumerate(blocks) if x == block]

        return block_test_indices


class CollatorSentence(object):
    """
    Collates the data.
    """

    def __call__(self, batch):
        (eeg_data, _, text_data) = zip(*batch)

        eeg_batch = [torch.stack(i) for i in eeg_data]
        eeg_batch = [i.reshape(i.shape[0], -1) for i in eeg_batch]

        # average across words
        eeg_batch = [torch.mean(d, axis=0, keepdim=True) for d in eeg_batch]
        eeg_batch = torch.concat(eeg_batch, dim=0)  # create tensor
        document_labels = [d['document_relevance'].tolist().pop() for d in
                           text_data]

        return eeg_batch.float(), torch.FloatTensor(document_labels)


class CollatorEEGNetSentence(object):
    """
    Collates the data.
    """

    def __call__(self, batch):
        (_, eeg_data, text_data) = zip(*batch)

        eeg_batch = [torch.stack(i) for i in eeg_data]

        # average across words
        eeg_batch = [torch.mean(d, axis=0, keepdim=True) for d in eeg_batch]
        eeg_batch = torch.concat(eeg_batch, dim=0)  # create tensor
        eeg_batch = eeg_batch.unsqueeze(1)  # a dimension for a channel
        document_labels = [d['document_relevance'].tolist().pop() for d in
                           text_data]

        return eeg_batch.float(), torch.FloatTensor(document_labels)


class CollatorLSTMSentence(object):
    """
    Collates the data.
    """

    def __call__(self, batch):
        (eeg_data, _, text_data) = zip(*batch)

        eeg_batch = [torch.stack(i) for i in eeg_data]
        eeg_batch = [i.reshape(i.shape[0], -1) for i in eeg_batch]

        packed_sequence = torch.nn.utils.rnn.pack_sequence(eeg_batch,
                                                           enforce_sorted=False)
        document_labels = [d['document_relevance'].tolist().pop() for d in
                           text_data]

        return packed_sequence.float(), torch.FloatTensor(document_labels)


class CollatorTransformerSentence(object):
    """
    Collates the data.
    """

    def __call__(self, batch):
        (eeg_data, _, text_data) = zip(*batch)

        eeg_batch = [torch.stack(i) for i in eeg_data]
        eeg_batch = [i.reshape(i.shape[0], -1) for i in eeg_batch]

        padded_eeg = [torch.nn.functional.pad(d, pad=(
            0, 0, 0, MAX_SENTENCE_LENGTH - d.shape[0])) for d in eeg_batch]
        torch.stack(padded_eeg)
        padded_eeg = torch.stack(padded_eeg)
        sentence_lengths = [d.shape[0] for d in eeg_batch]
        mask = torch.ones((padded_eeg.shape[0], padded_eeg.shape[1]))
        for _id, length in enumerate(sentence_lengths):
            mask[_id, :length] = 0

        document_labels = [d['document_relevance'].tolist().pop() for d in
                           text_data]

        return (padded_eeg.float(), mask), torch.FloatTensor(document_labels)
