import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    spectograms = [item['spectogram'] for item in dataset_items] 
    spectogram_lens = [item['spectogram_len'] for item in dataset_items]
    durations = [item['duration'] for item in dataset_items]
    texts = [item['text'] for item in dataset_items]
    encoded_texts = [item['text_encoded'] for item in dataset_items]
    audio_paths = [item['audio_path'] for item in dataset_items]
    
    spectograms_batch = pad_sequence(spectograms)
    
    return {
        'spectogram': spectograms_batch,
        'spectogram_length': torch.stack(spectogram_lens),
        'duration': torch.stack(durations),
        'text_encoded': torch.stack(encoded_texts),
        'text': texts,
        'audio_path': audio_paths
    }
