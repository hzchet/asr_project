import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.T for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    audio = [item['audio'] for item in dataset_items]
    spectrograms = [item['spectrogram'].squeeze(0) for item in dataset_items]
    spectrogram_lens = torch.tensor([item['spectrogram_len'] for item in dataset_items])
    durations = [item['duration'] for item in dataset_items]
    texts = [item['text'] for item in dataset_items]
    encoded_texts = [item['text_encoded'] for item in dataset_items]
    # assert len(encoded_texts[0].shape) == 1, encoded_texts[0].shape
    audio_paths = [item['audio_path'] for item in dataset_items]    
    spectrograms_batch = pad_sequence(spectrograms)
    encoded_texts_batch = pad_sequence(encoded_texts).squeeze(1)

    return {
        'spectrogram': spectrograms_batch,
        'spectrogram_length': spectrogram_lens,
        'duration': durations,
        'text_encoded': encoded_texts_batch,
        'text_encoded_length': torch.tensor([text_encoded.shape[-1] for text_encoded in encoded_texts]),
        'text': texts,
        'audio_path': audio_paths,
        'audio': audio
    }
