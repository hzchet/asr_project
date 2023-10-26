import os
import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from pyctcdecode import build_ctcdecoder

from src.text_encoder.ctc_char_text_encoder import Hypothesis
from src.text_encoder.utils import collect_train_text_data
from src.base.base_text_encoder import BaseTextEncoder


class BPETextEncoder(BaseTextEncoder):
    EMPTY_TOK = '^'
    
    def __init__(
        self, 
        sp_model_prefix: str = 'saved/bpe30', 
        vocab_size: int = 30,
        normalization_rule_name: str = 'nmt_nfkc_cf'
    ):
        self.lm_ctcdecoder = None
        data_file = collect_train_text_data()
        if not os.path.isfile(sp_model_prefix + '.model'):
             # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                input=data_file, vocab_size=vocab_size,
                model_type='bpe', model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name,
                user_defined_symbols=[self.EMPTY_TOK]
            )
        
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model')
        self.vocab = [
            self.sp_model.id_to_piece(ind) for ind in range(self.sp_model.get_piece_size())
        ]
        print(f'VOCAB_SIZE={len(self.vocab)}')
        
    def __len__(self):
        return self.sp_model.vocab_size()

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.sp_model.decode(item)

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor(self.sp_model.encode(text)).unsqueeze(0)
        except KeyError as e:
            raise Exception(
                f"Can't encode text '{text}'.")
            
    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return ''.join([self.sp_model.decode_ids([int(ind)]) for ind in vector]).strip()
    
    def ctc_decode(self, inds: List[int]) -> str:
        result = []
        
        last_ind = self.sp_model.encode_as_ids(self.EMPTY_TOK)[-1]
        print('EMPTY_TOK IND =', last_ind)
        for ind in inds:
            if ind == self.sp_model.encode_as_ids(self.EMPTY_TOK)[-1]:
                last_ind = ind
                continue
            if ind != last_ind:
                try:
                    result.append(self.sp_model.decode_ids([int(ind)]))
                except:
                    raise ValueError(f'input ind = {ind}')
            
            last_ind = ind
        
        return ''.join(result).replace(' ⁇ ', '')

    def ctc_beam_search_decode(self, probs: torch.tensor, probs_length,
                               beam_size: int = 4) -> str:
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == self.__len__()
        
        
        return self.ctc_beam_search(probs, probs_length, beam_size)[0].text.lower().replace(' ⁇ ', '').replace("'", '').replace('_', '')
    
    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 4) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == self.__len__()
        hypos: List[Hypothesis] = []
        
        state = {('', self.EMPTY_TOK): 1.0}
        for i in range(probs_length):
            frame = probs[i]
            state = self._extend_and_merge(frame, state)
            state = self._truncate(state, beam_size)
        
        hypos = [Hypothesis(text, prob) for (text, char), prob in state.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def _extend_and_merge(self, frame: torch.tensor, 
                          state: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        """
        Extends state dictionary
        """
        new_state = defaultdict(float)
        for next_char_index, next_char_proba in enumerate(frame):
            for (pref, last_char), pref_proba in state.items():
                next_char = self.sp_model.decode_ids([next_char_index])
                if next_char == last_char or next_char == self.EMPTY_TOK:
                    new_prefix = pref
                else:
                    new_prefix = pref + next_char
                
                last_char = next_char
                new_state[(new_prefix, last_char)] += pref_proba * next_char_proba
                
        return new_state

    def _truncate(self, state: Dict[Tuple[str, str], float], beam_size: int):
        """
        Truncates a state dictionary to the top most probable 'beam_size' entries
        """
        state_list =  list(state.items())
        state_list.sort(key=lambda x: -x[1])
        return dict(state_list[:beam_size])
