import logging
from typing import List, NamedTuple, Dict, Tuple
from collections import defaultdict

import torch
from pyctcdecode import build_ctcdecoder

from .char_text_encoder import CharTextEncoder

logger = logging.getLogger(__name__)


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lm_model_path: str = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.lm_ctcdecoder = None
        
        if lm_model_path is not None:
            self.lm_ctcdecoder = build_ctcdecoder(
                labels=[''] + [char.upper() for char in self.alphabet],
                kenlm_model_path=lm_model_path
            )
        
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        result = []
        
        last_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind == self.char2ind[self.EMPTY_TOK]:
                last_ind = ind
                continue
            if ind != last_ind:
                result.append(self.ind2char[ind])
            
            last_ind = ind
        
        return ''.join(result)
    
    def ctc_beam_search_decode(self, probs: torch.tensor, probs_length,
                               beam_size: int = 4, use_lm: bool = True) -> str:
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        
        if self.lm_ctcdecoder is None:
            return self.ctc_beam_search(probs, probs_length, beam_size)[0].text
        
        return self.lm_ctcdecoder.decode(probs[:probs_length].detach().cpu().numpy(), beam_width=beam_size).lower().replace("'", '')
    
    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 4) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
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
                next_char = self.ind2char[next_char_index]
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
