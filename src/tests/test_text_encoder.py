import unittest

import torch

from src.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        # TODO
        # text_encoder = CTCCharTextEncoder()
        
        # vocab_size = len(text_encoder.ind2char)
        # seq_len = 10
        
        # probs = torch.rand(seq_len, vocab_size)
        # row_sums = probs.sum(dim=1, keepdim=True)
        # probs /= row_sums
        # log_probs = torch.log(log_probs)
        