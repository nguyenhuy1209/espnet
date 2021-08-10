import numpy as np
import torch

##################################
from espnet.nets.transducer_decoder_interface import Hypothesis
from dataclasses import asdict
##################################

class SegmentStreamingE2E(object):
    """SegmentStreamingE2E constructor.

    :param E2E e2e: E2E ASR object
    :param recog_args: arguments for "recognize" method of E2E
    """

    def __init__(self, e2e, recog_args, beam_search, rnnlm=None):
        self._e2e = e2e
        self._recog_args = recog_args
        self._char_list = e2e.char_list
        self._rnnlm = rnnlm

        self._e2e.eval()

        self._blank_idx_in_char_list = -1
        for idx in range(len(self._char_list)):
            if self._char_list[idx] == self._e2e.blank:
                self._blank_idx_in_char_list = idx
                break

        self._subsampling_factor = np.prod(e2e.subsample)
        self._activates = 0
        self._blank_dur = 0

        self._previous_input = []
        self._previous_encoder_recurrent_state = None
        self._encoder_states = []
        self._ctc_posteriors = []

        ################################################
        self.beam_search = beam_search
        self.beam = min(beam_search.beam_size, beam_search.vocab_size)
        self.beam_k = min(self.beam, (beam_search.vocab_size - 1))
        self.dec_state = self.beam_search.decoder.init_state(1)
        self.kept_hyps = [Hypothesis(score=0.0, yseq=[beam_search.blank], dec_state=self.dec_state)]
        self.cache = {}

        self._encoder_output = []
        ################################################

        assert (
            self._recog_args.batchsize <= 1
        ), "SegmentStreamingE2E works only with batch size <= 1"
        assert (
            "b" not in self._e2e.etype
        ), "SegmentStreamingE2E works only with uni-directional encoders"

    def accept_input_for_tt(self, x):
        """Call this method each time a new batch of input is available."""

        h = self._e2e.subsample_frames(x)
        self._previous_input.append(h)

        hyp = None

        if self._activates == 0:
            if "custom" in self._e2e.etype:
                h = self._e2e.encode_custom(h)
            else:
                h = self._e2e.encode_rnn(h)
            self._encoder_states.append(h)
            nbest_hyps = self.beam_search(h, self.beam, self.beam_k, self.dec_state, self.kept_hyps, self.cache)
            
            z = nbest_hyps[0].yseq
            if any(z) != self._blank_idx_in_char_list:
                self._activates = 1

        else:
            # h = torch.cat(self._previous_input, dim=0).view(
            #             -1, self._previous_input[0].size(1)
            #         )
            if "custom" in self._e2e.etype:
                h = self._e2e.encode_custom(h)
            else:
                h = self._e2e.encode_rnn(h)
            self._encoder_states.append(h)
            h = torch.cat(self._encoder_states, dim=0).view(
                        -1, self._encoder_states[0].size(1)
                    )
            print(h.size())
            hyp = self.beam_search(h, self.beam, self.beam_k, self.dec_state, self.kept_hyps, self.cache)

        return [asdict(n) for n in hyp] if hyp != None else hyp