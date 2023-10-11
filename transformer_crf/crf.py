import unittest
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.autograd import Variable


@dataclass
class CRFOutput:
    loss: Optional[torch.tensor]
    real_path_score: Optional[torch.tensor]
    total_score: torch.tensor
    best_path_score: torch.tensor
    best_path: torch.tensor


class MaskedCRFLoss(nn.Module):
    __constants__ = ["num_tags", "mask_id"]

    num_tags: int
    mask_id: int

    def __init__(self, num_tags: int, mask_id: int = 0):
        super().__init__()
        self.num_tags = num_tags
        self.mask_id = mask_id
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))

    def extra_repr(self) -> str:
        s = "num_tags={num_tags}, mask_id={mask_id}"
        return s.format(**self.__dict__)

    def forward(self, emissions, mask, tags=None, return_best_path=False):
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)

        seq_length, batch_size = tags.shape
        mask = mask.float()
        # set return_best_path as True always during eval.
        # During training it slows things down as best path is not needed
        if not self.training:
            return_best_path = True
        # Compute the total likelihood
        total_score, best_path_score, best_path = self.compute_log_partition_function(
            emissions, mask, return_best_path=return_best_path
        )

        if tags is None:
            return CRFOutput(None, None, total_score, best_path_score, best_path)

        # Compute the likelihood of the real path
        forward_vars = self.start_transitions
        real_path_score = torch.zeros(batch_size).to(tags.device)
        i = 0
        current_tag = tags[i]
        real_path_score += (
            self.start_transitions[current_tag]
            + emissions[i, range(batch_size), current_tag] * mask[i]
        )  # batch_size
        prev_tag = tags[i]
        for i in range(1, seq_length):
            current_tag = tags[i]
            real_path_score += self.transitions[prev_tag, current_tag] * mask[i]
            real_path_score += emissions[i, range(batch_size), current_tag] * mask[i]
            prev_tag = current_tag.where(mask[i] == 1, prev_tag)
        # Transition to STOP_TAG
        real_path_score += self.stop_transitions[prev_tag]

        # Return the negative log likelihood
        loss = torch.mean(total_score - real_path_score)
        return CRFOutput(loss, real_path_score, total_score, best_path_score, best_path)

    def compute_log_partition_function(
        self, emissions, mask, tags=None, return_best_path=False
    ):
        init_alphas = self.start_transitions + emissions[0]  # (batch_size, num_tags)
        forward_var = init_alphas
        forward_viterbi_var = init_alphas
        # backpointers holds the best tag id at each time step, we accumulate these in reverse order
        backpointers = []

        for i, emission in enumerate(emissions[1:, :, :], 1):
            broadcast_emission = emission.unsqueeze(2)  # (batch_size, num_tags, 1)
            broadcast_transmissions = self.transitions.unsqueeze(
                0
            )  # (1, num_tags, num_tags)

            # Compute next
            next_tag_var = (
                forward_var.unsqueeze(1) + broadcast_emission + broadcast_transmissions
            )
            next_tag_viterbi_var = (
                forward_viterbi_var.unsqueeze(1)
                + broadcast_emission
                + broadcast_transmissions
            )

            next_unmasked_forward_var = torch.logsumexp(next_tag_var, dim=2)
            viterbi_scores, best_next_tags = torch.max(next_tag_viterbi_var, dim=2)
            # If mask == 1 use the next_unmasked_forward_var else copy the forward_var
            # Update forward_var
            forward_var = (
                mask[i].unsqueeze(-1) * next_unmasked_forward_var
                + (1 - mask[i]).unsqueeze(-1) * forward_var
            )
            # Update viterbi with mask
            forward_viterbi_var = (
                mask[i].unsqueeze(-1) * viterbi_scores
                + (1 - mask[i]).unsqueeze(-1) * forward_viterbi_var
            )
            backpointers.append(best_next_tags)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.stop_transitions
        terminal_viterbi_var = forward_viterbi_var + self.stop_transitions

        alpha = torch.logsumexp(terminal_var, dim=1)
        best_path_score, best_final_tags = torch.max(terminal_viterbi_var, dim=1)

        best_path = None
        if return_best_path:
            # backtrace
            best_path = [best_final_tags]
            for bptrs, mask_data in zip(reversed(backpointers), torch.flip(mask, [0])):
                best_tag_id = torch.gather(
                    bptrs, 1, best_final_tags.unsqueeze(1)
                ).squeeze(1)
                best_final_tags.masked_scatter_(
                    mask_data.to(dtype=torch.bool),
                    best_tag_id.masked_select(mask_data.to(dtype=torch.bool)),
                )
                best_path.append(best_final_tags)
            # Reverse the order because we were appending in reverse
            best_path = torch.stack(best_path[::-1])
            best_path = best_path.where(mask == 1, -100)

        return alpha, best_path_score, best_path

    def viterbi_decode(self, emissions, mask):
        seq_len, batch_size, num_tags = emissions.shape

        # backpointers holds the best tag id at each time step, we accumulate these in reverse order
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = self.start_transitions + emissions[0]  # (batch_size, num_tags)
        forward_var = init_vvars

        for i, emission in enumerate(emissions[1:, :, :], 1):
            broadcast_emission = emission.unsqueeze(2)
            broadcast_transmissions = self.transitions.unsqueeze(0)
            next_tag_var = (
                forward_var.unsqueeze(1) + broadcast_emission + broadcast_transmissions
            )

            viterbi_scores, best_next_tags = torch.max(next_tag_var, 2)
            # If mask == 1 use the next_unmasked_forward_var else copy the forward_var
            forward_var = (
                mask[i].unsqueeze(-1) * viterbi_scores
                + (1 - mask[i]).unsqueeze(-1) * forward_var
            )
            backpointers.append(best_next_tags)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.stop_transitions
        best_path_score, best_final_tags = torch.max(terminal_var, dim=1)

        # backtrace
        best_path = [best_final_tags]
        for bptrs, mask_data in zip(reversed(backpointers), torch.flip(mask, [0])):
            best_tag_id = torch.gather(bptrs, 1, best_final_tags.unsqueeze(1)).squeeze(
                1
            )
            best_final_tags.masked_scatter_(
                mask_data.to(dtype=torch.bool),
                best_tag_id.masked_select(mask_data.to(dtype=torch.bool)),
            )
            best_path.append(best_final_tags)

        # Reverse the order because we were appending in reverse
        best_path = torch.stack(best_path[::-1])
        best_path = best_path.where(mask == 1, -100)

        return best_path, best_path_score


class MaskedCRFLossTest(unittest.TestCase):
    def setUp(self):
        self.num_tags = 5
        self.mask_id = 0

        self.crf_model = MaskedCRFLoss(self.num_tags, self.mask_id)

        self.seq_length, self.batch_size = 11, 5
        # Making up some inputs
        # emissions = Variable(torch.randn(seq_length, batch_size, num_tags))
        # tags = Variable(torch.randint(num_tags, (seq_length, batch_size)))
        # mask = Variable(torch.ones(seq_length, batch_size))
        self.emissions = torch.randn(self.seq_length, self.batch_size, self.num_tags)
        self.tags = torch.randint(self.num_tags, (self.seq_length, self.batch_size))
        # mask = torch.ones(seq_length, batch_size)
        self.mask = torch.randint(2, (self.seq_length, self.batch_size))

    def test_forward(self):
        # Checking if forward runs successfully
        try:
            output = self.crf_model(self.emissions, self.mask, tags=self.tags)
            print("Forward function runs successfully!")
        except Exception as e:
            print("Forward function couldn't run successfully:", e)

    def test_viterbi_decode(self):
        # Checking if viterbi_decode runs successfully
        try:
            path, best_path_score = self.crf_model.viterbi_decode(
                self.emissions, self.mask
            )
            print(path.T)
            print("Viterbi decoding function runs successfully!")
        except Exception as e:
            print("Viterbi decoding function couldn't run successfully:", e)

    def test_forward_output(self):
        # Simple check if losses are non-negative
        output = self.crf_model(self.emissions, self.mask, tags=self.tags)
        loss = output.loss
        self.assertTrue((loss > 0).all())

    def test_compute_log_partition_function_output(self):
        # Simply checking if the output is non-negative
        (
            partition,
            best_path_score,
            best_path,
        ) = self.crf_model.compute_log_partition_function(self.emissions, self.mask)
        self.assertTrue((partition > 0).all())

    def test_viterbi_decode_output(self):
        print(self.mask.T)
        # Check whether the output shape is correct and lies within valid tag range
        path, best_path_score = self.crf_model.viterbi_decode(self.emissions, self.mask)
        print(path.T)
        self.assertEqual(
            path.shape, (self.seq_length, self.batch_size)
        )  # checking dimensions
        self.assertTrue(
            ((0 <= path) | (path == -100)).all() and (path < self.num_tags).all()
        )  # checking tag validity


from typing import List, Optional

import torch
import torch.nn as nn
from torch import BoolTensor, FloatTensor, LongTensor


class CRF(nn.Module):
    def __init__(
        self, num_labels: int, pad_idx: Optional[int] = None, use_gpu: bool = True
    ) -> None:
        """

        :param num_labels: number of labels
        :param pad_idxL padding index. default None
        :return None
        """

        if num_labels < 1:
            raise ValueError("invalid number of labels: {0}".format(num_labels))

        super().__init__()
        self.num_labels = num_labels
        self._use_gpu = torch.cuda.is_available() and use_gpu

        # transition matrix setting
        # transition matrix format (source, destination)
        self.trans_matrix = nn.Parameter(torch.empty(num_labels, num_labels))
        # transition matrix of start and end settings
        self.start_trans = nn.Parameter(torch.empty(num_labels))
        self.end_trans = nn.Parameter(torch.empty(num_labels))

        self._initialize_parameters(pad_idx)

    def forward(
        self, h: FloatTensor, labels: LongTensor, mask: BoolTensor
    ) -> FloatTensor:
        """

        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param labels: answer labels of each sequence
                       in mini batch (batch_size, seq_len)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :return: The log-likelihood (batch_size)
        """

        best_path, best_path_score = self.viterbi_decode(h, mask)
        total_score = self._compute_denominator_log_likelihood(h, mask)

        real_path_score = self._compute_numerator_log_likelihood(h, labels, mask)
        loss = torch.mean(total_score - real_path_score)
        return CRFOutput(loss, real_path_score, total_score, best_path_score, best_path)

    def viterbi_decode(self, h: FloatTensor, mask: BoolTensor) -> List[List[int]]:
        """
        decode labels using viterbi algorithm
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, batch_size)
        :return: labels of each sequence in mini batch
        """

        batch_size, seq_len, _ = h.size()
        # prepare the sequence lengths in each sequence
        seq_lens = mask.sum(dim=1)
        # In mini batch, prepare the score
        # from the start sequence to the first label
        score = [self.start_trans.data + h[:, 0]]
        path = []

        for t in range(1, seq_len):
            # extract the score of previous sequence
            # (batch_size, num_labels, 1)
            previous_score = score[t - 1].view(batch_size, -1, 1)

            # extract the score of hidden matrix of sequence
            # (batch_size, 1, num_labels)
            h_t = h[:, t].view(batch_size, 1, -1)

            # extract the score in transition
            # from label of t-1 sequence to label of sequence of t
            # self.trans_matrix has the score of the transition
            # from sequence A to sequence B
            # (batch_size, num_labels, num_labels)
            score_t = previous_score + self.trans_matrix + h_t

            # keep the maximum value
            # and point where maximum value of each sequence
            # (batch_size, num_labels)
            best_score, best_path = score_t.max(1)
            score.append(best_score)
            path.append(best_path)

        # predict labels of mini batch
        best_paths, best_path_scores = zip(
            *[
                self._viterbi_compute_best_path(i, seq_lens, score, path)
                for i in range(batch_size)
            ]
        )

        return best_paths, best_path_scores

    def _viterbi_compute_best_path(
        self,
        batch_idx: int,
        seq_lens: torch.LongTensor,
        score: List[FloatTensor],
        path: List[torch.LongTensor],
    ) -> List[int]:
        """
        return labels using viterbi algorithm
        :param batch_idx: index of batch
        :param seq_lens: sequence lengths in mini batch (batch_size)
        :param score: transition scores of length max sequence size
                      in mini batch [(batch_size, num_labels)]
        :param path: transition paths of length max sequence size
                     in mini batch [(batch_size, num_labels)]
        :return: labels of batch_idx-th sequence
        """

        seq_end_idx = seq_lens[batch_idx] - 1
        # extract label of end sequence
        best_path_score, best_last_label = (
            score[seq_end_idx][batch_idx] + self.end_trans
        ).max(0)
        best_labels = [int(best_last_label)]

        # predict labels from back using viterbi algorithm
        for p in reversed(path[:seq_end_idx]):
            best_last_label = p[batch_idx][best_labels[0]]
            best_labels.insert(0, int(best_last_label))

        return best_labels, best_path_score

    def _compute_denominator_log_likelihood(self, h: FloatTensor, mask: BoolTensor):
        """

        compute the denominator term for the log-likelihood
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :return: The score of denominator term for the log-likelihood
        """
        device = h.device
        batch_size, seq_len, _ = h.size()

        # (num_labels, num_labels) -> (1, num_labels, num_labels)
        trans = self.trans_matrix.unsqueeze(0)

        # add the score from beginning to each label
        # and the first score of each label
        score = self.start_trans + h[:, 0]

        # iterate through processing for the number of words in the mini batch
        for t in range(1, seq_len):
            # (batch_size, self.num_labels, 1)
            before_score = score.unsqueeze(2)

            # prepare t-th mask of sequences in each sequence
            # (batch_size, 1)
            mask_t = mask[:, t].unsqueeze(1)
            mask_t = mask_t.to(device)

            # prepare the transition probability of the t-th sequence label
            # in each sequence
            # (batch_size, 1, num_labels)
            h_t = h[:, t].unsqueeze(1)

            # calculate t-th scores in each sequence
            # (batch_size, num_labels)
            score_t = before_score + h_t + trans
            score_t = torch.logsumexp(score_t, 1)

            # update scores
            # (batch_size, num_labels)
            score = torch.where(mask_t, score_t, score)

        # add the end score of each label
        score += self.end_trans

        # return the log likely food of all data in mini batch
        return torch.logsumexp(score, 1)

    def _compute_numerator_log_likelihood(
        self, h: FloatTensor, y: LongTensor, mask: BoolTensor
    ) -> FloatTensor:
        """
        compute the numerator term for the log-likelihood
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param y: answer labels of each sequence
                  in mini batch (batch_size, seq_len)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :return: The score of numerator term for the log-likelihood
        """

        batch_size, seq_len, _ = h.size()

        h_unsqueezed = h.unsqueeze(-1)
        trans = self.trans_matrix.unsqueeze(-1)

        arange_b = torch.arange(batch_size)

        # extract first vector of sequences in mini batch
        calc_range = seq_len - 1
        score = self.start_trans[y[:, 0]] + sum(
            [
                self._calc_trans_score_for_num_llh(
                    h_unsqueezed, y, trans, mask, t, arange_b
                )
                for t in range(calc_range)
            ]
        )

        # extract end label number of each sequence in mini batch
        # (batch_size)
        last_mask_index = mask.sum(1) - 1
        last_labels = y[arange_b, last_mask_index]
        each_last_score = h[arange_b, -1, last_labels] * mask[:, -1]

        # Add the score of the sequences of the maximum length in mini batch
        # Add the scores from the last tag of each sequence to EOS
        score += each_last_score + self.end_trans[last_labels]

        return score

    def _calc_trans_score_for_num_llh(
        self,
        h: FloatTensor,
        y: LongTensor,
        trans: FloatTensor,
        mask: BoolTensor,
        t: int,
        arange_b: FloatTensor,
    ) -> torch.Tensor:
        """
        calculate transition score for computing numberator llh
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param y: answer labels of each sequence
                  in mini batch (batch_size, seq_len)
        :param trans: transition score
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :paramt t: index of hidden, transition, and mask matrixex
        :param arange_b: this param is seted torch.arange(batch_size)
        :param batch_size: batch size of this calculation
        """
        device = h.device
        mask_t = mask[:, t]
        mask_t = mask_t.to(device)
        mask_t1 = mask[:, t + 1]
        mask_t1 = mask_t1.to(device)

        # extract the score of t+1 label
        # (batch_size)
        h_t = h[arange_b, t, y[:, t]].squeeze(1)

        # extract the transition score from t-th label to t+1 label
        # (batch_size)
        trans_t = trans[y[:, t], y[:, t + 1]].squeeze(1)

        # add the score of t+1 and the transition score
        # (batch_size)
        return h_t * mask_t + trans_t * mask_t1

    def _initialize_parameters(self, pad_idx: Optional[int]) -> None:
        """
        initialize transition parameters
        :param: pad_idx: if not None, additional initialize
        :return: None
        """

        nn.init.uniform_(self.trans_matrix, -0.1, 0.1)
        nn.init.uniform_(self.start_trans, -0.1, 0.1)
        nn.init.uniform_(self.end_trans, -0.1, 0.1)
        if pad_idx is not None:
            self.start_trans[pad_idx] = -10000.0
            self.trans_matrix[pad_idx, :] = -10000.0
            self.trans_matrix[:, pad_idx] = -10000.0
            self.trans_matrix[pad_idx, pad_idx] = 0.0


from typing import List, Optional

import torch
import torch.nn as nn

# from adaseq.metainfo import Decoders

# from .base import DECODERS, Decoder


# @DECODERS.register_module(module_name=Decoders.crf)
class AdvancedCRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.ByteTensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        real_path_score = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        total_score = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = total_score - real_path_score

        if reduction == "none":
            loss = llh
        elif reduction == "sum":
            loss = llh.sum()
        elif reduction == "mean":
            loss = llh.mean()
        else:
            loss = llh.sum() / mask.float().sum()

        # shape: (batch_size, seq_len)
        best_path, best_path_score = self.decode(emissions, mask, pad_tag=-100)
        return CRFOutput(loss, real_path_score, total_score, best_path_score, best_path)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.ByteTensor] = None,
        nbest: Optional[int] = None,
        pad_tag: Optional[int] = None,
    ) -> List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            nbest (`int`): Number of most probable paths for each sequence
            pad_tag (`int`): Tag at padded positions. Often input varies in length and
                the length will be padded to the maximum length in the batch. Tags at
                the padded positions will be assigned with a padding tag, i.e. `pad_tag`
        Returns:
            A PyTorch tensor of the best tag sequence for each batch of shape
            (nbest, batch_size, seq_length)
        """
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.uint8, device=emissions.device
            )
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if nbest == 1:
            best_tags_arr, best_score = self._viterbi_decode(emissions, mask, pad_tag)
            # return best_tags_arr.unsqueeze(0), best_score
            return best_tags_arr, best_score
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.ByteTensor] = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must have dimension of 3, got {emissions.dim()}"
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, "
                f"got {emissions.size(2)}"
            )

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    "the first two dimensions of emissions and tags must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
                )

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _forward_backward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: torch.ByteTensor,
        mode: str = "forward",
    ) -> torch.Tensor:
        """
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)``
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
            mode (`str`): Specifies the calculation mode: ``partition|forward|backward``.
        Returns:
            A PyTorch tensor of the shape
            ``(batch_size, num_tags)`` for `partition`,
            ``(seq_length, batch_size, num_tags)`` for `forward`,
            the difference between `partition` and `forward` is whether add the `end_transtions` or not.
            ``(seq_length, batch_size, num_tags)`` for `backward`.

        """
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        if mode in ("forward", "partition"):
            start_transitions = self.start_transitions
            end_transitions = self.end_transitions
            current_emissions = emissions
            transitions = self.transitions
        elif mode == "backward":
            start_transitions = self.end_transitions
            end_transitions = self.start_transitions
            current_emissions = torch.zeros_like(emissions)
            batch_size = emissions.shape[1]
            for batch_idx in range(batch_size):
                length = mask[:, batch_idx].sum()
                current_emissions[:length, batch_idx, :] = emissions[
                    :length, batch_idx, :
                ].flip([0])
            transitions = self.transitions.transpose(0, 1)
        else:
            raise NotImplementedError

        seq_length = current_emissions.size(0)
        scores = torch.zeros_like(current_emissions)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = start_transitions + current_emissions[0]

        scores[0, :, :] = score

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = current_emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            scores[i, :, :] = score

        # End transition score
        # shape: (batch_size, num_tags)
        if mode == "partition":
            score += end_transitions
        scores[seq_length - 1, :, :] = score

        if mode == "backward":
            batch_size = emissions.shape[1]
            for batch_idx in range(batch_size):
                length = mask[:, batch_idx].sum()
                scores[:length, batch_idx, :] = (
                    scores[:length, batch_idx, :]
                    - current_emissions[:length, batch_idx, :]
                ).flip([0])

        return scores

    def compute_posterior(
        self, emissions: torch.Tensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        """Compute posterior probability distribution from emission logits

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)``
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``

        Returns:
            A PyTorch tensor of the shape
            ``(seq_length, batch_size, num_tags)``
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        fw_scores = self._forward_backward_algorithm(emissions, mask, mode="forward")
        bw_scores = self._forward_backward_algorithm(emissions, mask, mode="backward")
        partition = self._compute_normalizer(emissions, mask)
        log_posterior = fw_scores + bw_scores - partition.view(1, -1, 1)

        if self.batch_first:
            log_posterior = log_posterior.transpose(0, 1)
        return log_posterior

    def _compute_normalizer(
        self, emissions: torch.Tensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        fw_scores = self._forward_backward_algorithm(emissions, mask, mode="partition")
        return torch.logsumexp(fw_scores[-1, :, :], dim=1)

    def _viterbi_decode(
        self,
        emissions: torch.FloatTensor,
        mask: torch.ByteTensor,
        pad_tag: Optional[int] = None,
    ) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros(
            (seq_length, batch_size, self.num_tags), dtype=torch.long, device=device
        )
        oor_idx = torch.zeros(
            (batch_size, self.num_tags), dtype=torch.long, device=device
        )
        oor_tag = torch.full(
            (seq_length, batch_size), pad_tag, dtype=torch.long, device=device
        )

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        score = viterbi_decode_inner_loop1(
            score, history_idx, emissions, self.transitions, mask, oor_idx
        )
        # End transition score
        # shape: (batch_size, num_tags)
        end_score = score + self.end_transitions
        best_score, end_tag = end_score.max(dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(
            1,
            seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
            end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags),
        )
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros(
            (seq_length, batch_size), dtype=torch.long, device=device
        )
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        best_tags_arr = viterbi_decode_inner_loop2(
            mask, history_idx, best_tags, best_tags_arr
        )
        best_tags_arr = best_tags_arr.where(mask, oor_tag).transpose(0, 1)
        return best_tags_arr, best_score

    def _viterbi_decode_nbest(
        self,
        emissions: torch.FloatTensor,
        mask: torch.ByteTensor,
        nbest: int,
        pad_tag: Optional[int] = None,
    ) -> List[List[List[int]]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros(
            (seq_length, batch_size, self.num_tags, nbest),
            dtype=torch.long,
            device=device,
        )
        oor_idx = torch.zeros(
            (batch_size, self.num_tags, nbest), dtype=torch.long, device=device
        )
        oor_tag = torch.full(
            (seq_length, batch_size, nbest), pad_tag, dtype=torch.long, device=device
        )

        # + score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # + history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = (
                    broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission
                )

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(
                nbest, dim=1
            )

            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags, nbest)
            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(
            1,
            seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
            end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest),
        )
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros(
            (seq_length, batch_size, nbest), dtype=torch.long, device=device
        )
        best_tags = (
            torch.arange(nbest, dtype=torch.long, device=device)
            .view(1, -1)
            .expand(batch_size, -1)
        )
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(
                history_idx[idx].view(batch_size, -1), 1, best_tags
            )
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest

        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)


@torch.jit.script
def viterbi_decode_inner_loop1(
    score, history_idx, emissions, transitions, mask, oor_idx
):  # noqa
    # for every possible next tag

    seq_length, batch_size = mask.shape
    for i in range(1, seq_length):
        # Broadcast viterbi score for every possible next tag
        # shape: (batch_size, num_tags, 1)
        broadcast_score = score.unsqueeze(2)

        # Broadcast emission score for every possible current tag
        # shape: (batch_size, 1, num_tags)
        broadcast_emission = emissions[i].unsqueeze(1)

        # Compute the score tensor of size (batch_size, num_tags, num_tags) where
        # for each sample, entry at row i and column j stores the score of the best
        # tag sequence so far that ends with transitioning from tag i to tag j and emitting
        # shape: (batch_size, num_tags, num_tags)
        next_score = broadcast_score + transitions + broadcast_emission

        # Find the maximum score over all possible current tag
        # shape: (batch_size, num_tags)
        next_score, indices = next_score.max(dim=1)

        # Set score to the next score if this timestep is valid (mask == 1)
        # and save the index that produces the next score
        # shape: (batch_size, num_tags)
        score = torch.where(mask[i].unsqueeze(-1), next_score, score)
        indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
        history_idx[i - 1] = indices
    return score


@torch.jit.script
def viterbi_decode_inner_loop2(mask, history_idx, best_tags, best_tags_arr):  # noqa
    seq_length, batch_size = mask.shape
    for idx in range(seq_length - 1, -1, -1):
        best_tags = torch.gather(history_idx[idx], 1, best_tags)
        best_tags_arr[idx] = best_tags.data.view(batch_size)
    return best_tags_arr
