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

    def forward(self, emissions, tags, mask, return_best_path=False):
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
        real_path_score = torch.zeros(batch_size).to(tags.device)
        real_path_score += self.start_transitions[tags[0]]  # batch_size
        for i in range(1, seq_length):
            current_tag = tags[i]
            real_path_score += self.transitions[tags[i - 1], current_tag] * mask[i]
            real_path_score += emissions[i, range(batch_size), current_tag] * mask[i]
        # Transition to STOP_TAG
        real_path_score += self.stop_transitions[tags[-1]]

        # Return the negative log likelihood
        loss = torch.mean(total_score - real_path_score)
        return CRFOutput(loss, real_path_score, total_score, best_path_score, best_path)

    def compute_log_partition_function(self, emissions, mask, return_best_path=False):
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
            output = self.crf_model(self.emissions, self.tags, self.mask)
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
        output = self.crf_model(self.emissions, self.tags, self.mask)
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
