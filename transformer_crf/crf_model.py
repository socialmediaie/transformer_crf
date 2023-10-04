from dataclasses import dataclass
from typing import Optional, Tuple

import datasets
import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from .crf import MaskedCRFLoss


@dataclass
class TokenClassifierCRFOutput(TokenClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    real_path_score: Optional[torch.FloatTensor] = None
    total_score: torch.FloatTensor = None
    best_path_score: torch.FloatTensor = None
    best_path: Optional[torch.LongTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PretrainedCRFModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModelForTokenClassification.from_pretrained(
            config._name_or_path, config=config
        )
        self.crf_model = MaskedCRFLoss(self.config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        return_best_path=False,
        **kwargs
    ):
        encoder_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Convert output to seq length as first dim

        emissions = encoder_output.logits.transpose(1, 0)
        tags = labels.transpose(1, 0)
        mask = tags != -100
        tags = tags.where(mask, 0)  # CRF cant support -100 id

        crf_output = self.crf_model(
            emissions, tags, mask, return_best_path=return_best_path
        )

        # Convert best_path to batch first
        best_path = crf_output.best_path
        if best_path is not None:
            best_path = best_path.transpose(1, 0)

        output = TokenClassifierCRFOutput(
            loss=crf_output.loss,
            real_path_score=crf_output.real_path_score,
            total_score=crf_output.total_score,
            best_path_score=crf_output.best_path_score,
            best_path=best_path,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )
        return output
