from dataclasses import dataclass
from typing import Optional, Tuple

import datasets
import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from .crf import CRF, AdvancedCRF, MaskedCRFLoss
from .utils import subword_to_word_embeddings


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
        self.crf_model = AdvancedCRF(self.config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        return_best_path=False,
        valid_subword_mask=None,
        **kwargs
    ):
        encoder_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        emissions = encoder_output.logits

        if valid_subword_mask is not None:
            mask = valid_subword_mask == 1
            # emissions = subword_to_word_embeddings(encoder_output.logits, mask, reduce_str="mean")
            emissions, _ = subword_to_word_embeddings(
                emissions, mask, subword_tags=None, reduce_str="mean"
            )

        # Convert output to seq length as first dim

        emissions = emissions.transpose(1, 0)
        tags = None
        if labels is not None:
            mask = labels != -100
            tags = labels.where(mask, 0)  # CRF can't use -100 label
            tags = tags.transpose(1, 0)

        mask = mask.transpose(1, 0)

        crf_output = self.crf_model(emissions=emissions, tags=tags, mask=mask)

        # Convert best_path to batch first
        best_path = crf_output.best_path
        if best_path is not None:
            pass
            # best_path = best_path.transpose(1, 0)
            # best_path = [torch.LongTensor(p) for p in best_path]

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
