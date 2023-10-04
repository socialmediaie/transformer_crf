import evaluate
import numpy as np
import torch
from transformers import DataCollatorForTokenClassification

seqeval = evaluate.load("seqeval")


def get_aligned_tags(ner_tags, label_mask, ner_tag_id, mask_label_id=-100):
    aligned_tags = [
        ner_tags[i - 1] if mask else mask_label_id
        for i, mask in zip(ner_tag_id, label_mask)
    ]
    return aligned_tags


def create_batch(tokenizer, batch, label_col=None):
    inputs = tokenizer.batch_encode_plus(
        batch["tokens"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        is_split_into_words=True,
        return_special_tokens_mask=True,
        return_length=True,
        return_offsets_mapping=True,
    )
    # Start offest for first subword in tokens is 0
    label_mask = (
        (inputs["special_tokens_mask"] == 0) & (inputs["offset_mapping"][:, :, 0] == 0)
    ).int()
    ner_tag_id = label_mask.cumsum(axis=-1)
    labels = None
    if label_col is not None:
        labels = torch.LongTensor(
            [
                get_aligned_tags(ner_tags, mask, tag_ids, mask_label_id=-100)
                for ner_tags, mask, tag_ids in zip(
                    batch[label_col], label_mask, ner_tag_id
                )
            ]
        )
    else:
        inputs["label_mask"] = label_mask
        inputs["ner_tag_id"] = ner_tag_id
    return inputs, labels


class NERDataCollator(DataCollatorForTokenClassification):
    def __init__(self, label_col, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_col = label_col

    def create_batch(self, batch):
        # print(batch)

        inputs = self.tokenizer.batch_encode_plus(
            batch["tokens"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_special_tokens_mask=True,
            return_length=True,
            return_offsets_mapping=True,
        )
        # Start offest for first subword in tokens is 0
        label_mask = (
            (inputs["special_tokens_mask"] == 0)
            & (inputs["offset_mapping"][:, :, 0] == 0)
        ).int()
        ner_tag_id = label_mask.cumsum(axis=-1)
        labels = None
        if self.label_col in batch:
            labels = torch.LongTensor(
                [
                    get_aligned_tags(ner_tags, mask, tag_ids, mask_label_id=-100)
                    for ner_tags, mask, tag_ids in zip(
                        batch[self.label_col], label_mask, ner_tag_id
                    )
                ]
            )
        return inputs, labels

    def torch_call(self, features):
        batch = {k: [] for k in features[0].keys()}
        for b in features:
            for k in batch:
                batch[k].append(b[k])
        inputs, labels = self.create_batch(batch)
        inputs = {
            k: inputs[k] for k in ["input_ids", "token_type_ids", "attention_mask"]
        }
        if labels is not None:
            inputs["labels"] = labels
        return inputs


def get_crf_predictions(predictions, labels, label_list):
    # predictions = np.argmax(predictions, axis=2)
    predictions = predictions[3]  # 3rd element in crf_prediction is the best_path

    true_predictions, true_labels = [], []
    for prediction, label in zip(predictions, labels):
        true_predictions.append([])
        true_labels.append([])
        for p, l in zip(prediction, label):
            if l == -100:
                continue
            true_predictions[-1].append(label_list[p])
            true_labels[-1].append(label_list[l])
    return true_predictions, true_labels


class GetCRFMetrics(object):
    def __init__(self, label_list):
        self.label_list = label_list

    def __call__(self, p):
        predictions, labels = p
        true_predictions, true_labels = get_crf_predictions(
            predictions, labels, self.label_list
        )
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return results
        # return {
        #     "precision": results["overall_precision"],
        #     "recall": results["overall_recall"],
        #     "f1": results["overall_f1"],
        #     "accuracy": results["overall_accuracy"],
        # }
