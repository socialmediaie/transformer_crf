import evaluate
import numpy as np
import torch
from transformers import DataCollatorForTokenClassification

seqeval = evaluate.load("seqeval")


def subword_to_word_embeddings(
    subword_embeddings, valid_subword_mask, subword_tags=None, reduce_str="mean"
):
    batch_size, subword_seq_len, d = subword_embeddings.shape
    # valid_subword_mask = (offset_mapping[:, :, 0] == 0) & (offset_mapping[:, :, 0] != offset_mapping[:, :, 1])
    seq_lengths = valid_subword_mask.sum(axis=-1)
    word_seq_len = seq_lengths.max()

    token_idx_to_word_idx = (valid_subword_mask.cumsum(-1) - 1).clip(0)

    # Zero invalid subword embeddings
    # Make sure to use the correct val instead of 0 if using a different reduce
    subword_embeddings_masked = subword_embeddings.where(
        valid_subword_mask.unsqueeze(-1), 0
    )

    word_embeddings = torch.zeros(batch_size, word_seq_len, d).to(
        subword_embeddings.device
    )
    word_embeddings.scatter_reduce_(
        1,
        token_idx_to_word_idx.unsqueeze(-1).tile([1, 1, d]),
        subword_embeddings,
        reduce_str,
    )

    word_tags = None

    if subword_tags is not None:
        word_tags = torch.full((batch_size, word_seq_len), -100).to(subword_tags.device)
        word_tags.scatter_reduce_(1, token_idx_to_word_idx, subword_tags, "amax")

    return word_embeddings, word_tags


def get_aligned_tags(ner_tags, label_mask, ner_tag_id, mask_label_id=-100):
    aligned_tags = [
        ner_tags[i - 1] if mask else mask_label_id
        for i, mask in zip(ner_tag_id, label_mask)
    ]
    return aligned_tags


def create_batch(batch, tokenizer, label_col="label", out_subword_labels=False):
    # print(batch)

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
    valid_subword_mask = (
        (inputs["special_tokens_mask"] == 0) & (inputs["offset_mapping"][:, :, 0] == 0)
    ).int()
    ner_tag_id = valid_subword_mask.cumsum(axis=-1)
    token_idx_to_word_idx = (ner_tag_id - 1).clip(0)

    inputs["valid_subword_mask"] = valid_subword_mask
    inputs["token_idx_to_word_idx"] = token_idx_to_word_idx

    labels = None
    if label_col in batch:
        if isinstance(batch[label_col][0], list):
            batch[label_col] = [torch.LongTensor(l) for l in batch[label_col]]
        word_tags = torch.nn.utils.rnn.pad_sequence(
            batch[label_col], padding_value=-100, batch_first=True
        )
        labels = word_tags
        if out_subword_labels:
            # Get aligned labels for first subwords
            labels = word_tags.gather(1, token_idx_to_word_idx).where(
                valid_subword_mask == 1, -100
            )
            # labels = torch.LongTensor([
            #    get_aligned_tags(ner_tags, mask, tag_ids, mask_label_id=-100)
            #    for ner_tags, mask, tag_ids in zip(batch[self.label_col], label_mask, ner_tag_id)
            # ])
    return inputs, labels


class NERDataCollator(DataCollatorForTokenClassification):
    def __init__(self, label_col, *args, out_subword_labels=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_col = label_col
        self.out_subword_labels = out_subword_labels

    def create_batch(self, batch):
        inputs, labels = create_batch(
            batch,
            tokenizer=self.tokenizer,
            label_col=self.label_col,
            out_subword_labels=self.out_subword_labels,
        )
        return inputs, labels

    def torch_call(self, features):
        batch = {k: [] for k in features[0].keys()}
        for b in features:
            for k in batch:
                v = b[k]
                if k == self.label_col:
                    v = torch.tensor(v)
                batch[k].append(v)
        inputs, labels = self.create_batch(batch)
        inputs = {
            k: inputs[k]
            for k in [
                "input_ids",
                "token_type_ids",
                "attention_mask",
                "valid_subword_mask",
                # "token_idx_to_word_idx",
            ]
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
