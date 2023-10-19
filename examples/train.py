from transformers import Trainer, TrainingArguments
import accelerate
from transformers.training_args import is_accelerate_available

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from transformer_crf.crf_model import PretrainedCRFModel, PretrainedTaggerModel
from transformer_crf.utils import create_batch, NERDataCollator, GetCRFMetrics

from datasets import load_dataset
import datasets

dataset_path = "napsternxg/nyt_ingredients"
dataset = load_dataset(dataset_path)

train_test_dataset = dataset["train"].train_test_split(test_size=0.05)
train_valid_dataset = train_test_dataset["train"].train_test_split(test_size=0.05)
# train_valid_dataset = train_test_dataset["train"].train_test_split(test_size=0.01)
train_valid_dataset

dataset = datasets.DatasetDict(
    {
        "train": train_valid_dataset["train"],
        "validation": train_valid_dataset["test"],
        "test": train_test_dataset["test"],
    }
)

# model_type = "sentence-transformers/paraphrase-MiniLM-L3-v2"
model_type = "napsternxg/gte-small-L3-ingredient-v2"
tokenizer = AutoTokenizer.from_pretrained(model_type)


label_col = "label"  # ner_tags
label_list = dataset["train"].features[label_col].feature.names
num_labels = len(dataset["train"].features[label_col].feature.names)
id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in enumerate(label_list)}

use_crf = False

if use_crf:
    model = PretrainedCRFModel.from_pretrained(
        model_type, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
else:
    model = PretrainedTaggerModel.from_pretrained(
        model_type, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

compute_metrics = GetCRFMetrics(label_list)


infix = "crf-tagger" if use_crf else "tagger"
output_dir = f"{dataset_path.split('/')[1]}-{infix}-{model_type.split('/')[1]}"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    do_train=True,
    do_eval=True,
    # evaluation_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=1000,
    # save_strategy="epoch",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    dataloader_num_workers=10,
    remove_unused_columns=False,
    num_train_epochs=10,
    push_to_hub=False,
    hub_model_id=output_dir,
    load_best_model_at_end=True,
    # label_smoothing_factor=0.1
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=NERDataCollator(
        label_col=label_col, tokenizer=tokenizer, out_subword_labels=False
    ),
    train_dataset=dataset["train"],
    # train_dataset=dataset["validation"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

train_output = trainer.train()

trainer.save_model()
trainer.save_state()

for split in dataset:
    print(split)
    # if split in {"train", "test"}: continue
    metrics = trainer.evaluate(dataset[split])
    trainer.save_metrics(split, metrics)

trainer.push_to_hub()
