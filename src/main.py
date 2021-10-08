import json
import os
from argparse import ArgumentParser
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lineflow.core import IterableDataset
from torch.utils.data import DataLoader
from transformers import AdamW, RobertaModel, RobertaTokenizer  # type: ignore


def load_jsonl(dpath: str) -> List[Dict[str, Union[str, int]]]:
    """Load jsonline formatted file given its path."""
    datas = []
    with open(dpath, "r") as _f:
        for line in _f.readlines():
            datas.append(json.loads(line))
    return datas


def preprocess(tokenizer: RobertaTokenizer, d: Dict[str, Union[str, int]]) -> Any:
    """Basic tokenization by pretrained tokenizer."""
    model_inputs = tokenizer(d["s1"], d["s2"], return_tensors="pt", padding="max_length", max_length=256)  # type: ignore
    model_inputs["label"] = torch.LongTensor([int(d["label"])])
    for k in ["input_ids", "attention_mask", "label"]:
        model_inputs[k] = model_inputs[k].squeeze()  # type: ignore
    return model_inputs


def get_dataloaders(dpath: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Load file, preprocess (tokenize), pack into pytorch dataloader."""
    train_ds = IterableDataset(load_jsonl(os.path.join(dpath, "train.jsonl")))
    val_ds = IterableDataset(load_jsonl(os.path.join(dpath, "val.jsonl")))

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    preprocessor = partial(preprocess, tokenizer)

    train_ds = train_ds.map(preprocessor)
    val_ds = val_ds.map(preprocessor)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  # type: ignore
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)  # type: ignore

    return train_dl, val_dl


class Classifier(nn.Module):
    """Classification head to be on top of RoBERTa."""

    def __init__(self, config):
        super().__init__()
        class_n = 2
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, class_n)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RTEModule(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super().__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.classifier = Classifier(config=self.model.config)
        self.accuracy = pl.metrics.Accuracy()  # type: ignore
        self.save_hyperparameters(hparams)

    def forward(self, batch: Dict):
        # Get feature vectors from RoBERTa
        out = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        # Take last hidden state from out, to know the format of `out` refer [here](https://huggingface.co/transformers/model_doc/roberta.html#transformers.RobertaModel.forward)
        last_hidden_state = out[0]

        logits = self.classifier(
            last_hidden_state
        )  # Run classification given features.
        return logits

    def training_step(self, batch, _):
        logits = self.forward(batch)
        loss = F.cross_entropy(logits.view(-1, 2), batch["label"].view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        logits = self.forward(batch)
        loss = F.cross_entropy(logits.view(-1, 2), batch["label"].view(-1))
        self.log("val_loss", loss)

        acc = self.accuracy(logits, batch["label"])
        self.log("val_acc", acc)

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters_roberta = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_grouped_parameters_clf = [
            {
                "params": [
                    p
                    for n, p in self.classifier.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.classifier.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters_roberta + optimizer_grouped_parameters_clf,
            lr=self.hparams["lr"],
        )
        return optimizer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument(
        "--max-epochs", type=int, required=True, help="Maximum epochs to train."
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Maximum epochs to train."
    )
    parser.add_argument(
        "--dpath", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--default-root-dir",
        type=str,
        required=True,
        help="Path to save logs and trained models.",
    )
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use")
    args = parser.parse_args()

    hparams = vars(args)
    train_dl, val_dl = get_dataloaders(hparams["dpath"], hparams["batch_size"])

    module = RTEModule(hparams)

    trainer = pl.Trainer(
        gpus=args.gpus,
        default_root_dir=args.default_root_dir,
        max_epochs=args.max_epochs,
    )
    trainer.fit(module, train_dl, val_dl)
