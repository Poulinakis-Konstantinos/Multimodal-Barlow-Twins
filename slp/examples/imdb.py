import pytorch_lightning as pl
import torch.nn as nn
from loguru import logger
from torch.optim import Adam
from torchnlp.datasets import imdb_dataset  # type: ignore

from slp.data.collators import SequenceClassificationCollator
from slp.modules.classifier import RNNTokenSequenceClassifier
from slp.plbind.dm import PLDataModuleFromCorpus
from slp.plbind.helpers import FromLogits
from slp.plbind.module import RnnPLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.util.log import configure_logging

MAX_LENGTH = 1024
collate_fn = SequenceClassificationCollator(device="cpu", max_length=MAX_LENGTH)
# collate_fn = SequenceClassificationCollator(device="cpu")


if __name__ == "__main__":
    pl.utilities.seed.seed_everything(seed=42)
    EXPERIMENT_NAME = "imdb-words-sentiment-classification"

    configure_logging(f"logs/{EXPERIMENT_NAME}")

    train, test = imdb_dataset(directory="./data/", train=True, test=True)

    raw_train = [d["text"] for d in train]
    labels_train = [d["sentiment"] for d in train]

    raw_test = [d["text"] for d in test]
    labels_test = [d["sentiment"] for d in test]

    ldm = PLDataModuleFromCorpus(
        raw_train,
        labels_train,
        test=raw_test,
        test_labels=labels_test,
        batch_size=64,
        batch_size_eval=32,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=1,
        tokens="words",
        embeddings_file="./cache/glove.6B.50d.txt",
        embeddings_dim=50,
        lower=True,
        max_length=MAX_LENGTH,
        limit_vocab_size=-1,
        lang="en_core_web_md",
    )
    ldm.setup()

    model = RNNTokenSequenceClassifier(
        3,
        embeddings=ldm.embeddings,
        bidirectional=True,
        merge_bi="sum",
        finetune_embeddings=True,
        attention=True,
        nystrom=True,
        num_landmarks=32,
        num_heads=2,
    )

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    lm = RnnPLModule(
        model,
        optimizer,
        criterion,
        metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
    )

    trainer = make_trainer(
        EXPERIMENT_NAME,
        max_epochs=100,
        gpus=1,
        save_top_k=1,
    )
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())
