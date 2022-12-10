import pytorch_lightning as pl
import torch.nn as nn
from loguru import logger
from slp.data.cmusdk import mosi
from slp.data.collators import MultimodalSequenceClassificationCollator
from slp.data.multimodal import MOSI
from slp.modules.classifier import TransformerLateFusionClassifier
from slp.plbind.dm import PLDataModuleFromDatasets
from slp.plbind.helpers import FromLogits
from slp.plbind.module import MultimodalTransformerClassificationPLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.util.log import configure_logging
from torch.optim import AdamW


if __name__ == "__main__":
    EXPERIMENT_NAME = "mosi-transformer"

    configure_logging(f"logs/{EXPERIMENT_NAME}")

    modalities = {"text", "audio", "visual"}
    max_length = 1024
    collate_fn = MultimodalSequenceClassificationCollator(device="cpu")

    train_data, dev_data, test_data, w2v = mosi(
        "data/mosi_final_aligned/",
        pad_front=True,
        modalities=modalities,
        already_aligned=True,
        align_features=False,
    )

    train = MOSI(train_data, modalities=modalities, binary=False, text_is_tokens=False)
    dev = MOSI(dev_data, modalities=modalities, binary=False, text_is_tokens=False)
    test = MOSI(test_data, modalities=modalities, binary=False, text_is_tokens=False)

    ldm = PLDataModuleFromDatasets(
        train,
        val=dev,
        test=test,
        batch_size=8,
        batch_size_eval=8,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,
    )
    ldm.setup()

    feature_sizes = {"audio": 74, "visual": 35, "text": 300}

    model = TransformerLateFusionClassifier(
        feature_sizes,
        1,
        max_length=2 * max_length,
        nystrom=False,
        kernel_size=None,
        num_layers=2,
        num_heads=2,
        dropout=0.3,
        hidden_size=100,
        inner_size=200,
        prenorm=True,
        scalenorm=True,
    )

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    criterion = nn.MSELoss()

    lm = MultimodalTransformerClassificationPLModule(
        model,
        optimizer,
        criterion,
        # metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
    )

    trainer = make_trainer(
        EXPERIMENT_NAME, max_epochs=100, gpus=1, save_top_k=1, gradient_clip_val=1.0
    )
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())
