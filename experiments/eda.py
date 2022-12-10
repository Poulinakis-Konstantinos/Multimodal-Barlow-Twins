import os
import sys
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
from loguru import logger
from slp.config.config_parser import make_cli_parser, parse_config
from slp.plbind.dm import PLDataModuleFromDatasets
from slp.plbind.helpers import FromLogits
from slp.plbind.module import RnnPLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.util.log import configure_logging
from slp.util.system import is_file, safe_mkdirs
from torch.optim import Adam
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from utils.cmusdk import mosei
from utils.collators import MultimodalSequenceClassificationCollator
from utils.multimodal import MOSEI
from modules.classifier import RNNSymAttnFusionRNNClassifier
from modules.baseline import AudioVisualTextMaskedClassifier
from modules.ssl_modules import Multimodal_Barlow_Twins
from utils.metrics import MoseiAcc2, MoseiAcc5, MoseiAcc7, MoseiF1
from utils.mosei import get_mosei_parser

from modules.baseline import GloveEncoder, TextClassifier

if __name__ == "__main__":
    parser = get_mosei_parser()
    parser = make_cli_parser(parser, PLDataModuleFromDatasets)

    config = parse_config(parser, parser.parse_args().config)

    configure_logging(f"logs/eda")
    modalities = set(config.modalities)
    max_length = 100 # config.model.max_length
    collate_fn = MultimodalSequenceClassificationCollator(
        device="cpu", modalities=modalities
    )

    train_data, dev_data, test_data, w2v = mosei(
        "data/mosei_final_aligned/",
        modalities=modalities,
        max_length=-1,
        pad_back=config.preprocessing.pad_back,
        pad_front=config.preprocessing.pad_front,
        remove_pauses=config.preprocessing.remove_pauses,
        already_aligned=config.preprocessing.already_aligned,
        align_features=config.preprocessing.align_features,
        cache="./cache/mosei_avt_unpadded.p"
    )

    print(type(train_data), len(train_data))

    print(type(train_data[0]), train_data[0].keys())

    for i,x in enumerate(train_data,0):
        if "glove" in x:
            print(x['audio'].shape, x['visual'].shape, x['glove'].shape, len(x['text']), x['video_id'],
            x['segment_id'], x['label'] )
            x["text"] = x["glove"]
        if i == 1 : break

    for x in dev_data:
        if "glove" in x:
            x["text"] = x["glove"]

    for x in test_data:
        if "glove" in x:
            x["text"] = x["glove"]
    

    train = MOSEI(train_data, modalities=modalities, text_is_tokens=False)
    dev = MOSEI(dev_data, modalities=modalities, text_is_tokens=False)
    test = MOSEI(test_data, modalities=modalities, text_is_tokens=False)
    print(type(train))
    print(train)

    ldm = PLDataModuleFromDatasets(
        train,
        val=dev,
        test=test,
        batch_size=config.data.batch_size,
        batch_size_eval=config.data.batch_size_eval,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
        num_workers=config.data.num_workers,
    )
    print(ldm)
    ldm.setup()

    model = TextClassifier(
                    feature_sizes=300,
                    num_layers = config.model.num_layers,
                    num_classes = 1,
                    bidirectional=config.model.bidirectional,
                    merge_bi = config.model.merge_bi,
                    rnn_type = config.model.rnn_type,
                    attention = config.model.attention,
                    hidden_size = config.model.hidden_size,
                    dropout = config.model.dropout
    )

    # print(model)
    # input = torch.Tensor(train_data[0]['glove']).view(1, -1, 300)
    # input = torch.rand(32, 42, 300)
    # length = torch.randint(42, (32,1))
    # inputs = {'text' : input}
    # lengths = {'text' : length}
    # print(input.shape)
    # out = model(inputs, lengths)
    # print(out.shape)
    # print(out)
    # exit()

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )

    lr_scheduler = None

    if config.lr_schedule:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config.lr_schedule
        )

    criterion = nn.L1Loss()

    lm = RnnPLModule(
        model,
        optimizer,
        criterion,
        lr_scheduler=lr_scheduler,
        metrics={
            "acc2": MoseiAcc2(exclude_neutral=True),
            "acc2_zero": MoseiAcc2(exclude_neutral=False),
            "acc5": MoseiAcc5(),
            "acc7": MoseiAcc7(),
            "f1": MoseiF1(exclude_neutral=True),
            "f1_zero": MoseiF1(exclude_neutral=False),
            "mae": torchmetrics.MeanAbsoluteError(),
        },
    )

    trainer = make_trainer(**config.trainer) 
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    from utils.mosei import test_mosei

    results = test_mosei(lm, ldm, trainer, modalities)
    print(results)