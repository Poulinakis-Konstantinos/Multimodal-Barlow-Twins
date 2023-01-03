import os 
import sys

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
import wandb
from utils.mosei import get_mosei_parser
from utils.metrics import MoseiAcc2, MoseiAcc5, MoseiAcc7, MoseiF1
from modules.ssl_modules import Multimodal_Barlow_Twins, Barlow_Twins_Loss, BT_Loss_metric
from modules.baseline import AudioVisualTextMaskedClassifier
from modules.classifier import RNNSymAttnFusionRNNClassifier
from utils.multimodal import MOSEI
from utils.collators import MultimodalSequenceClassificationCollator
from utils.cmusdk import mosei
from utils.mosei import test_mosei

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
from torch import load


if __name__ == "__main__":
    parser = get_mosei_parser()
    parser = make_cli_parser(parser, PLDataModuleFromDatasets)

    config = parse_config(parser, parser.parse_args().config)

    # if config.trainer.experiment_name != "Multimodal_Barlow_Twins":
    #     config.trainer.experiment_name = "Multimodal_Barlow_Twins"

    configure_logging(f"logs/{config.trainer.experiment_name}")
    modalities = set(config.modalities)
    max_length = config.model.max_length
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

    for x in train_data:
        if "glove" in x:
            x["text"] = x["glove"]

    for x in dev_data:
        if "glove" in x:
            x["text"] = x["glove"]

    for x in test_data:
        if "glove" in x:
            x["text"] = x["glove"]

    train = MOSEI(train_data, modalities=modalities, text_is_tokens=False)
    dev = MOSEI(dev_data, modalities=modalities, text_is_tokens=False)
    test = MOSEI(test_data, modalities=modalities, text_is_tokens=False)

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
    ldm.setup()
    feature_sizes = config.model.feature_sizes

    model = Multimodal_Barlow_Twins(
        feature_sizes,
        num_classes = 1,
        ssl_mode = True,
        num_layers=config.model.num_layers,
        projector_size=config.barlow_twins.projector_size,
        mm_aug_probs=config.transformations.mm_aug_p,
        gauss_noise_p=config.transformations.gauss_noise_p,
        batch_first=config.model.batch_first,
        bidirectional=config.model.bidirectional,
        packed_sequence=config.model.packed_sequence,
        merge_bi=config.model.merge_bi,
        rnn_type=config.model.rnn_type,
        attention=config.model.attention,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        max_length=config.model.max_length,
        dropout=config.model.dropout,
        nystrom=False,
        multi_modal_drop=config.model.multi_modal_drop,
        p_mmdrop=config.model.p_mmdrop,
        mmdrop_before_fuse=config.model.mmdrop_before_fuse,
        mmdrop_after_fuse=config.model.mmdrop_after_fuse,
        p_drop_modalities=config.model.p_drop_modalities,
        m3_augment=config.model.use_m3_augment,
        p_aug=config.model.p_augment,
    )

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

    criterion = Barlow_Twins_Loss(alpha=config.barlow_twins.alpha)  # nn.L1Loss()

    lm = RnnPLModule(
        model,
        optimizer,
        criterion,
        lr_scheduler=lr_scheduler,
        metrics={
            "BT_loss" : BT_Loss_metric(alpha=config.barlow_twins.alpha)
        },
    )

    trainer = make_trainer(**config.trainer)
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    # # cut projector head and add clf on top
    # model.projector, model.bn = nn.Identity(), nn.Identity()
    #num_classes = 1
    #clf = nn.Linear(model.encoder.out_size, num_classes)
    # model = nn.Sequential(model, clf)
    # print(model)
    model_pred = Multimodal_Barlow_Twins(
        feature_sizes,
        num_classes = 1,
        ssl_mode = False,
        num_layers=config.model.num_layers,
        projector_size=config.barlow_twins.projector_size,
        mm_aug_probs=config.transformations.mm_aug_p,
        gauss_noise_p=config.transformations.gauss_noise_p,
        batch_first=config.model.batch_first,
        bidirectional=config.model.bidirectional,
        packed_sequence=config.model.packed_sequence,
        merge_bi=config.model.merge_bi,
        rnn_type=config.model.rnn_type,
        attention=config.model.attention,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        max_length=config.model.max_length,
        dropout=config.model.dropout,
        nystrom=False,
        multi_modal_drop=config.model.multi_modal_drop,
        p_mmdrop=config.model.p_mmdrop,
        mmdrop_before_fuse=config.model.mmdrop_before_fuse,
        mmdrop_after_fuse=config.model.mmdrop_after_fuse,
        p_drop_modalities=config.model.p_drop_modalities,
        m3_augment=config.model.use_m3_augment,
        p_aug=config.model.p_augment,
    )

    lm_clf = RnnPLModule(
        model_pred,
        optimizer,
        nn.L1Loss(),
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
    ckpt_path = trainer.checkpoint_callback.best_model_path
    ckpt = load(ckpt_path, map_location="cpu")
    lm_clf.load_state_dict(ckpt["state_dict"])
    lm_clf = lm_clf.cuda()
    
    if config.eval.freeze_grads:
        model.requires_grad_(False)
        model.clf.requires_grad_(True)

    #print(lm_clf.model)
    # lm_clf.model.projector, lm_clf.model.bn = nn.Identity(), nn.Identity()
    # lm_clf.model = mySequential(lm_clf.model, clf)
    # print(lm_clf.model)

    results = test_mosei(lm_clf, ldm, trainer, modalities, load_best=False)
    wandb.log({'mae': results['mae'], 'corr': results['corr'], 'acc_7':results['acc_7'], 'acc_5': results['acc_5'],
              'f1_pos':results['f1_pos'], 'bin_acc_pos': results['bin_acc_pos'],
              'f1_neg': results['f1_neg'], 'bin_acc_neg' : results['bin_acc_neg'],
              'f1':results['f1'], 'bin_acc' : results['bin_acc']})
    wandb.save('/home/poulinakis/Multimodal-Barlow-Twins/configs/my-config.yml')

    print(' RESULTS ')
    print(results)

    exit()

    # uncomment the following lines if you want result logging for multiple runs

    # import csv
    # import os

    # csv_folder_path = os.path.join(
    #     config.trainer.experiments_folder, config.trainer.experiment_name, "results_csv"
    # )

    # csv_name = os.path.join(csv_folder_path, "results.csv")
    # fieldnames = list(results.keys())

    # if is_file(csv_name):
    #     # folder already exits and so does the .csv
    #     csv_exists = True
    #     print(f"csv already exists")
    # else:
    #     csv_exists = False
    #     safe_mkdirs(csv_folder_path)

    # with open(csv_name, "a") as csv_file:
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    #     if not csv_exists:
    #         writer.writeheader()
    #     writer.writerow(results)
