import os
import sys
import random
from math import floor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
import wandb
from utils.mosei import get_mosei_parser
from utils.metrics import MoseiAcc2, MoseiAcc5, MoseiAcc7, MoseiF1
from modules.ssl_modules import (
    Multimodal_Barlow_Twins,
    Barlow_Twins_Loss,
    BT_Loss_metric,
)
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

# import slp
# logger.debug(f'{slp.__file__}')
from slp.util.log import configure_logging
from slp.util.system import is_file, safe_mkdirs
from torch.optim import Adam
import torch.optim as optim
from torch import load
from torch import device, cuda

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

    ## Dataset Initialization-Processing
    train_data, dev_data, test_data, w2v = mosei(
        "data/mosei_final_aligned/",
        modalities=modalities,
        max_length=-1,
        pad_back=config.preprocessing.pad_back,
        pad_front=config.preprocessing.pad_front,
        remove_pauses=config.preprocessing.remove_pauses,
        already_aligned=config.preprocessing.already_aligned,
        align_features=config.preprocessing.align_features,
        cache="./cache/mosei_avt_unpadded.p",
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

    ssl_percent = config.data_ssl.data_percentage
    ssl_train = random.sample(train_data, int(floor(len(train_data)*ssl_percent))) if ssl_percent != -1 else train_data
    ssl_dev = random.sample(dev_data,  int(floor(len(dev_data)*ssl_percent))) if ssl_percent != -1 else dev_data

    logger.debug(f'Whole train dataset shape = {len(train_data)}, type: {type(train_data)},  Dev data shape= {len(dev_data)},  Test data sha[e = {len(test_data)}')
    logger.debug(f'SSL Pre-training Train-set:  samples used = {len(ssl_train)}, percentage={ssl_percent*100}%')
    logger.debug(f'SSL Pre-training Dev-set:  samples used = {len(ssl_dev)}, percentage={ssl_percent*100}%')

    train = MOSEI(ssl_train, modalities=modalities, text_is_tokens=False)
    dev = MOSEI(ssl_dev, modalities=modalities, text_is_tokens=False)
    test = MOSEI(test_data, modalities=modalities, text_is_tokens=False)

    # data into pytorch-lighting module
    ldm = PLDataModuleFromDatasets(
        train,
        val=dev,
        test=test,
        batch_size=config.data_ssl.batch_size,
        batch_size_eval=config.data_ssl.batch_size_eval,
        collate_fn=collate_fn,
        pin_memory=config.data_ssl.pin_memory,
        num_workers=config.data_ssl.num_workers,
    )
    ldm.setup()
    feature_sizes = config.model.feature_sizes

    ## Define Self-Supervised model
    ssl_model = Multimodal_Barlow_Twins(
        feature_sizes,
        num_classes=1,  # It's a regression problem
        ssl_mode=True,  # For self-supervised training
        transformation_order=config.transformations.order, # order in which the ssl transformations are applied.
        num_layers=config.model.num_layers,
        projector_size=config.barlow_twins.projector_size,
        # mmaug related parameters
        mmaug_p=config.transformations.mmaug_p,
        mmaug_p_t=config.transformations.mmaug_p_t,
        # masking related parameters
        masking_p=config.transformations.masking_p,
        masking_percentage = config.transformations.mask_percentage,
        masking_mode= config.transformations.masking_mode,
        # gaussian noise related parameters
        gauss_noise_p=config.transformations.gauss_noise_p,
        gauss_noise_m=config.transformations.gauss_noise_mean,
        gauss_noise_std=config.transformations.gauss_noise_std,

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
        mmdrop_before_fuse=config.model.mmdrop_before_fuse,
        mmdrop_after_fuse=config.model.mmdrop_after_fuse,
        p_drop_modalities=config.model.p_drop_modalities,
        m3_augment=config.model.use_m3_augment,
        p_aug=config.model.p_augment,
    )

    # define optimizer and lr configs
    optimizer = Adam(
        [p for p in ssl_model.parameters() if p.requires_grad],
        lr=config.optim.lr,
        weight_decay=config.ssl_optimization.optim.weight_decay,
    )
    lr_scheduler = None
    if config.ssl_optimization.lr_schedule:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config.ssl_optimization.lr_schedule
        )
    # Self-Supervised Criterion for Barlow Twins model
    criterion = Barlow_Twins_Loss(alpha=config.barlow_twins.alpha)
    # Torch Model to Pytorch Lightning Module
    lm = RnnPLModule(
        ssl_model,
        optimizer,
        criterion,
        lr_scheduler=lr_scheduler,
        # metrics={
        #     "BT_loss" : BT_Loss_metric(alpha=config.barlow_twins.alpha)
        # },
    )
    # initialize trainer object based on config
    trainer = make_trainer(**config.trainer_ssl)
    logger.debug(f"Trainer callbacks {trainer.callbacks}")
    logger.debug("Trainer callbacks metrics {trainer.callback_metrics}")
    watch_model(trainer, ssl_model)
    wandb.run.name = config.run_name
    logger.debug(f"INIT DEVICE {next(ssl_model.parameters()).device}")
    # log important params (common between ssl-supervised)
    wandb.log(
        {   
            "transformations": list(config.transformations.order),
            "epochs_ssl": config.trainer_ssl.max_epochs,
            "batch_size_ssl": config.data_ssl.batch_size,
            "num_layers": config.model.num_layers,
            "hidden_size": config.model.hidden_size,
            "bi_lstm": config.model.bidirectional,
            "proj_size": list(config.barlow_twins.projector_size)[-1],
            "weight_decay_ssl": config.ssl_optimization.optim.weight_decay,  # the only not shared
            "lr_ssl": config.ssl_optimization.optim.lr,  # the only not shared
            "dropout": config.model.dropout,
            "p_noise1": list(config.transformations.gauss_noise_p)[0],
            "p_noise2": list(config.transformations.gauss_noise_p)[1],
            "noise_std1": list(config.transformations.gauss_noise_std)[0],
            "noise_std2": list(config.transformations.gauss_noise_std)[1],
            "noise_mean1": list(config.transformations.gauss_noise_mean)[0],
            "noise_mean2": list(config.transformations.gauss_noise_mean)[1],
            "p_masking1": list(config.transformations.masking_p)[0],
            "p_masking2": list(config.transformations.masking_p)[1],
            "masking_percentage1": list(config.transformations.mask_percentage)[0],
            "masking_percentage2": list(config.transformations.mask_percentage)[1],
            "masking_mode": config.transformations.masking_mode,
            "p_mmaug1": list(config.transformations.mmaug_p)[0],
            "p_mmaug2": list(config.transformations.mmaug_p)[1],
            "p_mmaug_t1": list(config.transformations.mmaug_p_t)[0],
            "p_mmaug_t2": list(config.transformations.mmaug_p_t)[1],
            "data_ssl": config.data_ssl.data_percentage,
        }
    )
    # Train model
    trainer.fit(lm, datamodule=ldm)

    ################   Supervised Fine-Tuning of self-supervised model   ##########################
    # Define an identical model with self-supervision mode off
    model = Multimodal_Barlow_Twins(
        feature_sizes,
        num_classes=1,  # Regression problem
        ssl_mode=False,  # No self-supervision mode -> Supervised fine tuning
        num_layers=config.model.num_layers,
        projector_size=config.barlow_twins.projector_size,
        # mmaug related parameters
        mmaug_p=config.transformations.mmaug_p,
        mmaug_p_t=config.transformations.mmaug_p_t,
        # masking related parameters
        masking_p= config.transformations.masking_p,
        masking_percentage=config.transformations.mask_percentage,
        # gauss noise related parameters
        gauss_noise_p=config.transformations.gauss_noise_p,
        gauss_noise_m=config.transformations.gauss_noise_mean,
        gauss_noise_std=config.transformations.gauss_noise_std,
        
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

    # Load best weights from self-supervised training into the new model
    ckpt_path = trainer.checkpoint_callback.best_model_path
    ckpt = load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    logger.debug(f"MODEL PRED DEVICE {next(model.parameters()).device}")
    # If defined in config freeze ssl network weights and only fine tune the linear layer.
    if config.tune.freeze_grads:
        model.requires_grad_(False)
    model.clf.requires_grad_(True)
    # define a new optimizer and lr configs
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.optimization.optim.lr,
        # weight_decay=config.optim.weight_decay,
    )
    lr_scheduler = None
    if config.optimization.lr_schedule:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config.optimization.lr_schedule
        )
    # Torch model to Pytorch Lighting module
    lm_clf = RnnPLModule(
        model,
        optimizer,
        nn.L1Loss(),  # We now use L1 Loss
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
    logger.debug(f"MODEL PRED LM_CLF DEVICE {next(lm_clf.model.parameters()).device}")

    # Use a subset of the training data for fine tuning
    logger.info(f"Initiating Supervised fine-tuning ...")
    percent = config.data.data_percentage  # e.g. 0.1
    logger.debug(f'Percentage {percent}')
    logger.debug(f'Sampled data {int(floor(len(train_data)*percent))}')
    logger.debug(f'train data length {len(train_data)}')
    tuning_tr = random.sample(train_data, int(floor(len(train_data)*percent))) if percent != -1 else train_data #int(floor(len(train_data) * percent))) if percent != -1 else train_data
    tuning_dev = random.sample(dev_data, int(floor(len(dev_data)*percent))) if percent != -1 else dev_data
    logger.debug(f'Whole train dataset shape = {len(train_data)}, type: {type(train_data)},  Dev data shape= {len(dev_data)},  Test data shape = {len(test_data)}')
    logger.debug(f'Fine tuning Train-set:  samples used = {len(tuning_tr)}, percentage={percent*100}%')
    logger.debug(f'Fine-tuning Dev-set:  samples used = {len(tuning_dev)}, percentage={percent*100}%')

    train = MOSEI(tuning_tr, modalities=modalities, text_is_tokens=False)
    dev = MOSEI(tuning_dev, modalities=modalities, text_is_tokens=False)
    #dev = MOSEI(dev_data, modalities=modalities, text_is_tokens=False)


    # Convert dataset to Pytorch Lightning module
    ldm = PLDataModuleFromDatasets(
        train,
        val=dev,
        test=test,  # test set remains the same
        batch_size=config.data.batch_size,
        batch_size_eval=config.data.batch_size_eval,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
        num_workers=config.data.num_workers,
    )
    ldm.setup()

    # New trainer for the new fine tuned model
    trainer = make_trainer(**config.trainer)
    watch_model(trainer, model)
    ################  Model  Zero-Shot Evaluation  ########################
    results = test_mosei(lm_clf, ldm, trainer, modalities, load_best=False)
    # Log results in wandb online platform
    wandb.log(
        {   
            "batch_size": config.data.batch_size,
            "data_percent": config.data.data_percentage,
            "zs_mae": results["mae"],
            "zs_corr": results["corr"],
            "zs_acc_7": results["acc_7"],
            "zs_acc_5": results["acc_5"],
            "zs_f1_pos": results["f1_pos"],
            "zs_bin_acc_pos": results["bin_acc_pos"],
            "zs_f1_neg": results["f1_neg"],
            "zs_bin_acc_neg": results["bin_acc_neg"],
            "zs_f1": results["f1"],
            "zs_bin_acc": results["bin_acc"],
            "weight_decay": config.optimization.optim.weight_decay,
            "lr": config.optimization.optim.lr,
            "freeze_grads": config.tune.freeze_grads,
            "alpha": config.barlow_twins.alpha,
        }
    )
    logger.info("ZERO-SHOT RESULTS")
    logger.info(f"{results}")

    ##### Now fine tune model  #####
    trainer.fit(lm_clf, datamodule=ldm)

    ################  Fine Tuned Model  Evaluation ########################
    results = test_mosei(lm_clf, ldm, trainer, modalities, load_best=True)
    # Log results in wandb online platform
    wandb.log(
        {
            "mae": results["mae"],
            "corr": results["corr"],
            "acc_7": results["acc_7"],
            "acc_5": results["acc_5"],
            "f1_pos": results["f1_pos"],
            "bin_acc_pos": results["bin_acc_pos"],
            "f1_neg": results["f1_neg"],
            "bin_acc_neg": results["bin_acc_neg"],
            "f1": results["f1"],
            "bin_acc": results["bin_acc"],
        }
    )
    # Log the config file in wandb online platform
    wandb.save("/home/poulinakis/Multimodal-Barlow-Twins/configs/my-config.yml")
    logger.info("FINE TUNED MODEL RESULTS")
    logger.info(f"{results}")

    exit()
