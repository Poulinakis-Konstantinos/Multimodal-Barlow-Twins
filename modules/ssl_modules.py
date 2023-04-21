import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.baseline import AudioVisualTextEncoder, AudioVisualTextMaskedEncoder
from torchmetrics import Metric
from typing import List, Optional
from modules.ssl_transforms import Transformator
from loguru import logger

def off_diagonal(x):
    """Given a 2D  square matrix returns a flattened view of the off-diagonal elements.
    E.g. [[0, 1, 2, 3],
          [4, 5, 6, 7,],
          [8, 9, 10,11],
          [12,13,14,15]] -> [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]

    Parameters:
        x (torch.tensor) : The input matrix (e.g. a cross correlation matrix).

    Returns:
        A flattened torch tensor with the off-diagonal elements
    """
    n, m = x.shape
    # assert the matrix is indeed square
    assert n == m
    # it works, trust me
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Barlow_Twins_Loss(nn.Module):
    def __init__(self, alpha=0.005):
        """Computes the loss between the cross correlation matrix and the identity matrix I.
        Formally : L = Sum_i[1- (Cii)^2] + alpha*Sum_i{ Sum_j [(Cij)^2 ] }.
        See https://arxiv.org/abs/2103.03230 / https://medium.com/mlearning-ai/barlow-twins-self-supervised-learning-model-explained-python-torch-code-tutorial-e8f3688bbb6d for more info.

        Parameters:
            alpha (float) : The significance of the off-diagonal loss (redundancy reduction term) in the Barlow Twins loss. Defaults to 0.005.
        Returns :
            loss (torch.tensor): The loss value.
        """
        super(Barlow_Twins_Loss, self).__init__()
        self.alpha = alpha

    def forward(self, cross_corr, target=1):
        """cross_corr (2D torch.tensor): Two dimensional cross correlation matrix of 2 vectors."""
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(cross_corr).pow_(2).sum()
        loss = on_diag + self.alpha * off_diag
        return loss


class BT_Loss_metric(Metric):
    def __init__(self, alpha=0.005):
        super().__init__()
        self.add_state(
            "correct", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.alpha = alpha
        self.Bt = Barlow_Twins_Loss(self.alpha)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds = self._input_format(preds)

        self.total = self.total + 1
        target = preds
        self.correct = self.Bt(preds)

    def compute(self):
        return self.correct


class Multimodal_Barlow_Twins(nn.Module):
    def __init__(
        self,
        feature_sizes,
        num_classes=1,
        ssl_mode=True,
        transformation_order: Optional[List[str]]=['noise'],
        masking_p=[1.0, 0],
        masking_percentage = [1.0, 0.0],
        masking_mode = 'timestep',
        mmaug_p=[1.0, 0.0],
        mmaug_alpha=[0.15, 0.1, 0.2],
        mmaug_p_t=[1.0, 0.0],
        gauss_noise_p=[0.5, 0.1],
        gauss_noise_m=[0.0, 0.0],
        gauss_noise_std=[0.1, 0.1],
        num_layers=1,
        bidirectional=True,
        rnn_type="lstm",
        attention=True,
        hidden_size=100,
        dropout=0.1,
        projector_size=[],
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=False,
        masking=False,
        m3_sequential=False,
        m3_augment=False,
        **kwargs,
    ) -> None:
        """A multimodal Barlow Twins architecture. The architecture consists of
        1. Augmentation module
        2. Multimodal Encoder (M unimodal Encoders + Fusion module)
        3. Barlow Twins Loss  (Cross correlation matrix should converge to the Identity matrix.)

        Args:

            feature_sizes (dict[int]) : Contains the dimensionality of the features for each modality. E.g. {'text':100, 'audio'=500}
            num_classes (int) : Number of neurons in the classification layer.
            ssl_mode (boolean) : Whether model is in trainign mode or not. In training mode it passes inputs through projector and outputs cross correlation matrix.
                                 When not in training mode it outputs predictions through the clf layer and ignores projection layer.
            transformation_order (Optional(List)): Defines the order in which transformation are executed during ssl training. If a transformation name is not included
                                in the list then it is not executed. Transformations can be repeated. Strictly limited to ['noise', 'masking']. 
                                For example ['noise', 'masking', 'noise'] executed Gaussian_noise, then Masking and then Gaussian_noise again.
                                Defaults to ['noise'].
            num_layers (int) : The number of layers used in each unimodal encoder. See AudioVisualTextEncoder class in baseline.py for more info.
            bidirectional (bool) : Whether RNNs in unimodal encoders are bidirectional or not (where applicable).
            rnn_type (str) : The type of RNN used in unimodal encoders. One of 'lstm' or 'gru'. Defaults to 'lstm;.
            attention (bool) : Whether RNNs will use attention mechanism.
            hidden_size (int) : RNNs hidden dimension.
            dropout (float) : Dropout value in RNNs.
            projector_size (list(int)) : The dimensions of the projector after the Multimodal Encoder in the Siamese networks.
                        If [] an FFC layer of 2044 is used. Defaults to []
                        E.g. : If [2000, 1000], 3 FFC layers are applied FFC[2044], FFC[2000], FFC[1000].
            p_mmdrop (float): mask/drop probability in the fuser mechanism. M3 mechanism see mmdrop.py for more details.
            p_mod (Optional[List[float]]): Drop probabilities for each modality. Defaults to None. M3 mechanism see mmdrop.py for more details.
            multi_modal_drop (str): Hard or soft M3. Default to 'mmdrop_hard'. See mmdrop.py for more details.
            mmdrop_before_fuse (bool) : Whether to apply M3 before fusing different modalities (on the unimodal inputs). Defaults to True.
            mmdrop_after_fuse (bool) : Whether to apply M3 after fusing different modalities on unimodal+bimodal+trimodal inputs. Default to False.
                                       E.g. For inputs A,V,T apply mmdrop on A,V,T, AV, AT, VT, AVT (7 inputs)
            masking (bool): use M3 with no re-scaling. Defaults to False.
            m3_sequential (bool): per timestep modality masking. Defaults to False.
            m3_augment (function) : Augmentation applied to the inputs. Different than the one applied to create different views for Siamese Networks. Defaults to None.
                            augmentation_module (MM_Aug object): A transformation applied on each unimodal input to create two different views of the same 'object'.
                                                 Defaults to an 'mm_aug'.
            mm_aug_probs (list[float]) : Two values that define the probability of applying mm_aug to the input.
            gauss_noise_m (list[float]) : The mean values of the gaussian noises applied as transformations.
            gauss_noise_std (list[float]) : The std of the gaussian noises applied as transformations.
            gauss_noise_p  (list[float]) : The probabilities of applying gaussian noises.
            mmaug_p (list[float]) : The probabilities of applying mm_aug to the input.
            mm_aug_p_t (list[float]) : The probabilities of applying mm_aug to the input per timestep.
            **(kwargs)
        """

        super(Multimodal_Barlow_Twins, self).__init__()
        self.ssl_mode = ssl_mode
        logger.info("Self-Training :", self.ssl_mode)
        self.transformations = Transformator(
            transformation_order,
            noise_p1=gauss_noise_p[0],
            noise_p2=gauss_noise_p[1],
            noise_mean1=gauss_noise_m[0],
            noise_mean2=gauss_noise_m[1],
            noise_std1=gauss_noise_std[0],
            noise_std2=gauss_noise_std[1],
            masking_p1=masking_p[0],
            masking_p2=masking_p[1],
            masking_percentage_1 = masking_percentage[0],
            masking_percentage_2 = masking_percentage[1],
            masking_mode = masking_mode,
            mmaug_p1=mmaug_p[0],
            mmaug_p2=mmaug_p[1],
            mmaug_alpha=mmaug_alpha,
            mmaug_p1_t=mmaug_p_t[0],
            mmaug_p2_t=mmaug_p_t[1],
        )

        logger.info(f" SSL Transformations: {self.transformations}")

        self.encoder = AudioVisualTextEncoder(
            feature_sizes,
            num_layers=num_layers,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            attention=attention,
            hidden_size=hidden_size,
            dropout=dropout,
            p_mmdrop=0,
            p_drop_modalities=p_drop_modalities,
            multi_modal_drop=multi_modal_drop,
            mmdrop_before_fuse=mmdrop_before_fuse,
            mmdrop_after_fuse=mmdrop_after_fuse,
            masking=masking,
            m3_sequential=m3_sequential,
            m3_augment=m3_augment,
        )

        logger.info("Transformation sets : \n ", self.transformations)

        # projector properties. A linear layer of 2044 is always used. Subsequent layers are added on top.
        sizes = list([self.encoder.out_size, 2044]) + list(projector_size)
        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        # wrap the linear layers + batch_norm + relu in a projector attribute
        self.projector = nn.Sequential(*layers)
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.clf = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, inputs: dict, lengths):
        """Given a multimodal input, apply transformations to create multiple views,
        apply forward operation, compute the cross-correlation matrix of the embeddings (projector's output)
        and apply the Barlow Twins loss.

        Parameters:
            inputs (dict) : A dictionary {modality : values}. Values are torch.tensors"""
        # create two views of the same input. We use a different set of augmentations for each input.
        inputs1 = {}
        inputs2 = {}
        if self.ssl_mode:
            inputs1, inputs2 = self.transformations(
                                                    inputs["text"],
                                                    inputs["audio"],
                                                    inputs["visual"]
                                                    )
            # fused Encoder's output
            z1 = self.projector(
                self.encoder(
                    inputs1[0],
                    inputs1[1],
                    inputs1[2],
                    lengths["text"],
                )
            )

            z2 = self.projector(
                self.encoder(
                    inputs2[0],
                    inputs2[1],
                    inputs2[2],
                    lengths["text"],
                )
            )

            # empirical cross-correlation matrix
            cross_corr = self.bn(z1).T @ self.bn(z2)
            return cross_corr
        # classification
        else:
            return self.clf(
                self.encoder(
                    inputs["text"],
                    inputs["audio"],
                    inputs["visual"],
                    lengths["text"],
                )
            )


# class mySequential(nn.Sequential):
# ''' Used as a workaround so that nn.Sequential accepts multiple inputs'''
#     def forward(self, *inputs):
#         for module in self._modules.values():
#             if type(inputs) == tuple:
#                 inputs = module(*inputs)
#             else:
#                 inputs = module(inputs)
#         return inputs
