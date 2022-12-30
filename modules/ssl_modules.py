import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from modules.baseline import AudioVisualTextEncoder, AudioVisualTextMaskedEncoder
from torchmetrics import Metric


def off_diagonal(x):
    ''' Given a 2D  square matrix returns a flattened view of the off-diagonal elements.
        E.g. [[0, 1, 2, 3],
              [4, 5, 6, 7,],
              [8, 9, 10,11],
              [12,13,14,15]] -> [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]

        Parameters:
            x (torch.tensor) : The input matrix (e.g. a cross correlation matrix).

        Returns:
            A flattened torch tensor with the off-diagonal elements
            '''
    n, m = x.shape
    # assert the matrix is indeed square
    assert n == m
    # it works, trust me
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Barlow_Twins_Loss(nn.Module):
    def __init__(self, alpha=0.005):
        ''' Computes the loss between the cross correlation matrix and the identity matrix I.
            Formally : L = Sum_i[1- (Cii)^2] + alpha*Sum_i{ Sum_j [(Cij)^2 ] }.
            See https://arxiv.org/abs/2103.03230 / https://medium.com/mlearning-ai/barlow-twins-self-supervised-learning-model-explained-python-torch-code-tutorial-e8f3688bbb6d for more info.

            Parameters:
                alpha (float) : The significance of the off-diagonal loss (redundancy reduction term) in the Barlow Twins loss. Defaults to 0.005.
            Returns :
                loss (torch.tensor): The loss value.
        '''
        super(Barlow_Twins_Loss, self).__init__()
        self.alpha = alpha

    def forward(self, cross_corr, target=1):
        '''cross_corr (2D torch.tensor): Two dimensional cross correlation matrix of 2 vectors.'''
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(cross_corr).pow_(2).sum()
        loss = on_diag + self.alpha * off_diag
        return loss


class BT_Loss_metric(Metric):
    def __init__(
        self,
        alpha=0.005
        ):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0, dtype=torch.float),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float),
                       dist_reduce_fx="sum")
        self.alpha = alpha
        self.Bt = Barlow_Twins_Loss(self.alpha)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds = self._input_format(preds)

        self.total = self.total +  1
        target = preds
        #print('Preds ', target.size(), target)

        self.correct += self.Bt(preds)
       # print('Self.correct shape :', self.correct.size(), self.correct)

    def compute(self):
       # print('Self.total ', self.total)
       # print(self.correct / self.total)
        return self.correct / self.total


class MM_Aug(object):
    def __init__(
        self,
        p: float = 0.2,
    ):

        self.p = p

    def __call__(self, tensor):
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(prob={0})'.format(self.p)


class Gaussian_noise(object):
    def __init__(
        self,
        mean,
        std
    ) -> None:
        '''A transformation that adds Gaussian noise to an input tensor.

           Args:
                mean (float): The mean of the Gaussian distribution.
                std (float) : The standard deviation of the Gaussian distribution.
         '''
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        
        # torch.randn samples for a Normal distribution with mean=0 and std=1.
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        #TODO get the device directly from config (?) or somehow pass it into the transformation 
        device = tensor.device   
        tensor.to(device)
        return tensor + torch.randn(tensor.size()).to(device)*self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Transform:
    def __init__(
            self,
            mm_aug_p1=0.2,
            mm_aug_p2=0.2,
            noise_p1=0.5,
            noise_p2=0.1,
            noise_mean1=0.0,
            noise_mean2=0.0,
            noise_std1=0.1,
            noise_std2=0.1):
        '''Applies two different sets of transformations to a single input to create two views of the same object.'''

        self.transform = transforms.Compose([
            MM_Aug(p=mm_aug_p1),
            transforms.RandomApply(
                [Gaussian_noise(mean=noise_mean1, std=noise_std1)],
                p=noise_p1
            )
        ])

        self.transform_prime = transforms.Compose([
            MM_Aug(p=mm_aug_p2),
            transforms.RandomApply(
                [Gaussian_noise(mean=noise_mean2, std=noise_std2)],
                p=noise_p2
            )
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

    def __repr__(self):
        return self.__class__.__name__ + '(transform_1={0}, transform_2={1})'.format(self.transform, self.transform_prime)


class Multimodal_Barlow_Twins(nn.Module):

    def __init__(
        self,
        feature_sizes,
        num_classes = 1,
        ssl_mode = True,
        mm_aug_probs=[0.2, 0.2],
        gauss_noise_m=[0.0, 0.0],
        gauss_noise_std=[0.1, 0.1],
        gauss_noise_p=[0.5, 0.1],
        num_layers=1,
        bidirectional=True,
        rnn_type="lstm",
        attention=True,
        hidden_size=100,
        dropout=0.1,
        projector_size=[],
        p_mmdrop=0.0,
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=False,
        masking=False,
        m3_sequential=False,
        m3_augment=False,
        **kwargs,
    ) -> None:
        ''' A multimodal Barlow Twins architecture. The architecture consists of 
            1. Augmentation module
            2. Multimodal Encoder (M unimodal Encoders + Fusion module)
            3. Barlow Twins Loss  (Cross correlation matrix should converge to the Identity matrix.)

            Args:

                feature_sizes (dict[int]) : Contains the dimensionality of the features for each modality. E.g. {'text':100, 'audio'=500}
                num_classes (int) : Number of neurons in the classification layer.
                ssl_mode (boolean) : Whether model is in trainign mode or not. In training mode it passes inputs through projector and outputs cross correlation matrix.
                                     When not in training mode it outputs predictions through the clf layer and ignores projection layer.
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
                **(kwargs)
            '''

        super(Multimodal_Barlow_Twins, self).__init__()
        self.ssl_mode = ssl_mode
        print('Self-Training :',self.ssl_mode)
        self.transformations = Transform(
            mm_aug_probs[0], mm_aug_probs[1],
            gauss_noise_p[0], gauss_noise_p[1],
            gauss_noise_m[0], gauss_noise_m[1],
            gauss_noise_std[0], gauss_noise_std[1])

        self.encoder = AudioVisualTextEncoder(
            feature_sizes,
            num_layers=num_layers,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            attention=attention,
            hidden_size=hidden_size,
            dropout=dropout,
            p_mmdrop=p_mmdrop,
            p_drop_modalities=p_drop_modalities,
            multi_modal_drop=multi_modal_drop,
            mmdrop_before_fuse=mmdrop_before_fuse,
            mmdrop_after_fuse=mmdrop_after_fuse,
            masking=masking,
            m3_sequential=m3_sequential,
            m3_augment=m3_augment,
        )

        print('Transformation sets : \n ', self.transformations)

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
        ''' Given a multimodal input, apply transformations to create multiple views,
        apply forward operation, compute the cross-correlation matrix of the embeddings (projector's output)
        and apply the Barlow Twins loss.

        Parameters:
            inputs (dict) : A dictionary {modality : values}. Values are torch.tensors'''
        # create two views of the same input. We use a different set of augmentations for each input.
        inputs1 = {}
        inputs2 = {}
        for modality in ['text', 'audio', 'visual'] :
            inputs1[modality], inputs2[modality] = self.transformations(inputs[modality])
            #inputs1, inputs2 = self.transformations(inputs)

        if self.ssl_mode :
            # fused Encoder's output
            z1 = self.projector(self.encoder(
                inputs1["text"],
                inputs1["audio"],
                inputs1["visual"],
                lengths["text"]
            ))

            z2 = self.projector(self.encoder(
                inputs2["text"],
                inputs2["audio"],
                inputs2["visual"],
                lengths["text"]
            ))

            # empirical cross-correlation matrix
            cross_corr = self.bn(z1).T @ self.bn(z2)
            return cross_corr
        # classification
        else :
            return self.clf(self.encoder(
                inputs1["text"],
                inputs1["audio"],
                inputs1["visual"],
                lengths["text"]
            ))

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs