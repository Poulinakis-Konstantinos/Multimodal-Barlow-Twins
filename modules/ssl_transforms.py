import torch
import torchvision.transforms as transforms
import torch.distributions as D
from torch import nn
from typing import List, Optional
import random
from loguru import logger
from modules.mmaug import MMAug

# class MM_Aug(object):
#     def __init__(
#         self,
#         p: float = 0.2,
#     ):

#         self.p = p

#     def __call__(self, tensor):
#         return tensor

#     def __repr__(self):
#         return self.__class__.__name__ + "(prob={0})".format(self.p)



class Masking(nn.Module):
    def __init__(self, p_mask, mode='timestep') -> None:
        '''Masks input time series of [Batch, SeqLen, Feature_Dim] shape. Masking is applied as zero values.
           Whole time steps are zeroed out. The mask is applied on the SeqLen dimension and p_mask parameters 
           controls the percentage of time steps masked. For example p_mask=0.01 (1%) will mask only a single 
           timestep on each batch of input x:(32, 100, 200) e.g. x[_][5][:] =0 .
           
           mode (str) : Whether mask will be applied to L dimension or F dimensions. One of ('timestep', 'feature').'''
        super().__init__()
        # denotes the percentage of input to be masked
        self.p_mask = p_mask
        # translate p_mask to probability for transformation to take place 
        self.which_timesteps = D.Bernoulli( probs=(1-p_mask))
        self.mode = mode
    
    def forward(self, mods):
        # mods[k] : (Batch, Len, Feature Dim)
        mods = list(mods)
        for i in range(len(mods)):
            device = mods[i].device
            if self.mode == 'timestep':
                bsz, seqlen = mods[i].shape[0], mods[i].shape[1]
                # unsqueeze at feature dimensions to allow multiplication with input
                self.time_mask = self.which_timesteps.sample((bsz, seqlen)).unsqueeze(2)
            elif self.mode == 'feature':
                bsz, seqlen, feats = mods[i].shape[0], mods[i].shape[1], mods[i].shape[2]
                self.time_mask = self.which_timesteps.sample((bsz, seqlen, feats))

            #logger.debug(f' Mask shape {self.time_mask.shape}')
            #logger.critical('CHANGE : {} {}'.format( mods[i], self.time_mask.to(device) * mods[i]))
            mods[i] = self.time_mask.to(device) * mods[i]
            
        return mods

    def __repr__(self):
        return self.__class__.__name__ + "(p_mask={0}, mode={1})".format(self.p_mask, self.mode)


class Multimodal_masking_augmentator:
    def __init__(self) -> None:
        raise NotImplementedError


class Gaussian_noise(object):
    def __init__(self, mean, std) -> None:
        """A transformation that adds Gaussian noise to an input tensor.

        Args:
             mean (float): The mean of the Gaussian distribution.
             std (float) : The standard deviation of the Gaussian distribution.
        """
        self.mean = mean
        self.std = std

    def __call__(self, mods):

        mods = list(mods)
        for i in range(len(mods)):
            #logger.debug('Init shape  {}'.format(m.shape))
            if not isinstance(mods[i], torch.Tensor):
                mods[i] = torch.tensor(mods[i])
            device = mods[i].device
            #logger.critical('CHANGE : {} {}'.format( mods[i], mods[i] + torch.randn(mods[i].size()).to(device) * self.std + self.mean))
            mods[i] = mods[i] + torch.randn(mods[i].size()).to(device) * self.std + self.mean
            #logger.debug('After shape  {}'.format(m.shape))
            
        return mods

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Transformator:
    def __init__(
        self,
        transformation_order: List[str] = ["noise"],
        noise_p1=0.7,
        noise_p2=0.2,
        noise_mean1=0.0,
        noise_mean2=0.0,
        noise_std1=0.1,
        noise_std2=0.1,
        masking_p1=0.5,
        masking_p2=0.0,
        masking_percentage_1=0.5,
        masking_percentage_2=0.5,
        masking_mode = 'timestep',
        mmaug_p1= 1.0,
        mmaug_p2= 0.0,
        mmaug_p1_t= 1.0,
        mmaug_p2_t= 0.0,

        p_mod1: Optional[List[float]] = None,
        p_mod2: Optional[List[float]] = None,
        m3_sequential: bool = True,
    ) -> None:
        """ Wrapper class for other augmentation functions. It combines, controls the probability of application
          and calls other transformation functions.  

        Args:
            transformation_order (List[str], optional): Controls which transformations are applied and the order of application.
                                                         Defaults to ["noise"].
            noise_p1 (float, optional): _description_. Defaults to 0.7.
            noise_p2 (float, optional): _description_. Defaults to 0.2.
            noise_mean1 (float, optional): _description_. Defaults to 0.0.
            noise_mean2 (float, optional): _description_. Defaults to 0.0.
            noise_std1 (float, optional): _description_. Defaults to 0.1.
            noise_std2 (float, optional): _description_. Defaults to 0.1.
            masking_p1 (float, optional): The probability that masking transformation will be applied to (batch) input view_1. Defaults to 0.5.
            masking_p2 (float, optional): The probability that masking transformation will be applied to (batch) input view_2. Defaults to 0.0.
            masking_percentage_1 (float, optional): Controls the percentage of time steps masked on input view_1. Defaults to 0.5.
            masking_percentage_2 (float, optional):  Controls the percentage of time steps masked on input view_2. Defaults to 0.5.
            mmaug_p1 (float, optional): The probability that mmaugment transformation will be applied to (batch) input view_1. Defaults to 1.0.
            mmaug_p2 (float, optional): The probability that mmaugment transformation will be applied to (batch) input view_2. Defaults to 0.0.
            mmag_p1_t (float, optional): Resample probabilities for each modality across timesteps. Default=1's which means that we resample at every timestep Defaults to 1.0.
            mmaug_p2_t (float, optional): Resample probabilities for each modality across timesteps. Default=0's which means that we resample at every timestep Defaults to 1.0.
            p_mod1 (Optional[List[float]], optional): _description_. Defaults to None.
            p_mod2 (Optional[List[float]], optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to "hard".
            m3_sequential (bool, optional): _description_. Defaults to True.
        """

        assert all(
            isinstance(x, str) and x in ["noise", "masking", "mmaug"]
            for x in transformation_order
        ), "Allowed modes for transformation_order are ['noise' | 'masking' | 'mmaug']"

        # Define transformation objects in a dict to use afterwards.
        self.instructions = {
            "noise": [
                transforms.RandomApply(
                    [Gaussian_noise(noise_mean1, noise_std1)], noise_p1
                ),
                transforms.RandomApply(
                    [Gaussian_noise(noise_mean2, noise_std2)], noise_p2
                ),
            ],
            "masking": [
            transforms.RandomApply(
                [Masking(masking_percentage_1, masking_mode)], masking_p1
                ),
            transforms.RandomApply(
                [Masking(masking_percentage_2, masking_mode)], masking_p2 
                )
            ],
            "mmaug": [
            transforms.RandomApply(
                [MMAug(p_t_mod=[mmaug_p1_t])], mmaug_p1
                ),
            transforms.RandomApply(
                [MMAug(p_t_mod=[mmaug_p2_t])], mmaug_p2
                )
            ],
        }

        # Define the first transformator (using [0] key)
        self.transform = transforms.Compose(
            [self.instructions[order][0] for order in transformation_order]
        )
        # Define the second transformator`prime` (using [1] key)
        self.transform_prime = transforms.Compose(
            [self.instructions[order][1] for order in transformation_order]
        )

    def __call__(self, *mods):
        # B x L x Feats. Modality shapes for MOSEI:
        # torch.Size([170, 306, 300]), torch.Size([170, 306, 74]), torch.Size([170, 306, 35])
        #mods = list(mods)
        # Y1, Y2 have the same dimensions as mods
        y1 = self.transform(mods)
        y2 = self.transform_prime(mods)

        return y1, y2

    def __repr__(self):
        return (
            self.__class__.__name__
            + "Transformation Order: {0},  (transform_1={1}, transform_2={2})".format(
                self.instructions, self.transform, self.transform_prime
            )
        )







class Transform:
    def __init__(
        self,
        mm_aug_p1=0.2,
        mm_aug_p2=0.2,
        noise_p1=0.7,
        noise_p2=0.2,
        noise_mean1=0.0,
        noise_mean2=0.0,
        noise_std1=0.1,
        noise_std2=0.1,
    ):
        """--Important Note: The Transformator class above, provides more flexibility. This was the initial implementation which will be deprecated.
        Applies two different sets of transformations to a single input to create two views of the same object."""

        self.transform = transforms.Compose(
            [
                MM_Aug(p=mm_aug_p1),
                transforms.RandomApply(
                    [Gaussian_noise(mean=noise_mean1, std=noise_std1)], p=noise_p1
                ),
            ]
        )

        self.transform_prime = transforms.Compose(
            [
                MM_Aug(p=mm_aug_p2),
                transforms.RandomApply(
                    [Gaussian_noise(mean=noise_mean2, std=noise_std2)], p=noise_p2
                ),
            ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

    def __repr__(self):
        return self.__class__.__name__ + "(transform_1={0}, transform_2={1})".format(
            self.transform, self.transform_prime
        )
# class Guassian_noise_augmentator(object):
#     def __init__(
#         self,
#         noise_p1=0.7,
#         noise_p2=0.2,
#         noise_mean1=0.0,
#         noise_mean2=0.0,
#         noise_std1=0.1,
#         noise_std2=0.1,
#     ) -> None:
#         """Holds 2 torchvision.transforms.RandomApply transformations with the Gaussian_noise transformation.
#         Used in self-supervised training to create two distorted views of the same object [transform, transform_prime]

#         Args:
#             noise_p1 (float, optional): Noise probability for transform. Defaults to 0.7.
#             noise_p2 (float, optional): Noise probability for transform_prime. Defaults to 0.2.
#             noise_mean1 (float, optional): Noise distribution mean value for transform. Defaults to 0.0.
#             noise_mean2 (float, optional): Noise distribution mean value for transform_prime. Defaults to 0.0.
#             noise_std1 (float, optional): Noise distribution std for transform. Defaults to 0.1.
#             noise_std2 (float, optional): Noise distribution std for transform_prime. Defaults to 0.1.
#         """

#         self.noise_mean1 = noise_mean1
#         self.noise_mean2 = noise_mean2
#         self.noise_std1 = noise_std1
#         self.noise_std2 = noise_std2
#         self.noise_p1 = noise_p1
#         self.noise_p2 = noise_p2

#         self.transform = transforms.RandomApply(
#             [Gaussian_noise(mean=noise_mean1, std=noise_std1)], p=noise_p1
#         )
#         self.transform_prime = transforms.RandomApply(
#             [Gaussian_noise(mean=noise_mean2, std=noise_std2)], p=noise_p2
#         )

#     def __repr__(self):
#         return self.__class__.__name__ + "T1: (mean={0}, std={1})".format(
#             self.noise_mean1, self.noise_std1
#         )
