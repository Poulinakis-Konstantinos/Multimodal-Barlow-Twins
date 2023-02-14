import torch
import torchvision.transforms as transforms
from typing import List, Optional


class MM_Aug(object):
    def __init__(
        self,
        p: float = 0.2,
    ):

        self.p = p

    def __call__(self, tensor):
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(prob={0})".format(self.p)


class Multimodal_masking(object):
    def __init__(self, masking_p1, n_modalities_1, p_mod1, mode, m3_sequential) -> None:
        self.pmod1 = p_mod1

    def __call__(self, tensor):
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(prob={0})".format(self.p)


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

    def __call__(self, tensor):

        # torch.randn samples for a Normal distribution with mean=0 and std=1.
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        # TODO get the device directly from config (?) or somehow pass it into the transformation
        device = tensor.device
        tensor.to(device)
        return tensor + torch.randn(tensor.size()).to(device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
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
        masking_p1=0.3,
        masking_p2=0.3,
        n_modalities_1=3,
        n_modalities_2=3,
        p_mod1: Optional[List[float]] = None,
        p_mod2: Optional[List[float]] = None,
        mode: str = "hard",
        m3_sequential: bool = True,
    ) -> None:

        assert all(
            isinstance(x, str) and x in ["noise", "masking"]
            for x in transformation_order
        ), "Allowed mode for transformation_order ['noise' | 'masking']"
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
                Multimodal_masking(
                    masking_p1, n_modalities_1, p_mod1, mode, m3_sequential
                ),
                Multimodal_masking(
                    masking_p2, n_modalities_2, p_mod2, mode, m3_sequential
                ),
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

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
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
