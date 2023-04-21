'''
This is the implementation used in the ICASSP submission of MMAug.
Some hyperparameters which are redundant are set to their default value
in this implementation. Moreover, the padding is at the end (suffix) so the
implementation takes into account this fomat. The overall pipeline is
identical to not affect any results.
'''
import random
from typing import List, Optional, Dict

import torch
import numpy as np
import torch.nn as nn
import torch.distributions.beta as beta
import torch.distributions as D


class MMAug(nn.Module):
    def __init__(
        self,
        p_t_mod: (List[float]) = [1.0], #[1.0, 1.0, 1.0],
        mask_dim: bool = True,
        alpha: (List[float]) = [0.15, 0.10, 0.20],
        permute: str = "time",
        replacement: bool = False,
        time_window: Optional[str] = "uniform",
        window_len: Optional[int] = -1,
        maxlen: Optional[int] = 50,
        mixup: Optional[bool] = False,
        discard_zero_pad: Optional[bool] = True,
        constant_val: Optional[float] = -1.0,
        use_beta: Optional[bool] = True,
        single_pick: Optional[bool] = False,
    ):
        """
        Modified implementation of FeRe in which only one modality at a time is being picked
        Args:
            p_t_mod (List[float]): Resample probabilities for each modality across timesteps.
                default=1's which means that we resample at every timestep
            mask_dim (bool): copy the same "dimension" across all the timesteps of a given 
                modality. Induces time-invariance. Default: True
            alpha (List[float]): used in sampling uniform distribution for each modality.
                Encodes the amount of features to be resampled. In order to resample the same
                distribution one should give a list of multiple variables as input in [0,1).
            permute (str): which tensor "dimension" is going to be permuted to produce the tensor
                that is going to be pasted in the original one. Choices are "time" and "batch".
                - "time": induces the same label and simply shuffles the time-order of the tensor 
                iteself
                - "batch": respects the time order but mixes samples from different labels
            replacement (bool): determines whether to use replacement when shuffling 
                the sequence itself. True means that some features may be copied more than once
                whereas False denotes that once copied you cannot be copied again.
            time_window (str): "uniform" and "hann" available choices for neighborhood sampling
            window_len (int): number of total neighbors, assumed even, i.e. symmetric
            maxlen (int): must be given as input and refers to the maximum sequence length
                In this implementation we also assume that the sequences are already zero-padded
            mixup (bool): handles whether to use cutmix-like (FeRe) or mixup-like, default:False,
                Mixup might work better at intermed represesntations.
            discard_zero_pad (bool): when true resamples only from the real length sequence
                and ignores the zero padded part at the beginning
            constant_val (float): A constant value, i.e. distribution which is used to sample from
                Default: -1.0, when negative it skips this constant value. Suggested values are
                0.0 and 1.0. When 0.5, denotes random noise in with zero mean and 0.5 std
            use_beta (bool): when true samples from a Betta distribution for each one
                of the involved modalities, otherwise from a uniform
            single_pick (bool): when true samples from a Betta distribution for each one
                of the involved modalities, otherwise from a uniform
        """
        super(MMAug, self).__init__()
        self.p_t_mod = p_t_mod
        self.n_modalities = len(self.p_t_mod)
        self.mask_dim = mask_dim
        self.alpha = alpha
        self.permute = permute
        self.replacement = replacement
        self.time_window = time_window
        self.window_len = window_len
        self.maxlen = maxlen
        self.mixup = mixup
        self.discard_zero_pad = discard_zero_pad
        self.constant_val = constant_val
        self.use_beta = use_beta

        # neighbourhood sampling
        if self.window_len > 0:
            self.step_weights = \
                self.get_timestep_weights()
            self.relative_idx = \
                torch.arange(-self.window_len//2, self.window_len//2 + 1, step=1)
            self.absolute_idx = \
                torch.arange(0, self.maxlen)
                
        # A normal distribution was possibly used in the paper implementation  D.half_normal.HalfNormal(0.01)           
        self.beta_mod = self.set_beta_mod()


        # OPTIONAL for the time
        # distribution which defines which timesteps 
        # are going to be blended (time-oriented mask)
        self.time_distribution = []
        for k in range(self.n_modalities):
            self.time_distribution.append(
                self.get_bernoulli_distribution(float(self.p_t_mod[k]))
            )


    def set_beta_mod(self):
        beta_mod = []
        self.beta_mod_mean = []

        for feat_alpha in self.alpha:
            if self.use_beta:
                print("------------using BEta ------------")
                beta_mod.append(D.beta.Beta(feat_alpha, feat_alpha))
                self.beta_mod_mean.append(0)
            else:
                # an alternative would be to sample from uniform or half normal
                # with growing scale, i.e 0.0->loc and 0.4->mean
                print("----------- Using Half Normal ---------")
                beta_mod.append(D.half_normal.HalfNormal(0.01))
                self.beta_mod_mean.append(feat_alpha) # mean of Normal
        print(f"Beta mod is {beta_mod}")

        return beta_mod

    def reset_fere_alpha(self, alpha, verbose=True):
        "alpha here is a list of three values"
        if verbose:
            print(f"Changing fere-alpha from {self.alpha} to {alpha}")
        self.alpha = alpha
        self.beta_mod = self.set_beta_mod()

    ## Can be deleted
    # def get_timestep_weights(self):
    #     weights = torch.ones((self.maxlen, self.window_len))
    #     for i in range(self.maxlen):
    #         if i - self.window_len//2 < 0:
    #             weights[i, :(self.window_len//2 - i)] = 0
    #         if (i + self.window_len // 2) > self.maxlen - 1:
    #             weights[i, (self.maxlen-1-i-self.window_len//2):] = 0
    #     if self.time_window == "uniform":
    #         # import pdb; pdb.set_trace()
    #         return weights / torch.sum(weights, dim=1).unsqueeze(1)
    #     else:
    #         # hann case
    #         # import pdb; pdb.set_trace()
    #         hann_weights = \
    #                 torch.hann_window(window_length=self.window_len + 2,
    #                                   periodic=False)[1:-1]
    #         weights = weights * hann_weights
    #         return weights / torch.sum(weights, dim=1).unsqueeze(1)

    @staticmethod
    def get_bernoulli_distribution(zero_out_prob):
        '''defines a distribution from which we sample which feature-dimensions are going 
        to be blended for every feature tensor (area-oriented masking)
        Tip: probs defines the probability of drawing 1, i.e. the probability of keeping
        '''
        return D.bernoulli.Bernoulli(probs = 1 - torch.tensor(zero_out_prob))

    def set_permute(self, p_batch=0.5):
        # import pdb; pdb.set_trace()
        if p_batch >= random.random():
            self.permute = "batch"
        else:
            self.permute = "time"

    def forward(self, mods, m_ra=1, pad_len=None):
        """FeRe forward implementation
        Sample from $n_modalities$ independent Uniform distributions to mask "mask_area"
        for each one of the modalities.
        Args:
            mods (varargs torch.Tensor): [M, B, L, D_m] Modality representations
            m_ra (int): repeated augmentation index, default = 1
            pad_len (torch.Tensor): [B] tensor of ints denoting the preffix padded length of the
                sequence
        Returns:
            (List[torch.Tensor]): The modality representations. Some of them are dropped
        """
        mods = list(mods)
        local_device = mods[0].device
        if pad_len is None:
            pad_len=torch.ones((mods[0].size()[0], ), dtype=int).to(local_device) * mods[0].size()[0]  # get the batch size 

        #  already in a list form [text, audio, video]
        # mods = list(mods)

        # List of [B, L, D]
        if self.training:
            bsz, seqlen = mods[0].size(0), mods[0].size(1)
            bsz = int(m_ra * bsz)
            if m_ra > 1 and self.discard_zero_pad:
                pad_len = pad_len.repeat(m_ra)
           # import pdb; pdb.set_trace()
           # print(bsz)
            #print(self.beta_mod[0])

            for i_modal, p_modal in enumerate(self.p_t_mod):
                d_modal = mods[i_modal].size(2)
                
                # draws a tensor with size bsz, that defines some probs
                copy_val = self.beta_mod[i_modal].sample((bsz, )) + self.beta_mod_mean[i_modal]
                copy_area = [copy_val for _ in range(self.n_modalities)]

                # defines a distribution from which we sample 
                # which feature-dimensions are going 
                # to be blended for every feature tensor (area-oriented masking)
                self.area_distribution = \
                    self.get_bernoulli_distribution(copy_area[i_modal].clone().detach().float())

                # defines a distribution from which we sample which time-steps are going
                # to be blended for every batch tensor (time-oriented masking)
                modal_timestep_mask = \
                    self.time_distribution[i_modal].sample((bsz, seqlen)).to(mods[0].device)
                # unsqueeze so that you can multiply with the tensor
                modal_timestep_mask = modal_timestep_mask.unsqueeze(2)

                if self.mask_dim:
                    # draw the mask from the area distribution.
                    area_mask = \
                        self.area_distribution.sample((1, d_modal)).to(mods[0].device)
                    area_mask = area_mask.permute(2, 0, 1)      # now it's [B, L, D] 

                if m_ra > 1:
                    mods[i_modal] = mods[i_modal].repeat(m_ra, 1, 1)

               # pdb.set_trace()
                #print(1)

                tmp_tensor = mods[i_modal].clone().detach().to(mods[0].device)
                if self.permute == "time":
                    # print("-------------Time Blending-----------")
                    if self.time_window == "uniform":
                        # print("-------------I go in patches-----------")
                        if self.window_len == -1:
                            if self.discard_zero_pad:
                                # print("---------Sample from the real length only-------")
                                multinomial_probs = torch.ones(bsz, seqlen).to(mods[0].device)
                                #print('Bsz :', bsz)
                               # print('Mult probs', multinomial_probs.size())

                                for k in range(bsz):
                                  #  pdb.set_trace()
                                    ### THIS IS ACTUALLY THE REAL LENGTH
                                    multinomial_probs[k, pad_len[k]:] = 0

                               # pdb.set_trace()
                               # print(2)

                                ### this is the new implementation for the
                                ### front padded sequence
                                ### probability normalization
                                # shape is [B, L]
                                multinomial_probs =  multinomial_probs / pad_len.view(-1, 1).to(local_device)
                              #  print('Mult probs :', multinomial_probs)

                                # draws indices from the multinomial_probs. 
                                # example shape is [B, L]: [[1, 3, 2], [2, 2, 1]] if replacement=true
                                randperm_mask = \
                                    torch.multinomial(
                                        multinomial_probs,
                                        seqlen,
                                        replacement=self.replacement
                                    ).to(mods[0].device)
                                #pdb.set_trace()
                               # print(3)

                                # randperm outputs a "back" padded mask -> need to flip
                                # randperm_mask = torch.flip(randperm_mask, [1])
                                randperm_mask = \
                                    randperm_mask \
                                    + torch.arange(start=0, end=bsz).to(mods[0].device).reshape(bsz, 1) * seqlen
                            
                              #  pdb.set_trace()
                              #  print(4)

                                tmp_tensor = \
                                    tmp_tensor.reshape(-1, d_modal)
                                tmp_tensor = \
                                    tmp_tensor[randperm_mask, :].reshape(bsz, seqlen, d_modal)
                else:
                    randperm_mask = torch.randperm(n=bsz).view(-1)
                    tmp_tensor = tmp_tensor[randperm_mask, :, :]

                if self.constant_val >= 0.0:
                    print("Needs to be fixed for the back paddding scenario")


                ### cut-paste case
                # when area_mask is 1-> keep that feature else replace it
                # when modal_timestep_mask 1-> use existing feature else replace with blended feat
                if self.mixup:
                    # FIXME: here we need to refine the lamda
                    # a potential solution is to do:
                    # mods[i] = (1 - copy_area)*mods[i] +  copy_area*mods[i]
                    # print("------ intermed layer MixUp -------")
                    # mean_lambda = \
                    #     torch.mean(torch.stack(copy_area), dim=0).unsqueeze(1)
                    copy_area_tensor = copy_area[0].reshape(-1, 1, 1)
                    mods[i_modal] = (1 - copy_area_tensor) * mods[i_modal] \
                                    + copy_area_tensor * tmp_tensor 
                else:
                    # cutmix -like approach
                    mods[i_modal] = \
                        (mods[i_modal] * area_mask # protects some features (where area_mask=1)
                        + (1 - area_mask) * tmp_tensor) * (1 - modal_timestep_mask) \
                        + mods[i_modal] * modal_timestep_mask


            return mods

    def __repr__(self):
        shout = (
            self.__class__.__name__
            + "("
            + "p_t_mod="
            + str(self.p_t_mod)
            + f", mask_dim={self.mask_dim}"
            + f", alpha={self.alpha}"
            + f", permute={self.permute}"
            + f", reaplacement={self.replacement}"
            + ")"
        )
        return shout


if __name__ == "__main__":
    import torch
    fere = MMAug(p_t_mod=[0.2, 0.2, 0.2],p_modal=[1.0, 0.1, 0.5]) # Why does it still augment despite prob=0?
    ones = torch.ones(2, 3, 3)              # B x L x D

    fere = MMAug(p_t_mod=[0.2],p_modal=[1.0])
    
    # test FeRe with some custom random input
    rand = torch.tensor([     # M1
                            [[[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]],
                             [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]],
                             [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]],
                             [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]]],
                             
                            #  # M2
                            # [[[16,17,18], [19,20,21], [22,23,24], [25,26,27], [28,29,30]],
                            #  [[16,17,18], [19,20,21], [22,23,24], [25,26,27], [28,29,30]],
                            #  [[16,17,18], [19,20,21], [22,23,24], [25,26,27], [28,29,30]],
                            #  [[16,17,18], [19,20,21], [22,23,24], [25,26,27], [28,29,30]]],

                            # # # M3
                            # [[[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]],
                            #  [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]],
                            #  [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]],
                            #  [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]]]
                           
                        ])  # M x B x L x D
    
    # test FeRe with some custom random input
    # rand = torch.tensor([     # M1
    #                         [[[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]],
    #                          [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]]],
                             
    #                          # M2
    #                         [[[16,17,18], [19,20,21], [22,23,24], [25,26,27], [28,29,30]],
    #                          [[16,17,18], [19,20,21], [22,23,24], [25,26,27], [28,29,30]]],

    #                         # # M3
    #                         [[[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]],
    #                          [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]]]
                           
    #                     ])  # M x B x L x D
    
    print(rand.size())
    print(fere(rand, pad_len=torch.ones((rand.size()[1], ), dtype=int)*rand.size()[1]))# randint(0,2, (2,))))

    # for i in range(1000):
    #     rand = torch.rand(3, 2, 6,3)
    #     fere(rand, pad_len=torch.tensor((1,1))) # if 0 throws RuntimeError: probability tensor contains either `inf`, `nan` or element < 0



    # # test FeRe with some custom random input
    # rand = torch.tensor([     # M1
    #                         [[[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]],
    #                          [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]]],
                             
    #                          # M2
    #                         [[[16,17,18], [19,20,21], [22,23,24], [25,26,27], [28,29,30]],
    #                          [[16,17,18], [19,20,21], [22,23,24], [25,26,27], [28,29,30]]],

    #                         # # M3
    #                         [[[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]],
    #                          [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12], [13, 14, 15]]]
                           
    #                     ])  # M x B x L x D

    # print(rand.size())
    # print(fere(rand, pad_len=torch.tensor((2,3))))# randint(0,2, (2,))))


