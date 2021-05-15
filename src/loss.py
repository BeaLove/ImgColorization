import torch
import numpy as np
import misc.npy_loader.loader as npy

weight_mix = npy.load('weight_distribution_mix_with_uniform_distribution')
PRIOR_PROBS = npy.load('authors_prior_probs')
class RarityWeightedLoss():

    def __init__(self, weight_mix, lamda, num_bins):
        #distribution = pd.read_csv(pixelProbabilitiesCSV, encoding='UTF-8')
        self.weight_mix = torch.tensor(weight_mix)
        self.lamda = lamda
        self.Q = num_bins

    def __call__(self, prediction, target):
        '''computes the class rebalanced multinomial crossentropy loss
            in: prediction, target, soft encoded predicted and ground truth Z vectors
            out: loss, scalar value'''
        #cross entropy of z, z_hat (multiply and sum over q's):
        logs = torch.where(prediction > 0.0, torch.log(prediction), 0.0)
        cross_entropy = target * logs
        sum = torch.sum(cross_entropy, dim=2, keepdim=False)
        weights = self.weighting(target)
        weighted = sum * weights
        loss = -torch.sum(weighted)

        return loss

    def weighting(self, Z):
        rows = Z.shape[0]
        cols = Z.shape[1]
        most_likely_bin = torch.argmax(Z, axis=2)#

        v = torch.index_select(self.weight_mix, 0, torch.flatten(most_likely_bin))
        return v.reshape(rows, cols)/torch.sum(v)




