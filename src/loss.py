import torch
import pandas as pd
import numpy as np


weight_mix = np.load('weight distribution mix with uniform distribution.npy')
PRIOR_PROBS = np.load('prior_probs.npy')
class RarityWeightedLoss():

    def __init__(self, pixelProbabilities, lamda, num_bins):
        #distribution = pd.read_csv(pixelProbabilitiesCSV, encoding='UTF-8')
        self.weighting_factor = PRIOR_PROBS
        self.weight_mix = torch.tensor(weight_mix)
        self.pixel_dist = torch.tensor(pixelProbabilities)
        self.lamda = lamda
        self.Q = num_bins

    def __call__(self, prediction, target):
        '''computes the class rebalanced multinomial crossentropy loss
            in: prediction, target, soft encoded predicted and ground truth Z vectors
            out: loss, scalar'''
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




