import torch
import pandas as pd
import numpy as np

weight_mix = np.load('weight distribution mix with uniform distribution.npy')

class RarityWeightedLoss():

    def __init__(self, pixelProbabilities, lamda, num_bins):
        #distribution = pd.read_csv(pixelProbabilitiesCSV, encoding='UTF-8')
        self.pixel_dist = torch.tensor(pixelProbabilities)
        self.lamda = lamda
        self.Q = num_bins

    def __call__(self, prediction, target):
        rows = target.shape[0]
        cols = target.shape[1]
        #cross entropy of z, z_hat (multiply and sum over q's
        #non_z =  torch.nonzero(prediction)
        #multiply = target[non_z.data]
        Z_hat_log = torch.where(prediction > 0, -torch.log(prediction), 0)
        mul = Z_hat_log * target
        sum_q = torch.sum(mul, dim=2, keepdim=False)
        L = -torch.sum(self.weighting(target))*sum_q
        #q = torch.matmul(target, torch.t(-torch.log(prediction)))
        #sum_q = torch.sum(q, dim=2, keepdim=False)
        #get the most likely color
        # pixel_val_for_weight = torch.argmax(target, dim=2)
        # pixel_val_for_weight = torch.flatten(pixel_val_for_weight)
        #retrieve it's prior probability
        # probabilities = torch.tensor(self.pixel_dist[pixel_val_for_weight])
        # probabilities = torch.reshape(probabilities, (rows, cols))
        #mix with normal distribution
        # weights = 1/((1-self.lamda)*probabilities + self.lamda/self.Q)
        #normalize
        # weights = weights/torch.sum(weights)
        # loss = -torch.sum(weights*sum_q)
        return L

    def weighting(self, Z):
        most_likely_bin = np.argmax(Z, axis=2) # @todo Check the axis
        v = weight_mix(most_likely_bin)
        return v




