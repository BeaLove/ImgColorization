import torch
import pandas as pd
import numpy as np

PRIOR_PROBS = np.load('prior_probs.npy')
print("hellot")

class LossCriterion():

    def __init__(self, pixelProbabilitiesCSV, lamda, num_bins):
        distribution = pd.read_csv(pixelProbabilitiesCSV, encoding='UTF-8')
        self.pixel_dist = torch.tensor(distribution.values)
        self.lamda = lamda
        self.Q = num_bins

    def __call__(self, prediction, target):
        rows = target.shape[0]
        cols = target.shape[1]
        #cross entropy of z, z_hat (multiply and sum over q's
        q = torch.matmul(target, -torch.log(prediction))
        sum_q = torch.sum(q, dim=2, keepdim=False)
        #get the most likely color
        pixel_val_for_weight = torch.argmax(target, dim=2)
        #retrieve it's prior probability
        probabilities = torch.tensor(self.pixel_dist[pixel_val_for_weight.resize(
            rows*cols, 1)]).reshape(rows, cols)
        #mix with normal distribution
        weights = 1/((1-self.lamda)*probabilities + self.lamda/self.Q)
        #normalize
        weights = weights/torch.sum(weights)
        loss = -torch.sum(weights*sum_q)
        return loss


