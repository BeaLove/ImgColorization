import torch
import pandas as pd

class LossCriterion():

    def __init__(self, pixelProbabilitiesCSV, lamda, num_bins):
        distribution = pd.read_csv(pixelProbabilitiesCSV, encoding='UTF-8')
        self.pixel_dist = torch.tensor(distribution.values)
        self.lamda = lamda
        self.Q = num_bins

    def __call__(self, prediction, target):
        rows = target.shape[0]
        cols = target.shape[1]
        q = torch.matmul(target, -torch.log(prediction))
        sum_q = torch.sum(q, dim=2, keepdim=False)
        pixel_val_for_weight = torch.argmax(target, dim=2)
        probabilities = torch.tensor(self.pixel_dist[pixel_val_for_weight.resize(
            rows*cols, 1)]).reshape(rows, cols)
        weights = 1/((1-self.lamda)*probabilities + self.lamda/self.Q)
        loss = -torch.sum(weights*sum_q)
        return loss


