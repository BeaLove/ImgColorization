import torch
import numpy as np
import misc.npy_loader.loader as npy


weight_mix = npy.load('weight_distribution_mix_with_uniform_distribution_normalized')
#PRIOR_PROBS = npy.load('full_probabilities')
class RarityWeightedLoss():

    def __init__(self, weight_mix, lamda=0.5, num_bins=441):
        #distribution = pd.read_csv(pixelProbabilitiesCSV, encoding='UTF-8')
        self.weight_mix = torch.tensor(weight_mix, requires_grad=True)
        #self.lamda = lamda
        #self.Q = num_bins

    def __call__(self, prediction, target):
        '''computes the class rebalanced multinomial crossentropy loss
            in: prediction, target, soft encoded predicted and ground truth Z vectors
            out: loss, scalar value'''
        #cross entropy of z, z_hat (multiply and sum over q's):
        batch_size = prediction.shape[0]
        height = prediction.shape[2]
        width = prediction.shape[3]
        #zero = torch.tensor(0, dtype=float)
        #logs = torch.where(prediction > 0.0, torch.tensor(torch.log(prediction), dtype=float), 0.0)
        logs = torch.log(prediction)
        cross_entropy = target * prediction
        sum = torch.sum(cross_entropy, dim=1, keepdim=True)
        #loss = -torch.sum(sum)/(batch_size*height*width*30)
        weights = self.weighting(target)
        weighted = sum * weights
        loss = -torch.sum(weighted)/(height*width)
        return loss

    def weighting(self, Z):
        batch_size = Z.shape[0]
        channels = Z.shape[1]
        height = Z.shape[2]
        width = Z.shape[2]
        most_likely_bin = torch.argmax(Z, axis=1)#channel dimension

        v = torch.index_select(self.weight_mix, 0, torch.flatten(most_likely_bin))

        return v.view(batch_size, 1, height, width)

class L2Loss():
    def __init__(self):
        self.loss = torch.nn.MSELoss()
    def __call__(self, prediction, target):
        return self.loss(prediction, target)


