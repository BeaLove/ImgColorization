import torch
import numpy as np
#import misc.npy_loader.loader as npy


#weight_mix = npy.load('weight_distribution_mix_with_uniform_distribution_normalized')
#PRIOR_PROBS = npy.load('full_probabilities')
class RarityWeightedLoss(torch.nn.Module):

    def __init__(self, weight_mix):
        super().__init__()
        #distribution = pd.read_csv(pixelProbabilitiesCSV, encoding='UTF-8')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.weight_mix = torch.tensor(weight_mix, requires_grad=True)
        self.weight_mix = self.weight_mix.to(device)


    def __call__(self, prediction, target):
        '''computes the class rebalanced multinomial crossentropy loss
            in: prediction, target, soft encoded predicted and ground truth Z vectors
            out: loss, scalar value'''
        #cross entropy of z, z_hat (multiply and sum over q's):
        height = prediction.shape[2]
        width = prediction.shape[3]
        cross_entropy = target * prediction
        sum = torch.sum(cross_entropy, dim=1, keepdim=True)

        weights = self.weighting(target)
        weighted = sum * weights
        loss = -torch.sum(weighted)/(height*width)
        return loss

    def weighting(self, Z):
        batch_size = Z.shape[0]
        height = Z.shape[2]
        width = Z.shape[2]
        most_likely_bin = torch.argmax(Z, axis=1)#channel dimension

        v = torch.index_select(self.weight_mix, 0, torch.flatten(most_likely_bin))

        return v.view(batch_size, 1, height, width)

class L2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
    def __call__(self, prediction, target):
        target = torch.tensor(target, dtype=float, requires_grad=True)
        loss = torch.tensor(self.loss(prediction, target), dtype=float, requires_grad=True)
        return loss


