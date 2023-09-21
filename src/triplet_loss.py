import torch
from torch.nn import Module
import torch.nn.functional as F

class TripletLoss(Module):

    '''
    Loss function used for the triplet dataset
    Compare the distance between:
                               - the image and the anchor
                               - the image and the negative
    and take into account a margin
    '''
    def __init__(self, margin : int):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs : torch.Tensor, labels : torch.Tensor) -> torch.Tensor:

        if len(outputs) == 3:

            positive_distance = F.pairwise_distance(outputs[0], outputs[1])
            negative_distance = F.pairwise_distance(outputs[0], outputs[2])
            losses = torch.relu(positive_distance - negative_distance + self.margin)

            return torch.mean(losses)

        raise ValueError('This is the Triplet Loss but more (or less) than 3 tensors were unpacked')