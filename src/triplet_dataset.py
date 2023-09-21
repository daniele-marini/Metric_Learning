import torch
import random
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TripletDataset(Dataset):

    '''
    Create different hard triplets dataset, where each element contains:
        - an anchor image
        - a positive (with the same class of the anchor)
        - a negative image (with a different class w.r.t. the anchor)

    Args:
        dataset: ImageFolder dataset containing all the images
        model: trained Siamese model
        embedding_space: embedding space created by the model
        margin: for the semi-hard triplets
        modality: specify the type of hard triplets:     - random
                                                         - hard_negative_positive
                                                         - hard_negative
                                                         - mixed-training
                                                         - semi-hard
                                                        

    Returns:
        triplet dataset
    '''

    def __init__(self, dataset, model=None, embedding_space=None, margin=1.5, modality='random'):
        self.dataset = dataset
        self.model = model
        self.embedding_space = embedding_space
        self.index_dict = self._create_dict()
        self._random=0
        self.modality=modality
        self.margin=margin

    def _create_dict(self):
        indexes = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in indexes:
                indexes[label] = []
            indexes[label].append(idx)

        return indexes

    def _get_hard_indexes(self, image, image_class):
        anchor_out = torch.squeeze(self.model.forward_once(image.unsqueeze(0)),dim=1)

        # extract positive index
        if self.modality == 'hard_negative_positive':
          # extract the hard positive
          positive_distances = {i:[0,0] for i in self.index_dict[image_class]}

          for i in self.index_dict[image_class]:
            tmp = self.embedding_space.embeddings[i].unsqueeze(0)
            euclidean_distance = torch.cdist(anchor_out,tmp , p=2.0)
            positive_distances[i]=[euclidean_distance.item(),tmp]

          sorted_distances = sorted(positive_distances.items(), key=lambda x: x[1])

          self._random = random.choice([idx for idx in list(self.index_dict.keys()) if idx != image_class])

          positive_index = sorted_distances[-1][0] if len(sorted_distances) >=2 else self._random
          hard_positive_embedding = sorted_distances[-1][1][1] if len(sorted_distances) >=2 else self.embedding_space.embeddings[self._random]
        else:
          #extract the random  positive
          positive_index = random.choice(self.index_dict[image_class])
          positive_embedding = self.embedding_space.embeddings[positive_index].unsqueeze(0)
          positive_distance = torch.cdist(anchor_out, positive_embedding, p=2.0)
          positive_distance = positive_distance.item()


        #extract the negative index
        indices = self.index_dict.values()
        merged_indices = [item for sublist in indices for item in sublist]
        negative_indices = [x for x in merged_indices if x not in self.index_dict[image_class]]
        negative_distances = {i:0 for i in negative_indices}

        for index in negative_indices:
          tmp = self.embedding_space.embeddings[index].unsqueeze(0)
          euclidean_distance = torch.cdist(anchor_out, tmp, p=2.0) if self.modality != 'mixed-training' else torch.cdist(positive_embedding.unsqueeze(0), tmp, p=2.0)
          negative_distances[index]=euclidean_distance.item()

        sorted_distances = sorted(negative_distances.items(), key=lambda x: x[1])

        if self.modality == 'semi-hard':
          negative_range = [index for index, value in sorted_distances if positive_distance <= value < positive_distance+self.margin]
          wither_range = [index for index, value in sorted_distances]
          negative_index = random.choice(negative_range) if negative_range != [] else random.choice(wither_range)
        else:
          negative_index = sorted_distances[0][0]

        return positive_index , negative_index

    def __getitem__(self, index):
        anchor, anchor_class = self.dataset[index]

        if self.modality == 'random':
          positive_idx = random.choice(self.index_dict[anchor_class])
          random_negative_class = random.choice([idx for idx in list(self.index_dict.keys()) if idx != anchor_class])
          negative_idx = random.choice(self.index_dict[random_negative_class])
        else:
          positive_idx , negative_idx = self._get_hard_indexes(anchor.to(device), anchor_class)

        positive, _ = self.dataset[positive_idx]
        negative, _ = self.dataset[negative_idx]

        inputs = torch.stack((anchor, positive, negative))
        target = torch.tensor([0, 1])

        return inputs, target

    def __len__(self):
        return len(self.dataset)