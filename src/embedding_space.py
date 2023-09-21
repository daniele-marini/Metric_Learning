import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from typing_extensions import Self



class EmbeddingSpace():

  '''
  Every image should be represented as an embedding vector,
  Here we have a representation of an embedding space
  '''

  def __init__(self, model: Module, loader_images : DataLoader, device: torch.device, k=5):

    """
    Create the embedding space

    Args:
        model: the CNN backbone.
        loader_images: dataloader of images.
        device: device where I'm working

    Returns:
        embedding space
    """
    self.device = device
    self.model = model.eval().to(self.device)
    self.embeddings = torch.tensor([]).to(self.device)
    self.classes = torch.tensor([]).to(self.device)
    self.loader_images=loader_images
    self.k = k

    for idx_batch, (images, images_class) in enumerate(loader_images):

      images, images_class = images.to(self.device), images_class.to(self.device)

      with torch.no_grad():
        out = torch.squeeze(model.forward_once(images),dim=1).to(self.device)
        self.embeddings = torch.cat((self.embeddings, out))
        self.classes = torch.cat((self.classes, images_class))


  def get_idx(self,class_index):
    class_num = 215 if self.zero_shot else 200
    idx = {k:[] for k in range(class_num)}
    for i in range(len(self.classes)):
      idx[int(self.classes[i].item())].append(i)
    return idx[class_index]

  def get_centroid(self,class_index):
    indexes = self.get_idx(class_index)
    spaces=[]
    for i in indexes:
      spaces.append(self.embeddings[i])
    # Convert the list of tensors to a single tensor
    tensor_data = torch.stack(spaces)
    # Calculate the centroid by taking the mean along the first axis
    centroid = torch.mean(tensor_data, dim=0)
    return centroid



  def centroid_accuracy(self):
    accuracy = []
    class_num = [k for k in range(200,215)] if self.zero_shot else [k for k in range(200)]
    for cluster in class_num:
      correct=0
      distances={k:0 for k in range(len(self.embeddings))}
      cluster_indexes = self.get_idx(cluster)
      cluster_centroid = self.get_centroid(cluster)
      cluster_centroid = cluster_centroid.reshape(1, -1)

      for index in range(len(self.classes)):
        tmp = self.embeddings[index].reshape(1, -1)
        euclidean_distance = torch.cdist(cluster_centroid, tmp, p=2.0)
        distance = euclidean_distance.item()

        distances[index] = distance

      sorted_items = sorted(distances.items(), key=lambda x: x[1])
      for closer_indexes in sorted_items[:self.k]:
        if closer_indexes[0] in cluster_indexes:
          correct +=1
      cluster_accuracy = (correct * 100) / self.k
      accuracy.append(cluster_accuracy)
    return sum(accuracy) / len(accuracy)

  def add_item(self, new_element,images_class):
        new_element = new_element.to(self.device)
        self.embeddings = torch.cat((self.embeddings, new_element))
        self.classes = torch.cat((self.classes, images_class))