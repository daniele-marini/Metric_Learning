import torch
import random
import numpy as np
import torchvision
from typing import Callable, Tuple
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def get_mean_and_std(dataset: torchvision.datasets.ImageFolder) -> Tuple[float, float]:

    """
    Compute mean and std for the dataset.

    Args:
        dataset: the dataset.

    Returns:
        The mean and the std on each channels computed over the dataset.
    """
    dataset_loader = DataLoader(
        dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=1
    )
    sums = torch.zeros(3)
    sums_of_square = torch.zeros(3)
    count = 0

    for images, _ in dataset_loader:
        b, _, h, w = images.shape
        num_pix_in_batch = b * h * w
        sums += torch.sum(images, dim=[0, 2, 3])
        sums_of_square += torch.sum(images ** 2, dim=[0, 2, 3])
        count += num_pix_in_batch

    mean = sums / count
    var = (sums_of_square / count) - (mean ** 2)
    std = torch.sqrt(var)

    return mean, std


def fix_random(seed: int) -> None:

    """
    Fix all the possible sources of randomness.

    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def show_grid(dataset: torchvision.datasets.ImageFolder,
              process: Callable,
              indices: list) -> None:

    """
    Shows a grid with random images taken from the dataset.

    Args:
        dataset: the dataset containing the images.
        process: a function to apply on the images before showing them.
    """
    if len(indices) > 8:
      indices=indices[:8]
    fig = plt.figure(figsize=(15, 5))

    for count, idx in enumerate(indices):
        fig.add_subplot(2, 5, count + 1)
        title = dataset.classes[dataset[idx][1]]
        plt.title(title)
        image_processed = process(dataset[idx][0]) if process is not None else dataset[idx][0]
        plt.imshow(transforms.ToPILImage()(image_processed))
        plt.axis("off")

    plt.tight_layout()
    plt.show()

class NormalizeInverse(torchvision.transforms.Normalize):

    def __init__(self, mean, std) -> None:

        """Reconstructs the images in the input domain by inverting
        the normalization transformation.

        Args:
            mean: the mean used to normalize the images.
            std: the standard deviation used to normalize the images.
        """
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
    

def get_indexes(class_index: int,dataset):
    idx = [index for index, (_, class_idx) in enumerate(dataset.imgs) if class_idx == class_index]
    return idx


def get_centroid(embedding_space,class_index,dataset):
  indexes=get_indexes(class_index,dataset)
  spaces=[]
  for i in indexes:
    spaces.append(embedding_space.embeddings[i])
  # Convert the list of tensors to a single tensor
  tensor_data = torch.stack(spaces)
  # Calculate the centroid by taking the mean along the first axis
  centroid = torch.mean(tensor_data, dim=0)
  return centroid

def show_errors(embedding_space,target_class,margin,dataset):

  centroid=get_centroid(embedding_space,target_class,dataset).reshape(1, -1)
  indexes=get_indexes(target_class,dataset)
  errors_index=[0]
  min_distance=100

  for i in indexes:
    tmp=embedding_space.embeddings[i].reshape(1, -1)

    euclidean_distance = torch.cdist(centroid, tmp, p=2.0)
    distance = euclidean_distance.item()
    print(i,distance)

    if distance < min_distance:
      min_distance=distance
      errors_index[0]=i

    if distance > margin:
      errors_index.append(i)

def show_closer(embedding_space,image_index,dataset,proc):

  img = embedding_space.embeddings[image_index].reshape(1, -1)
  distances={k:0 for k in range(len(embedding_space.embeddings))}

  for i in range(len(embedding_space.embeddings)):

    tmp=embedding_space.embeddings[i].reshape(1, -1)

    euclidean_distance = torch.cdist(img, tmp, p=2.0)
    distance = euclidean_distance.item()

    distances[i] = distance


  sorted_items = sorted(distances.items(), key=lambda x: x[1])

  closer_values = [item for item in sorted_items[:5]]
  print(closer_values)

  closer_indexes = [item[0] for item in sorted_items[:5]]

  show_grid(dataset=dataset, process=proc , indices=closer_indexes)
