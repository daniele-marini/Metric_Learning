import torch
import os
import torchvision
import numpy as np
import random
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.optim import Adam

from utils import *
from early_stopper import EarlyStopper
from triplet_dataset import TripletDataset
from siamese_model import SiameseNetwork
from training_functions import training_loop
from triplet_loss import TripletLoss
from embedding_space import EmbeddingSpace
from eval import *

from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"You are currently using: {device}")

# load your kaggle.json file in the content of colab and then:
    # !pip -qq install kaggle
    # !mkdir ~/.kaggle
    # !cp kaggle.json ~/.kaggle/
    # !chmod 600 ~/.kaggle/kaggle.json
    # !kaggle datasets download daniilonishchenko/mushrooms-images-classification-215
    # !unzip -qq mushrooms-images-classification-215.zip

# DATASET PRE-PROCESSING

folder_path = '/content/data/data'
dataset_for_std = torchvision.datasets.ImageFolder(folder_path, transform=transforms.ToTensor())
mean_image_net, std_image_net = get_mean_and_std(dataset_for_std)

normalize = transforms.Normalize(mean_image_net, std_image_net)
size_image = 224
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize((size_image, size_image),antialias=True),
                                normalize])

torch.set_default_dtype(torch.float32)
fix_random(42)
generator = torch.Generator().manual_seed(42)
workers = os.cpu_count()
denormalize = NormalizeInverse(mean_image_net, std_image_net)
batch_size=32

dataset = ImageFolder(folder_path, transform = transform)
num_datapoints = 2897
# Create a Subset of the original dataset with the first 200 classes
main_dataset = data.Subset(dataset, range(num_datapoints))
# split dataset into train and validation set
train_dataset, val_dataset = random_split(main_dataset, (0.8, 0.2), generator = generator)
# Create a Subset of the original dataset with the last 15 classes
test_dataset = data.Subset(dataset, range(num_datapoints,3122))
# dataloader definitions
final_dataloader = DataLoader(dataset,shuffle = False,num_workers = workers,pin_memory = True,batch_size = batch_size)
full_dataloader = DataLoader(main_dataset,shuffle = False,num_workers = workers,pin_memory = True,batch_size = batch_size)
train_dataloader = DataLoader(train_dataset,shuffle = False,num_workers = workers,pin_memory = True,batch_size = batch_size)
val_dataloader = DataLoader(val_dataset,shuffle = False,num_workers = workers,pin_memory = True,batch_size = batch_size)
test_dataloader = DataLoader(test_dataset,shuffle = False,num_workers = workers,pin_memory = True,batch_size = batch_size)

# TRAINING RANDOM TRIPLET MODEL
triplet_train_dataset = TripletDataset(train_dataset)
triplet_val_dataset = TripletDataset(val_dataset)

triplet_train_loader = DataLoader(triplet_train_dataset,shuffle = True,num_workers = workers,pin_memory = True,batch_size = batch_size)
triplet_val_loader = DataLoader(triplet_val_dataset,shuffle = False,num_workers = workers,pin_memory = True,batch_size = batch_size)

backbone = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
lr = 1e-4
num_epochs = 500

embeddig_dimension = 64
model=SiameseNetwork(output = embeddig_dimension, backbone = backbone, freezed=False).to(device)

optimizer = Adam(model.parameters(), lr = lr)
early_stopper = EarlyStopper(patience = 3, min_delta = 0)

history = training_loop(num_epochs=num_epochs,
                        optimizer=optimizer,
                        model = model,
                        train_loader = triplet_train_loader,
                        val_loader = triplet_val_loader,
                        loss_func = TripletLoss,
                        device = device,
                        accuracy_margin = 0.5,
                        verbose = True,
                        early_stopping = early_stopper
)
# save the model
path = "/content/drive/MyDrive/Mushroom"
if not os.path.exists(path):
  os.makedirs(path)

dir=os.path.join(path,'main_model.pdh')
torch.save(model,dir)

# load the model
model = torch.load('/content/drive/MyDrive/Mushroom/main_model.pdh',map_location=torch.device(device))

# TRAINING HARD TRIPLET MODEL
embedding_space = EmbeddingSpace(model, train_dataloader, device)

mod='hard_negative' # or semi-hard or mixed-training or hard_positive_negative
hard_train_dataset = TripletDataset(train_dataset, model, embedding_space,modality=mod) #
hard_val_dataset = TripletDataset(val_dataset, model, embedding_space,modality=mod)

hard_train_loader = DataLoader(hard_train_dataset,shuffle = True,num_workers = 0,pin_memory = True,batch_size = batch_size)
hard_val_loader = DataLoader(hard_val_dataset,shuffle = True,num_workers = 0,pin_memory = True,batch_size = batch_size)

backbone = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
lr = 1e-4
num_epochs = 500

embeddig_dimension = 64
model=SiameseNetwork(output = embeddig_dimension, backbone = backbone, freezed=False).to(device)

optimizer = Adam(model.parameters(), lr = lr)
early_stopper = EarlyStopper(patience = 3, min_delta = 0)

history = training_loop(num_epochs=num_epochs,
                        optimizer=optimizer,
                        model = model,
                        train_loader = triplet_train_loader,
                        val_loader = triplet_val_loader,
                        loss_func = TripletLoss,
                        device = device,
                        accuracy_margin = 0.5,
                        verbose = True,
                        early_stopping = early_stopper
)
# save the model
path = "/content/drive/MyDrive/Mushroom"
if not os.path.exists(path):
  os.makedirs(path)
dir=os.path.join(path,'hard_model.pdh')
torch.save(model,dir)

# load the model
model = torch.load('/content/drive/MyDrive/Mushroom/hard_model.pdh',map_location=torch.device(device))

# EMBEDDING SPACE TEST
margin=2
show_errors(embedding_space,111,margin,dataset) # 111 is the class
show_closer(embedding_space,1618,dataset,denormalize) # 1618 is the index of the image

# EVALUATION

model_name = 'main_model-pdh'

# cluster evaluation
cluster_acc = evaluate(model_name,full_dataloader,k=5)
print(cluster_acc)

# K accuracy
model_accuracy = k_accuracy(model_name,train_dataloader,val_dataset,k=5)
print(model_accuracy)

# one-shot accuracy
model_one_shot_accuracy = one_shot_accuracy(model_name, test_dataset, train_dataloader, k=1) # change k for k-shot learning
print(model_one_shot_accuracy)
