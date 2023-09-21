import torch
import os

from embedding_space import EmbeddingSpace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_name,dataloader,k):
  
  '''
  perform the cluster evaluation, it evaluates the
  representativity of the cluster by testing the k
  closer images to the centroid of a certain class

  Args: 
        - model name : name of the model to be evaluated
        - dataloader : data loader for the test set
        - k          : number nearest neighbour to consider
  Return:
        - accuracy of the clusters
  '''
  dir = '/content/drive/MyDrive/Mushroom/'
  model_path = os.path.join(dir,model_name)
  model = torch.load(model_path,map_location=torch.device(device))
  embedding_space = EmbeddingSpace(model, dataloader, device,k)
  return embedding_space.validate()



def k_accuracy(model_name,train_dataloader,val_dataset,k):

  dir = '/content/drive/MyDrive/Mushroom/'
  model_path = os.path.join(dir,model_name)
  model = torch.load(model_path,map_location=torch.device(device))
  embedding_space = EmbeddingSpace(model, train_dataloader, device,k)

  correct = 0
  distances={k:[] for k in range(len(val_dataset))}

  for item in range(len(val_dataset)):
    img , label = val_dataset[item]
    img = img.to(device)
    embedding = model.forward_once(img.unsqueeze(0))

    for index in range(len(embedding_space.embeddings)):

      train_emb = embedding_space.embeddings[index].unsqueeze(0)
      train_class = int(embedding_space.classes[index].item())

      euclidean_distance = torch.cdist(embedding,train_emb , p=2.0)
      distance = euclidean_distance.item()
      distances[item].append([train_class , distance, label])

  sorted_dist = {key: sorted(value, key=lambda x: x[1]) for key, value in distances.items()}
  for idx in sorted_dist:
    for c in sorted_dist[idx][:k]:
      if c[0]==c[2]:
        correct += 1
        break

  accuracy = 100 * correct / len(val_dataset)
  return accuracy

def closer(embedding_space,emb,label):

  img = emb.reshape(1, -1)
  distances={k:[] for k in range(len(embedding_space.embeddings))}

  for i in range(len(embedding_space.embeddings)):
    tmp = embedding_space.embeddings[i].reshape(1, -1)
    euclidean_distance = torch.cdist(img, tmp, p=2.0)
    distance = euclidean_distance.item()
    distances[i].append([distance,int(embedding_space.classes[i].item())])

  sorted_items = sorted(distances.items(), key=lambda x: x[1])
  predicted=[]
  predicted.append(sorted_items[0][1][0][1])
  predicted.append(sorted_items[1][1][0][1])
  predicted.append(sorted_items[2][1][0][1])

  return True if label in predicted else False

def get_indexes_sub(class_index: int, dataset):
    idx = [index for index, (_, class_idx) in enumerate(dataset) if class_idx == class_index]
    return idx

def one_shot_accuracy(model_name,test_dataset,train_dataloader,k=1):

  dir = '/content/drive/MyDrive/Mushroom/'
  model_path = os.path.join(dir,model_name)
  model = torch.load(model_path,map_location=torch.device(device))

  embedding_space = EmbeddingSpace(model, train_dataloader, device)
  correct = 0
  samples = 0
  for i in range(200,215):

    indexes = get_indexes_sub(i,test_dataset)
    samples+=len(indexes[1:])
    for num in range(k):
      new_img , new_label = test_dataset[indexes[num]]
      new_img=new_img.to(device)
      new_label = torch.tensor([new_label]).to(device)
      new_img_emb = model.forward_once(new_img.unsqueeze(0))
      embedding_space.add_item(new_img_emb,new_label)

    for idx in indexes[k:]:

      img,label = test_dataset[idx]
      img = img.to(device)
      label = torch.tensor([label]).to(device)
      emb = model.forward_once(img.unsqueeze(0))
      class_200 = closer(embedding_space,emb,i)
      correct+= 1 if class_200 else 0

  return correct*100/samples

