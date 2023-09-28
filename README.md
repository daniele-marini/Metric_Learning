# Metric_Learning

Project work for the "Machine Learning for Computer Vision" course of the Artificial Intelligence Master's Degree at University of Bologna.

# Task description
The aim of the Project work is to explore the potential of a metric learning approach simulating an open word enviroment. The goal is to correctly classify images between more than 200 different classes. The classification is made through a metric learning approach, so relying on the distances between the images. In order to do that, after the creation and the training of a Siamese model, I used it to create an embedding space. 
Finally, after several techniques to improve the efficiency of model, I tested it with unseen classes, in order to test his behaviour in a one-shot learning enviroment.

![image](https://github.com/daniele-marini/Metric_Learning/blob/main/imgs/cluster.png)

# Dataset
The dataset used for the task is the [”Mushrooms images classification 215” dataset](https://www.kaggle.com/datasets/daniilonishchenko/mushrooms-images-classification-215). 
This dataset contains 3122 images (512x512) of 215 different classes, each one with approximately 12/15 images.
The dataset have been used so as to allow the model to be used on unseen data, so the first 200 classes have been used for the training and the validation, while the remaining 15 for testing the zero-shot learning.

# Siamese model
The model used for the creation of the embedding space is a Siamese model composed using as backbones two pretrained ResNet50 followed by a new classifier composed by 3 linear layers. 
The use of a Siamese model allowed me to apply the chosen loss function and learn how to correctly place images inside the embedding space.

## Triplet Loss + Triplet Dataset
As loss function I used the Triplet Loss, so I had to modify the dataset so that each element is a triplet that contains:
* an **anchor image**
* a **positive image** (of the same class w.r.t. the anchor)
* a **negative image** (of a different class w.r.t. the anchor)
At the beginning both the positive and the negative images of each triplet are chosen randomly.

# Hard negative mining
In order to test the model under different conditions I tried to improve with Hard negative mining techniques:
* **Hard negative-positive mining** -> In these triplets the positive image is the farthest from the anchor, while the negative image is the closest the the anchor
* **Hard negative mining** -> the triplets are composed by the anchor, a random positive image and an hard negative image close to the anchor
* **Semi-hard negative mining** -> The positive image is still random and the negative image is taken so that the distance between the anchor and the negative is between the distance between the anchor and the positive and the same distance plus a margin
* **Mixed mining** -> the triplet are composed by the anchor, a random positive and a negative image close to the positive

# Evaluation
As we can see form the table below, the best performing model is always the one trained with random triplets (base model), while the ones trained with hard triplets have more difficulty, even with a bigger embedding space dimension.

![image](https://github.com/daniele-marini/Metric_Learning/blob/main/imgs/accuracy_table.png)

# One-shot learning
The 9 model created with the previous techniques have been tested on the last 15 unseen classes. In order to test it I added one unseen image in the embedding space and then I tested the other images of the same unseen class.

The results were not particularly satisfactory, the best accuracy was given by the main model with 64 dimension embedding space, which obtained around **20% of accuracy**.
## K-shot learning
Finally I tried to add three images instead on one.
In this case the results as expected are slightly better, reaching the **28% of accuracy**.

# Conlusion
In conclusion, metric learning has proven to be an effective approach for addressing clas- sification tasks, offering great adaptability through the application of one-shot learning in open-world scenarios. While right now metric learning finds predominant use in face recog- nition tasks, its potential extends to diverse fields.
The dataset employed in this project presented a strong challenge due to the pronounced similarities among classes. Yet, this challenging environment provided a valuable opportu- nity to explore and compare various solutions.
It is important to note that this work does not include all possible alternatives. The focus was placed on a specific loss function, with experimentation with different variants. However, numerous alternative loss functions, such as Contrastive Loss, Angular Loss, and many others, may offer greater efficacy for this dataset.
Further investigation into these alternatives could yield even more promising results.
