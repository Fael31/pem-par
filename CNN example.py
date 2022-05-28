from __future__ import print_function, division

import torch
import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import time
import os

from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import ReLU

cudnn.benchmark = True
plt.ion()   # interactive mode

print("Using torch", torch.__version__)
torch.manual_seed(42) # Setting the seed

data_transforms = {
    'Training': transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ToTensor()
    ]),

    'Testing': transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ToTensor()
    ]),
}

# sesuaikan untuk menuju folder FaceData
data_dir = '/content/gdrive/MyDrive/Colab_Notebooks/FaceData'
#

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['Training', 'Testing']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=2)
              for x in ['Training', 'Testing']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['Training', 'Testing']}
class_names = image_datasets['Training'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Ini ganti aja devicenya cuda:0 atau cpu sesuai kebutuhan 
class_names

def train_model(model, criterion, optimizer, num_epochs=25):
    phase='Training'
    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train() 

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['Training']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return model

def test_model(model):
    model.eval()
    phase = 'Testing'

    y_true = []
    y_pred = []

    for inputs, labels in dataloaders['Testing']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            # TASK
            # update variable y_true dan y_pred sehingga y_true lengkap berisi label asli 
            # dan y_pred berisi label prediksi model

            #credits: https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
            softmax = torch.exp(outputs).cpu()
            prob = list(softmax.numpy())
            predictions = list(np.argmax(prob, axis=1))
            #end of credits

            labels = list(labels.cpu().numpy())
            for i in range(len(labels)):
              y_true.append(labels[i])
              y_pred.append(predictions[i])
    
    # TASK
    # gunakan fungsi confusion_matrix(y_true, y_pred) untuk membantu kalian menghitung
    # akurasi, presisi, recall, f1-score, sensitivity dan specificity
    #print(y_true)
    #print(y_pred)

    evals = confusion_matrix(y_true=y_true, y_pred=y_pred)
    tp, fp, tn, fn = evals[0][0], evals[1][0], evals[1][1], evals[0][1]
    accuration = (tp+tn)/(tp+tn+fp+fn)
    precission = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precission*recall/(precission+recall)
    sensitivity = recall
    specificity = tn/(fn+tn)

    print("accuration:", accuration)
    print("precission:", precission)
    print("recall:", recall)
    print("f1:", f1)
    print("sensitivity:", sensitivity)
    print("specificity:", specificity)

# TASK
model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5), stride=1, padding=1), # 1 conv
            nn.ReLU(inplace=True),  # 1 act
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            
            
            nn.Flatten(),
            nn.Linear(12800, 1024), # 7 fc
            nn.ReLU(inplace=True), # 7 act
            nn.Linear(1024, 2), # 8 fc
            nn.Softmax(dim=1)
            )

# uji coba performa model

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

model = train_model(model, criterion, optimizer, num_epochs=10)

test_model(model)
