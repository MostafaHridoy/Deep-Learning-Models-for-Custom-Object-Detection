Swin Tranformer with k-fold Cross Validation:

!git clone https://github.com/mrdbourke/pytorch-deep-learning
!mv pytorch-deep-learning/going_modular .
!mv pytorch-deep-learning/helper_functions.py .
!rm -rf pytorch-deep-learning

!pip install timm

from going_modular.going_modular import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves

import numpy as np
import os
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models
import timm
from timm.loss import LabelSmoothingCrossEntropy
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
%matplotlib inline
import sys
from tqdm import tqdm
import time
import copy

def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes

def get_data_loaders(data_dir, batch_size, train = False):
    if train:
        #train
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            T.RandomErasing(p=0.1, value='random')
        ])
        train_data = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform = transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader, len(train_data)
    else:
        # val/test
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
        ])
        val_data = datasets.ImageFolder(os.path.join(data_dir, "valid/"), transform=transform)
        test_data = datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return val_loader, test_loader, len(val_data), len(test_data)

(train_loader, train_data_len) = get_data_loaders(dataset_path, 32, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(dataset_path, 32, train=False)

dataloaders = {
    "train": train_loader,
    "val": val_loader
}
dataset_sizes = {
    "train": train_data_len,
    "val": valid_data_len
}

classes = get_classes("Location")
print(classes, len(classes))


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim

'''def create_model():
    HUB_URL = "SharanSMenon/swin-transformer-hub:main"
    MODEL_NAME = "swin_tiny_patch4_window7_224"
    model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.head.in_features
    custom_head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(classes))
    )
    model.head = custom_head
    model = model.to(device)
    return model'''

num_folds = 5

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

best_models = []
fold_precisions = []
fold_recalls = []
fold_f1_scores = []
accuracies=[]

batch_size = 32
dataset_path = "/content/vision_transformer"
(train_loader, train_data_len) = get_data_loaders(dataset_path, batch_size, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(dataset_path, batch_size, train=False)

model = create_model()

criterion = LabelSmoothingCrossEntropy()
criterion = criterion.to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

for fold, (train_indices, val_indices) in enumerate(kf.split(range(train_data_len))):
    print(f'Fold {fold + 1}/{num_folds}')
    print("-" * 10)

    train_data = Subset(train_loader.dataset, train_indices)
    val_data = Subset(train_loader.dataset, val_indices)

    train_loader_fold = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader_fold = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    best_acc = 0.0

    for epoch in range(30):  
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader_fold):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader_fold.dataset)

        model.eval()
        running_corrects = 0.0
        for inputs, labels in tqdm(val_loader_fold):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        epoch_acc = running_corrects.double() / len(val_loader_fold.dataset)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    precision = precision_score(labels.cpu(), preds.cpu(), average='weighted')
    recall = recall_score(labels.cpu(), preds.cpu(), average='weighted')
    f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
    accuracy = accuracy_score(labels.cpu(), preds.cpu())


    accuracies.append(accuracy)
    fold_precisions.append(precision)
    fold_recalls.append(recall)
    fold_f1_scores.append(f1)

average_precision = np.mean(fold_precisions)
average_accuracy= np.mean(accuracies)
average_recall = np.mean(fold_recalls)
average_f1_score = np.mean(fold_f1_scores)
best_fold_index = np.argmax(fold_f1_scores)

print("Average Precision: {:.4f}".format(average_precision))
print("Average Recall: {:.4f}".format(average_recall))
print("Average F1 Score: {:.4f}".format(average_f1_score))
print("Average Accuracy Score : {:.4f}".format(average_accuracy))
#print("Best Fold (based on F1 score): {}".format(best_fold_index + 1))

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))


num_folds = 5


metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
average_scores = [average_accuracy, average_precision, average_recall, average_f1_score]


plt.plot(metrics, average_scores, marker='o', color='b', label='Average Scores')
for i, metric_value in enumerate(average_scores):
    plt.text(metrics[i], metric_value, f'{metric_value:.2f}', ha='center', va='bottom')


for fold_idx in range(num_folds):
    fold_scores = [accuracies[fold_idx], fold_precisions[fold_idx], fold_recalls[fold_idx], fold_f1_scores[fold_idx]]
    plt.plot(metrics, fold_scores, marker='o', linestyle='--', label=f'Fold {fold_idx + 1}')
    for i, metric_value in enumerate(fold_scores):
        plt.text(metrics[i], metric_value, f'{metric_value:.2f}', ha='center', va='bottom')

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Metrics Across Folds')
plt.legend()
plt.grid(True)
plt.show()

test_loss = 0.0
class_correct = list(0 for i in range(len(classes)))
class_total = list(0 for i in range(len(classes)))
model_ft.eval()

for data, target in tqdm(test_loader):
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        output = model_ft(data)
        loss = criterion(output, target)
    test_loss = loss.item() * data.size(0)
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    if len(target) == 32:
        for i in range(32):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

test_loss = test_loss / test_data_len
print('Test Loss: {:.4f}'.format(test_loss))
for i in range(len(classes)):
    if class_total[i] > 0:
        print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (
            classes[i], 100*class_correct[i]/class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])
        ))
    else:
        print("Test accuracy of %5s: NA" % (classes[i]))
print("Test Accuracy of %2d%% (%2d/%2d)" % (
            100*np.sum(class_correct)/np.sum(class_total), np.sum(class_correct), np.sum(class_total)
        ))

import requests

from going_modular.going_modular.predictions import pred_and_plot_image

custom_image_path = "/content/spd.jpg"
pred_and_plot_image(model=model_ft,
                    image_path=custom_image_path,
                    class_names=classes)

all_true_labels = []
all_probabilities = []

#model_ft.eval()
with torch.no_grad():
    for data, target in tqdm(test_loader):  # Use tqdm for progress visualization
        data, target = data.to(device), target.to(device)
        output = model_ft(data)
        all_true_labels.extend(target.cpu().numpy())
        all_probabilities.extend(F.softmax(output, dim=1).cpu().numpy())  # Store the predicted class probabilities


all_true_labels = np.array(all_true_labels)
all_probabilities = np.array(all_probabilities)

all_predictions = np.argmax(all_probabilities, axis=1)
f1_scores = []
for i in range(len(classes)):
    f1 = f1_score(all_true_labels == i, all_predictions == i)
    f1_scores.append(f1)


bar_colors = ['red', 'purple']

plt.figure(figsize=(10, 8))
bars = plt.bar(classes, f1_scores, color=bar_colors)
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title('F1-Score for Each Class')

for bar, f1 in zip(bars, f1_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{f1:.2f}', ha='center', va='bottom')

plt.show()

import seaborn as sns


# Function to plot confusion matrix using Seaborn
def plot_confusion_matrix_sns(y_true, y_pred, classes, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="rocket", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

# Get true labels and predicted labels for the test dataset
all_true_labels = []
all_predicted_labels = []

model_ft.eval()
with torch.no_grad():
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        output = model_ft(data)
        _, predicted = torch.max(output, 1)  # Get the predicted class labels
        all_true_labels.extend(target.cpu().numpy())
        all_predicted_labels.extend(predicted.cpu().numpy())

# Convert to NumPy arrays
all_true_labels = np.array(all_true_labels)
all_predicted_labels = np.array(all_predicted_labels)

# Calculate and plot confusion matrix using Seaborn
plot_confusion_matrix_sns(all_true_labels, all_predicted_labels, classes)

print(classification_report(all_true_labels, all_predicted_labels, target_names=classes))



