import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

# Dataloader

train_path = "training data path"
val_path = "validation data path"
test_path = "testing data path"

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform = train_transformer),
    batch_size = 64, shuffle = True, num_workers = 2
)
val_loader = DataLoader(
    torchvision.datasets.ImageFolder(val_path, transform = val_test_transformer),
    batch_size = 32, shuffle = False , num_workers = 2
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform = val_test_transformer),
    batch_size = 32, shuffle = False , num_workers = 2
)

# Labels
root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)


model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
# Freeze all layers except the final classification layer
for name, param in model.named_parameters():
    if "fc" in name:  # Unfreeze the final classification layer
        param.requires_grad = True
    else:
        param.requires_grad = False

# Define the loss function and optimizer
loss_function = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)  # Use all parameters


# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Calculating size of training and testing images
train_count = len(glob.glob(train_path + '/**/*.jpg'))
val_count = len(glob.glob(val_path + '/**/*.jpg'))
test_count = len(glob.glob(test_path + '/**/*.jpg'))
print(train_count,val_count,test_count)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

best_accuracy = 0.0
num_epochs = 10

for epoch in range(num_epochs):
    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).type(torch.float)  # Ensure labels are float type
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        outputs = torch.sigmoid(outputs).squeeze()  # Apply sigmoid activation function

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        predicted = (outputs >= 0.5).to(torch.float)  # Convert probabilities to binary predictions
        train_accuracy += torch.sum(predicted == labels).item()

    train_accuracy = train_accuracy / len(train_loader.dataset)
    train_loss = train_loss / len(train_loader.dataset)

    # Evaluation on validation dataset
    model.eval()  # Set the model to evaluation mode

    val_accuracy = 0.0
    val_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation during validation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).type(torch.float)  # Ensure labels are float type
            labels = labels.to(device)

            outputs = model(images)
            outputs = torch.sigmoid(outputs).squeeze()  # Apply sigmoid activation function

            loss = loss_function(outputs, labels)
            val_loss += loss.item() * images.size(0)

            predicted = (outputs >= 0.5).to(torch.float)  # Convert probabilities to binary predictions
            val_accuracy += torch.sum(predicted == labels).item()

    val_accuracy = val_accuracy / len(val_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Validation Accuracy: ' + str(val_accuracy))


    # Save the best model
    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'try2_model.pth')
        best_accuracy = val_accuracy


from torchvision import models
import torchmetrics

# Load the saved model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)  # Adjust to match the original model's output units
model.load_state_dict(torch.load('saved model path', map_location=torch.device('cpu')))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Instantiate metrics
confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=2, task="binary").to(device)

# Iterate through validation dataset
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(images)

    # Compute predictions
    preds = torch.sigmoid(outputs)

    # Convert predictions to binary (0 or 1)
    binary_preds = torch.round(preds).squeeze().long()

    # Update confusion matrix
    confusion_matrix.update(binary_preds, labels)

# Print confusion matrix
print(f"Confusion Matrix:\n{confusion_matrix.compute()}")

# Accuracy precision recall

# Our confusion matrix
confusion_matrix = [[699, 301],
                    [514, 287]]


def calculate_accuracy(confusion_matrix):
    TN, FP = confusion_matrix[0]
    FN, TP = confusion_matrix[1]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

def calculate_recall(confusion_matrix):
    TN, FP = confusion_matrix[0]
    FN, TP = confusion_matrix[1]
    recall = TP / (TP + FN)
    return recall

def calculate_precision(confusion_matrix):
    TN, FP = confusion_matrix[0]
    FN, TP = confusion_matrix[1]
    precision = TP / (TP + FP)
    return precision

def get_f1(precision, recall):
    f1_val = 2*(precision*recall)/(precision+recall)
    return f1_val

# Respective evaluation criteria

recall = calculate_recall(confusion_matrix)
precision = calculate_precision(confusion_matrix)
accuracy = calculate_accuracy(confusion_matrix)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 score:",get_f1(precision,recall))

