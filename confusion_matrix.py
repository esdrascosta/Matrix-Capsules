import matplotlib.pyplot as plt
import numpy as np
from model import capsules
from datasets import GTRSB
from PIL import Image
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels

import os
pretrained_model = 'snapshots/_gray_model.pth'
use_cuda=True
path = os.path.join('./data', 'gtrsb')

# dataset
full_dataset = GTRSB(path, download=True, 
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((48, 48), interpolation=Image.LANCZOS), 
                transforms.ToTensor()
            ]))    
train_size = int(0.9 * len(full_dataset)) 
test_size = len(full_dataset) - train_size

print(f"Train Size: {str(train_size)}")
print(f"Val Size: {str(test_size)}")

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, 
                batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, 
                batch_size=1, shuffle=True)

device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
# model

num_class = 43
A, B, C, D = 64, 8, 16, 16
model = capsules(A=A, B=B, C=C, D=D, E=num_class,
                  iters=2).to(device)
model.load_state_dict(torch.load(pretrained_model))
model.eval()
label_pred = []
label_true = []
for data, target in test_loader:

  # Send the data and label to the device
  data, target = data.to(device), target.to(device)

  # Forward pass the data through the model
  output = model(data)
  label_pred.append(output.argmax(1).item())
  label_true.append(target.item())

pred = np.array(label_pred)
true = np.array(label_true)


def group_analysis():

  groups = {
    'speed':         np.array([0,1,2,3,4,5,7,8]),
    'prohibitions':  np.array([9,10,15,16]),
    'derestriction': np.array([6,32,41,42]),
    'mandatory':     np.array([33,34,35,36,37,38,39,40]),
    'danger':        np.array([11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]),
    'unique':        np.array([12,13,14,17]),
    'all': None,
  }

  for k, v in groups.items():
  
    gtrue = true if k == 'all' else np.isin(true, v) * 1
    gpred = pred if k == 'all' else np.isin(pred, v) * 1

    print("Acc group {}:  {:.4f}".format(k, accuracy_score(gtrue, gpred)))    

  

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(50,50))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
  
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = None
            if i != j and cm[i,j] > 0:
                color = 'red'
            else:
              color = "white" if cm[i, j] > thresh else "black"

            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color=color)
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)

group_analysis()

# Plot non-normalized confusion matrix
classes = np.array([str(i) for i in range(0,43)])
plot_confusion_matrix(true, pred, classes=classes, normalize=False,
                      title='Confusion matrix of '+str(len(test_loader))+ ' samples' )

plt.show()
