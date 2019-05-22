from datasets.norb import smallNORB
from datasets.gtrsb import GTRSB
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import torch
from PIL import Image

# only for test
if __name__ == '__main__':

  path = os.path.join('./data', 'gtrsb')
  
  dataset = torch.utils.data.DataLoader(
    GTRSB(path, download=True, transform=transforms.Compose([ 
        transforms.Grayscale(),
        transforms.Resize((52,52), interpolation=Image.LANCZOS), 
        transforms.ToTensor()
      ])), 
    shuffle=True)
  
  # path = os.path.join('./data', 'smallNORB')
  # dataset = smallNORB(path, train=True, download=True,
  #               transform=transforms.Compose([
  #                   transforms.Resize(48),
  #                   transforms.RandomCrop(32),
  #                   transforms.ColorJitter(brightness=32./255, contrast=0.5),
  #                   transforms.ToTensor()
  #               ]))
  # i = 0
  for data, label in dataset:
    print(f"Data Shape: {str(data.shape)}")
    print(f"Label Shape: {str(label.shape)} {label}")
  
    plt.imshow(data.reshape((52,52)), cmap='gray')
    plt.show()
    i = i + 1
    if i > 2:
      break