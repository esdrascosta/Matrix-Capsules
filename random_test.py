from datasets.norb import smallNORB
from datasets.gtrsb import GTRSB
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import torch
from PIL import Image
from local_contrast_norm import LocalContrastNorm
# only for test
if __name__ == '__main__':

  path = os.path.join('./data', 'gtrsb')

  dataset = torch.utils.data.DataLoader(
    GTRSB(path, download=True, transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48,48), interpolation=Image.LANCZOS), 
        transforms.ToTensor()
      ])), 
    shuffle=True)
  i=0
  for data, label in dataset:
    print(f"Data Shape: {str(data.shape)}")
    print(f"Label Shape: {str(label.shape)} {label}")

    data = data[0,0,:,:].numpy()
    # data = data[0].permute(1,2,0)
    # data = (data - data.min())/(data.max() - data.min()) * 255
    # plt.imshow(data)
    plt.hist(data.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.show()
    i = i + 1
    if i > 2:
      break