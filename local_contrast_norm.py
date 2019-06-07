import torch
import torch.nn.functional as F
from torchvision.transforms import functional as IF
import numpy as np
from scipy.signal.signaltools import convolve2d
from PIL import Image
import matplotlib.pyplot as plt

class LocalContrastNorm(object):

  def __init__(self, radius=9):
    self.radius=radius

  def __call__(self, image):
    image = torch.Tensor([np.array(image).transpose((2,0,1))])
    radius=self.radius
    if radius%2 == 0:
        radius += 1
    def get_gaussian_filter(kernel_shape):
        x = np.zeros(kernel_shape, dtype='float64')
 
        def gauss(x, y, sigma=2.0):
            Z = 2 * np.pi * sigma ** 2
            return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))
 
        mid = np.floor(kernel_shape[-1] / 2.)
        for kernel_idx in range(0, kernel_shape[1]):
            for i in range(0, kernel_shape[2]):
                for j in range(0, kernel_shape[3]):
                    x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)
 
        return x / np.sum(x)
    
    n,c,h,w = image.shape[0],image.shape[1],image.shape[2],image.shape[3]

    gaussian_filter = torch.Tensor(get_gaussian_filter((1,c,radius,radius)))
    filtered_out = F.conv2d(image,gaussian_filter,padding=radius-1)
    mid = int(np.floor(gaussian_filter.shape[2] / 2.))
    ### Subtractive Normalization
    centered_image = image - filtered_out[:,:,mid:-mid,mid:-mid]
    
    ## Variance Calc
    sum_sqr_image = F.conv2d(centered_image.pow(2),gaussian_filter,padding=radius-1)
    s_deviation = sum_sqr_image[:,:,mid:-mid,mid:-mid].sqrt()
    per_img_mean = s_deviation.mean()
    
    ## Divisive Normalization
    divisor = np.maximum(per_img_mean.numpy(),s_deviation.numpy())
    divisor = np.maximum(divisor, 1e-4)
    new_image = centered_image / torch.Tensor(divisor)

    ret = new_image[0].numpy().transpose((1,2,0))
    # just for test
    # ret = (ret - ret.min())/(ret.max() - ret.min()) * 255
    
    return Image.fromarray(ret.astype(np.uint8))
if __name__ == '__main__':

  image = plt.imread('image_samples/img2.jpg')
  plt.imshow(image)
  plt.show()

  norm = LocalContrastNorm(7)
  ret = norm(image)

  plt.imshow(ret)
  plt.show()