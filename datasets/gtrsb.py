from __future__ import print_function
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch

class GTRSB(data.Dataset):

  raw_folder = 'raw'
  processed_folder = 'processed'
  training_file = 'training.pt'
  test_file = 'test.pt'
  urls = {
    # 'train_data': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip',
    # 'test_data': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip',
    'test_info': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip'
  }

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.train = train  # training set or test set

    if download:
      self.download()  

  def __getitem__(self, index):
    """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
            where target is index of the target class and info contains
        """
    pass

  def __len__(self):
    pass

  def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

  def download(self):
    from six.moves import urllib
    import zipfile

    if self._check_exists():
      return

    # download files
    try:
        os.makedirs(os.path.join(self.root, self.raw_folder))
        os.makedirs(os.path.join(self.root, self.processed_folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    for t, url in self.urls.items():
      print('Downloading '+url)
      data = urllib.request.urlopen(url)
      filename = url.rpartition('/')[2]
      file_path = os.path.join(self.root, self.raw_folder, filename)
      with open(file_path, 'wb') as f:
        f.write(data.read())
      with zipfile.ZipFile(file_path) as zip_f:
        zip_f.extractall(file_path.replace('.zip', ''))
      os.unlink(file_path)

      # process and save as torch files
      print('Processing...')

  def parse_train_file(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels

# only for test
if __name__ == '__main__':
  path = os.path.join('./data', 'gtrsb')
  GTRSB(path, download=True)    
