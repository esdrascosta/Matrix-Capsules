import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.transform import resize
from six.moves import urllib
from PIL import Image
import numpy as np
from os.path import join
import os.path
import zipfile
import errno
import torch
import csv
import os

class GTRSB(data.Dataset):

  raw_folder = 'raw'
  processed_folder = 'processed'
  dataset_file = 'dataset.pt'
  urls = {
    'train_data': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip',
    'test_data': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip',
    'test_info': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip'
  }

  def __init__(self, root, train=True, transform=None, target_transform=None,
   download=False):

    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform

    if download:
      self.download()  

    self.dataset, self.labels = torch.load(join(self.root, self.processed_folder, self.dataset_file))

    print(f"Dataset Loaded: {len(self.dataset)}")

  def __getitem__(self, index):
    """
    Args:
      index (int): Index
    Returns:
      tuple: (image, target)
      where target is index of the target class and info contains
    """

    img = self.dataset[index]
    label = self.labels[index]
    
    img = Image.fromarray(img)
    if self.transform is not None:
        img = self.transform(img)

    if self.target_transform is not None:
        label = self.target_transform(label)

    return img, label

  def __len__(self):
    return len(self.dataset)

  def _check_processed_exists(self):
    return os.path.exists(join(self.root, self.processed_folder, self.dataset_file))

  def download(self):

    if self._check_processed_exists():
      return  

    # download files
    try:
      os.makedirs(join(self.root, self.raw_folder))
      os.makedirs(join(self.root, self.processed_folder))
    except OSError as e:
      if e.errno == errno.EEXIST:
        pass
      else:
        raise
    
    self._download_files()
    self._extract_files()

    # process and save as torch files
    print('Processing...')
    # Join training and test dataset for future split
    train_data_folder = self.urls['train_data'].rpartition('/')[2].replace('.zip','')
    train_data, train_labels = self._parse_train_file(
      join(self.root, self.raw_folder,train_data_folder, 
        'GTSRB', 'Final_Training','Images'))

    print('Train Size '+str(len(train_data)))

    test_data_folder = self.urls['test_data'].rpartition('/')[2].replace('.zip','')
    test_info_folder = self.urls['test_info'].rpartition('/')[2].replace('.zip','')

    test_data, test_labels = self._parse_test_file(
      join(self.root, self.raw_folder, test_data_folder, 
        'GTSRB', 'Final_Test', 'Images'),
      join(self.root, self.raw_folder, test_info_folder))

    print('Test Size '+str(len(test_data)))

    data_images = train_data + test_data
    labels = train_labels + test_labels
    dataset = (
      data_images,
      labels
    )

    with open(join(self.root, self.processed_folder, self.dataset_file), 'wb') as f:
            torch.save(dataset, f)

  def _parse_train_file(self, rootpath):
    """
    Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels
    """

    images = [] 
    labels = [] 
    # loop over all 42 classes
    for c in range(0,43):
      prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
      gt_file = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
      gt_reader = csv.reader(gt_file, delimiter=';') # csv parser for annotations file
      next(gt_reader, None) # skip header
      # loop over all images in current annotations file
      for row in gt_reader:
          img = plt.imread(prefix + row[0])
          images.append(img) # the 1th column is the filename
          labels.append(int(row[7])) # the 8th column is the label
      gt_file.close()
    
    return images, labels

  def _parse_test_file(self, rootpath_data, rootpath_labels):
    """
    Reads traffic sign data, from test dataset, for German Traffic Sign Recognition Benchmark.
    
    Arguments:
      rootpath_data -- path to the traffic sign data
      rootpath_labels -- path to the traffic sign labels
    Returns:   list of images, list of corresponding labels
    """
    csv_file_name = 'GT-final_test.csv'
    gt_file = open(join(rootpath_labels, csv_file_name))

    gt_reader = csv.reader(gt_file, delimiter=';') 
    next(gt_reader, None) # skip header
    
    images = []
    labels = []
    # loop over all images in annotations file
    for row in gt_reader:
      # load image from data path folder
      img = plt.imread(join(rootpath_data, row[0]))
      images.append(img)
      labels.append(int(row[7])) # the 8th column is the label

    gt_file.close()

    return images, labels

  def _extract_files(self):
    for key, url in self.urls.items():
      filename = url.rpartition('/')[2]
      file_path = join(self.root, self.raw_folder, filename)

      if os.path.exists(file_path) and\
        not os.path.exists(file_path.replace('.zip', '')):

        print('Extracting '+filename) 
        with zipfile.ZipFile(file_path) as zip_f:
          zip_f.extractall(file_path.replace('.zip', ''))
        os.unlink(file_path)

  def _download_files(self):
    for key, url in self.urls.items():
      filename = url.rpartition('/')[2]
      file_path = join(self.root, self.raw_folder, filename)

      if not os.path.exists(file_path) and\
        not os.path.exists(file_path.replace('.zip', '')):

        print('Downloading '+url)
        data = urllib.request.urlopen(url)  
        with open(file_path, 'wb') as f:
          f.write(data.read())
  