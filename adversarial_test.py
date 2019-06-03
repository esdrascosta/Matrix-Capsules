from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model import capsules
from datasets import GTRSB
from PIL import Image

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = 'snapshots/model_29.pth'
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


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
   
    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

if __name__ == '__main__':
  accuracies = []
  examples = []

  # Run test for each epsilon
  for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)
