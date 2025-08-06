import torchvision
import numpy as np
import os

# Create directories if they don't exist
os.makedirs('./CIFAR10/trn_img', exist_ok=True)
os.makedirs('./CIFAR10/trn_lab', exist_ok=True)

# Download the CIFAR-10 dataset
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# Extract images and labels
images = cifar10_dataset.data
labels = np.array(cifar10_dataset.targets)

# Transpose and reshape the data
images_transposed = images.transpose((0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
images_reshaped = images_transposed.reshape(images.shape[0], -1)

# Save the images and labels to text files
np.savetxt('./CIFAR10/trn_img/0_trn_img.txt', images_reshaped, fmt='%d')
np.savetxt('./CIFAR10/trn_lab/0_trn_lab.txt', labels, fmt='%d')

print("CIFAR-10 data has been downloaded and saved in the required format.")
