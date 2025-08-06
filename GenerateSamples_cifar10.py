import collections
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import time
import matplotlib.pyplot as plt

print(f"CUDA version: {torch.version.cuda}")
t0 = time.time()

##############################################################################
"""args for models"""

args = {}
args['dim_h'] = 64          # factor controlling size of hidden layers
args['n_channel'] = 3       # number of channels in the input data 
args['n_z'] = 300           # number of dimensions in latent space
args['dataset'] = 'cifar10' # specify which dataset to use

##############################################################################

# Same fixed encoder and decoder classes as in training
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Linear(self.dim_h * 8 * 2 * 2, self.n_z)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 4 * 4),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.n_channel, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 4, 4)
        x = self.deconv(x)
        return x

##############################################################################

def biased_get_class1(c):
    """Get samples belonging to class c"""
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    return xbeg, ybeg

def G_SM1(X, y, n_to_sample, cl):
    """Generate SMOTE samples in latent space"""
    # Fit nearest neighbors
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # Generate samples
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    # SMOTE interpolation
    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1),
                                   X_neighbor - X_base)

    return samples, [cl] * n_to_sample

def save_image_from_array(image_array, filename, output_dir="generated_images"):
    """Save image array as PNG file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Handle different image formats
    if len(image_array.shape) == 3 and image_array.shape[0] in [1, 3]:
        image_array = image_array.transpose((1, 2, 0))
    
    if len(image_array.shape) == 3 and image_array.shape[2] == 1:
        image_array = image_array.squeeze(axis=2)
    
    # Denormalize from [-1, 1] to [0, 1]
    if image_array.min() < 0:
        image_array = (image_array + 1) / 2
    
    # Clip values to [0, 1]
    image_array = np.clip(image_array, 0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(image_array, cmap='gray' if len(image_array.shape) == 2 else None)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
    plt.close()

#############################################################################

np.set_printoptions(precision=5, suppress=True)

# Data paths
dtrnimg = './CIFAR10/trn_img'
dtrnlab = './CIFAR10/trn_lab'

# Create output directories
os.makedirs('./CIFAR10/trn_img_f', exist_ok=True)
os.makedirs('./CIFAR10/trn_lab_f', exist_ok=True)

# Get file lists
ids = os.listdir(dtrnimg)
idtri_f = [os.path.join(dtrnimg, image_id) for image_id in ids]
print("Image files:", idtri_f)

ids_lab = os.listdir(dtrnlab)
idtrl_f = [os.path.join(dtrnlab, label_id) for label_id in ids_lab]
print("Label files:", idtrl_f)

# Model paths
modpth = './CIFAR10/models/crs5/'

encf = []
decf = []
for p in range(len(ids)):
    enc = os.path.join(modpth, str(p), 'bst_enc.pth')
    dec = os.path.join(modpth, str(p), 'bst_dec.pth')
    encf.append(enc)
    decf.append(dec)

# Process each fold
for m in range(len(idtri_f)):
    print(f"\n=== Processing fold {m} ===")
    
    # Load data
    trnimgfile = idtri_f[m]
    trnlabfile = idtrl_f[m]
    print(f"Loading: {trnimgfile}")
    print(f"Loading: {trnlabfile}")
    
    dec_x = np.loadtxt(trnimgfile) 
    dec_y = np.loadtxt(trnlabfile)

    print('Images before reshape:', dec_x.shape)
    print('Labels:', dec_y.shape)

    # Reshape and normalize
    dec_x = dec_x.reshape(dec_x.shape[0], 3, 32, 32)
    dec_x = (dec_x - 127.5) / 127.5  # Normalize to [-1, 1]
    
    print('Class distribution:', collections.Counter(dec_y))
    print('Images after reshape:', dec_x.shape)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    path_enc = encf[m]
    path_dec = decf[m]

    if not os.path.exists(path_enc) or not os.path.exists(path_dec):
        print(f"Model files not found: {path_enc}, {path_dec}")
        continue

    encoder = Encoder(args)
    encoder.load_state_dict(torch.load(path_enc, map_location=device))
    encoder = encoder.to(device)

    decoder = Decoder(args)
    decoder.load_state_dict(torch.load(path_dec, map_location=device))
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    # Define imbalance ratios (target samples per class)
    imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

    resx = []
    resy = []

    # Generate samples for classes 1-9 (class 0 already has enough samples)
    for i in range(1, 10):
        print(f"\nProcessing class {i} ({classes[i]})...")
        
        xclass_orig, yclass = biased_get_class1(i)
        
        if len(xclass_orig) == 0:
            print(f"No samples found for class {i}")
            continue
            
        print(f"Original samples: {xclass_orig.shape}")
        print(f"Class label: {yclass[0]}")
            
        # Encode to latent space
        with torch.no_grad():
            xclass = torch.Tensor(xclass_orig).to(device)
            xclass_encoded = encoder(xclass)
            xclass_encoded = xclass_encoded.detach().cpu().numpy()
        
        print(f"Encoded shape: {xclass_encoded.shape}")
        
        # Calculate number of samples to generate
        n = imbal[0] - imbal[i]
        if n <= 0:
            continue
            
        print(f"Generating {n} samples...")
        
        # Generate SMOTE samples in latent space
        xsamp, ysamp = G_SM1(xclass_encoded, yclass, n, i)
        print(f"SMOTE samples shape: {xsamp.shape}")
        print(f"SMOTE labels: {len(ysamp)}")
        
        ysamp = np.array(ysamp)
    
        # Decode to image space
        with torch.no_grad():
            xsamp_tensor = torch.Tensor(xsamp).to(device)
            ximg = decoder(xsamp_tensor)
            ximn = ximg.detach().cpu().numpy()
        
        print(f"Generated images shape: {ximn.shape}")
        
        resx.append(ximn)
        resy.append(ysamp)

        # Save some sample images for verification
        if i == 1:  # Only for first minority class
            print("Saving sample images...")
            for j in range(min(5, len(xclass_orig), len(ximn))):
                # Original image
                original_image = xclass_orig[j]
                save_image_from_array(original_image, f"original_class_{i}_img_{j}.png")

                # Generated image
                generated_image = ximn[j]
                save_image_from_array(generated_image, f"generated_class_{i}_img_{j}.png")
    
    if len(resx) == 0:
        print("No samples were generated!")
        continue
        
    # Combine all generated samples
    resx1 = np.vstack(resx)
    resy1 = np.hstack(resy)
    
    print(f"Total generated samples: {resx1.shape}")
    print(f"Total generated labels: {resy1.shape}")

    # Flatten images for saving
    resx1 = resx1.reshape(resx1.shape[0], -1)
    print(f"Flattened generated samples: {resx1.shape}")
    
    # Denormalize back to [0, 255] for saving
    resx1 = ((resx1 + 1) * 127.5).astype(np.uint8)
    
    # Prepare original data
    dec_x1 = dec_x.reshape(dec_x.shape[0], -1)
    dec_x1 = ((dec_x1 + 1) * 127.5).astype(np.uint8)
    
    print(f"Original data shape: {dec_x1.shape}")
    
    # Combine original and generated data
    combx = np.vstack((resx1, dec_x1))
    comby = np.hstack((resy1, dec_y))

    print(f"Combined data shape: {combx.shape}")
    print(f"Combined labels shape: {comby.shape}")
    print(f"Final class distribution: {collections.Counter(comby)}")

    # Save combined data
    ifile = os.path.join('./CIFAR10/trn_img_f', f'{m}_trn_img.txt')
    lfile = os.path.join('./CIFAR10/trn_lab_f', f'{m}_trn_lab.txt')
    
    np.savetxt(ifile, combx, fmt='%d')
    np.savetxt(lfile, comby, fmt='%d')
    
    print(f"Saved to: {ifile}")
    print(f"Saved to: {lfile}")

t1 = time.time()
print(f'\nTotal execution time: {(t1 - t0)/60:.2f} minutes')
print("Sample generation completed successfully!")
