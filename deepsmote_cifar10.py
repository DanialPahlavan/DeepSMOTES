import collections
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os

print(torch.version.cuda) #10.1
t3 = time.time()
##############################################################################
"""args for AE"""

args = {}
args['dim_h'] = 64         # factor controlling size of hidden layers
args['n_channel'] = 3      # number of channels in the input data 

args['n_z'] = 300          # number of dimensions in latent space

args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.01      # hyper param for weight of discriminator loss
args['lr'] = 0.0002        # learning rate for Adam optimizer
args['epochs'] = 50        # how many epochs to run for
args['batch_size'] = 100   # batch size for SGD
args['save'] = True        # save weights at each epoch of training if True
args['train'] = True       # train networks if True, else load networks from

args['dataset'] = 'cifar10'  # specify which dataset to use

##############################################################################

## Fixed encoder model for CIFAR-10
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        # Convolutional layers for 32x32 input
        self.conv = nn.Sequential(
            # Input: 3x32x32
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),      # -> 64x16x16
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),      # -> 128x8x8
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),  # -> 256x4x4
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),  # -> 512x2x2
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Fixed: Correct size calculation for CIFAR-10
        self.fc = nn.Linear(self.dim_h * 8 * 2 * 2, self.n_z)  # 512 * 4 = 2048

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


## Fixed decoder model for CIFAR-10
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # Fixed: Correct size for CIFAR-10 (4x4 not 7x7)
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 4 * 4),
            nn.ReLU()
        )

        # Deconvolutional layers to reconstruct 32x32 image
        self.deconv = nn.Sequential(
            # Input: 512x4x4
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1),     # -> 256x8x8
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1),     # -> 128x16x16
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.dim_h * 2, self.n_channel, 4, 2, 1),     # -> 3x32x32
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 4, 4)
        x = self.deconv(x)
        return x

##############################################################################
"""set models, loss functions"""

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

##############################################################################
"""functions to create SMOTE images"""

def biased_get_class(c):
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    return xbeg, ybeg

def G_SM(X, y, n_to_sample, cl):
    # SMOTE algorithm implementation
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # Generate samples
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1),
                                   X_neighbor - X_base)

    return samples, [cl] * n_to_sample

###############################################################################

# File paths
dtrnimg = './CIFAR10/trn_img/'
dtrnlab = './CIFAR10/trn_lab/'

# Create directories if they don't exist
os.makedirs(dtrnimg, exist_ok=True)
os.makedirs(dtrnlab, exist_ok=True)
os.makedirs('./CIFAR10/models/crs5', exist_ok=True)

ids = os.listdir(dtrnimg)
idtri_f = [os.path.join(dtrnimg, image_id) for image_id in ids]
print("Training image files:", idtri_f)

ids = os.listdir(dtrnlab)
idtrl_f = [os.path.join(dtrnlab, image_id) for image_id in ids]
print("Training label files:", idtrl_f)

for i in range(len(ids)):
    print(f"\n=== Processing fold {i} ===")
    
    # Initialize models
    encoder = Encoder(args)
    decoder = Decoder(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Loss function
    criterion = nn.MSELoss().to(device)
    
    # Load data
    trnimgfile = idtri_f[i]
    trnlabfile = idtrl_f[i]
    
    print(f"Loading: {trnimgfile}")
    print(f"Loading: {trnlabfile}")
    
    dec_x = np.loadtxt(trnimgfile) 
    dec_y = np.loadtxt(trnlabfile)

    print('Images before reshape:', dec_x.shape) 
    print('Labels:', dec_y.shape) 
    print('Class distribution:', collections.Counter(dec_y))
    
    # Reshape for CIFAR-10: (N, 3072) -> (N, 3, 32, 32)
    dec_x = dec_x.reshape(dec_x.shape[0], 3, 32, 32)   
    print('Images after reshape:', dec_x.shape)
    
    # Normalize to [-1, 1] for tanh output
    dec_x = (dec_x - 127.5) / 127.5

    # Create DataLoader
    tensor_x = torch.Tensor(dec_x)
    tensor_y = torch.tensor(dec_y, dtype=torch.long)
    cifar10_dataset = TensorDataset(tensor_x, tensor_y) 
    train_loader = torch.utils.data.DataLoader(
        cifar10_dataset, 
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    best_loss = np.inf
    t0 = time.time()
    
    if args['train']:
        # Optimizers
        enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'])
        dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'])
    
        for epoch in range(args['epochs']):
            train_loss = 0.0
            tmse_loss = 0.0
            tdiscr_loss = 0.0
            
            # Set to training mode
            encoder.train()
            decoder.train()
        
            for batch_idx, (images, labs) in enumerate(train_loader):
                # Zero gradients
                encoder.zero_grad()
                decoder.zero_grad()
                
                images, labs = images.to(device), labs.to(device)
                
                # Forward pass - reconstruction loss
                z_hat = encoder(images)
                x_hat = decoder(z_hat)
                mse = criterion(x_hat, images)
                
                # SMOTE-based regularization
                tc = np.random.choice(10, 1)[0]
                
                # Get samples from random class
                class_mask = (dec_y == tc)
                if np.sum(class_mask) > 1:  # Need at least 2 samples
                    xbeg = dec_x[class_mask]
                    xlen = len(xbeg)
                    nsamp = min(xlen, 50)  # Reduced for efficiency
                    
                    if nsamp > 1:
                        ind = np.random.choice(list(range(len(xbeg))), nsamp, replace=False)
                        xclass = xbeg[ind]
                        
                        xclen = len(xclass)
                        xcminus = np.arange(1, xclen)
                        xcplus = np.append(xcminus, 0)
                        
                        # Create shifted version for consistency regularization
                        xcnew = xclass[xcplus]
                        
                        # Convert to tensors
                        xclass = torch.Tensor(xclass).to(device)
                        xcnew = torch.Tensor(xcnew).to(device)
                        
                        # Encode and decode
                        xclass_encoded = encoder(xclass)
                        xclass_encoded_shifted = encoder(xcnew)
                        
                        # Decode both
                        ximg = decoder(xclass_encoded)
                        ximg_shifted = decoder(xclass_encoded_shifted)
                        
                        # Consistency loss
                        mse2 = criterion(ximg, xclass) + criterion(ximg_shifted, xcnew)
                    else:
                        mse2 = torch.tensor(0.0).to(device)
                else:
                    mse2 = torch.tensor(0.0).to(device)
                
                # Combined loss
                comb_loss = mse + 0.1 * mse2  # Reduced weight for regularization
                comb_loss.backward()
                
                # Update parameters
                enc_optim.step()
                dec_optim.step()
                
                # Accumulate losses
                train_loss += comb_loss.item() * images.size(0)
                tmse_loss += mse.item() * images.size(0)
                tdiscr_loss += mse2.item() * images.size(0)
            
            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            tmse_loss = tmse_loss / len(train_loader.dataset)
            tdiscr_loss = tdiscr_loss / len(train_loader.dataset)
            
            print(f'Epoch: {epoch:3d} | Train Loss: {train_loss:.6f} | '
                  f'MSE: {tmse_loss:.6f} | Reg: {tdiscr_loss:.6f}')
            
            # Save best model
            if train_loss < best_loss:
                print('Saving best model...')
                path_enc = f'./CIFAR10/models/crs5/{i}/bst_enc.pth'
                path_dec = f'./CIFAR10/models/crs5/{i}/bst_dec.pth'
                
                os.makedirs(os.path.dirname(path_enc), exist_ok=True)
                
                torch.save(encoder.state_dict(), path_enc)
                torch.save(decoder.state_dict(), path_dec)
                
                best_loss = train_loss
        
        # Save final model
        path_enc = f'./CIFAR10/models/crs5/{i}/f_enc.pth'
        path_dec = f'./CIFAR10/models/crs5/{i}/f_dec.pth'
        
        torch.save(encoder.state_dict(), path_enc)
        torch.save(decoder.state_dict(), path_dec)
        
        print(f'Models saved to:\n{path_enc}\n{path_dec}')
              
    t1 = time.time()
    print(f'Fold {i} completed in {(t1 - t0)/60:.2f} minutes')

t4 = time.time()
print(f'\nTotal execution time: {(t4 - t3)/60:.2f} minutes')
