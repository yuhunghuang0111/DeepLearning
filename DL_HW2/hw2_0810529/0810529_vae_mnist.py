"""
Code for the VAE (Section 2 of the problem)
"""
#%%
from re import A
import numpy as np
mnist_data = np.load('TibetanMNIST.npz')
print(mnist_data['image'].shape)  #(12000, 28, 28)
threshold = 127
binary = np.where(mnist_data['image'] > threshold, 1, 0) 
print(binary[0])

#%%
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
batch_size = 64
class TrainDataset(Dataset):
    
    def __init__(self, image):
        self.image = image
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        image = self.transform((self.image[idx])).float()
        image = torch.flatten(image)
        return image.to(device)
train_data = TrainDataset(binary)
print(len(train_data))
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# %%
from torch import nn
import torch.optim as optim
#%%
class MnistVAE(nn.Module):
    def __init__(self):
        super(MnistVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 196),
            nn.BatchNorm1d(196),
            nn.ReLU(),
            nn.Linear(196, 49),
            nn.BatchNorm1d(49),
            nn.ReLU(),
            )
        self.enc_mu = nn.Linear(49, 2)
        self.enc_sig = nn.Linear(49, 2)
        self.decoder = nn.Sequential(
            nn.Linear(2, 49),
            nn.ReLU(),
            nn.Linear(49, 196),
            nn.ReLU(),
            nn.Linear(196, 784),
            nn.Sigmoid(),
        )
        


    def encode(self, x):
        y = self.encoder(x)
        return self.enc_mu(y), self.enc_sig(y)
    def decode(self, z):
        x = self.decoder(z)
        return x # BCE loss

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        kl_loss = (-0.5*(1 + logvar - mu**2 - torch.exp(logvar)).sum(dim=1)).mean(dim=0)
        return z, kl_loss
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) #  mu + eps * std
    
    def latent(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return z
        
        
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
LAMBDA = 100
learning_rate = 1e-3
def train(epochs):
    train_loss_list=[]
    net = MnistVAE()
    criterion = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.to(device)   
    for epoch in range(epochs):
        training_loss = 0.0
        net.train()
        for batch_idx, image in enumerate(train_dataloader):
            optimizer.zero_grad()
            out, kl_loss = net(image)
            recon_loss = criterion(out, image)
            loss = kl_loss*LAMBDA + recon_loss
            loss.backward()
            training_loss += loss.item()
            optimizer.step()

            
        avg_loss = training_loss/(12000)
        train_loss_list.append(avg_loss)
        print('\n'+'Epoch: {},  Train. Loss: {}'.format(epoch + 1, avg_loss))
    print('Finished Training')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_loss_list, label="Train")
    plt.title("Average Loss History")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()
    return net
net = train(100) # train 100 epochs
# %%
from torchvision.utils import save_image
LAT_DIM = 2
DIR_OUT = "results/mnist"
def interp_vectors(start, stop, ncols):
    steps = (1.0/(ncols-1)) * (stop - start)
    return np.array([(start + steps*x) for x in range(ncols)])

def generateImg(net, randn_mult=4, nrows=8):
    # real samples
    real = torch.from_numpy(binary[0:64]).to(device).float()  # (64, 28, 28)
    save_image(real.reshape(-1, 1, 28, 28), f"{DIR_OUT}/lambda_{LAMBDA}_real.png", nrow=8)
    
    # resconstruction
    real_cons = net.forward(real)[0]
    save_image(real_cons.reshape(-1, 1, 28, 28), f"{DIR_OUT}/lambda_{LAMBDA}_real_construction.png", nrow=8)
    
    # fake images
    fake_in = np.random.randn(nrows**2, LAT_DIM)*randn_mult
    fake_in = torch.from_numpy(fake_in).to(device).float()
    fake_imgs = net.decoder(fake_in)
    fake_imgs = fake_imgs.reshape(-1, 1, 28, 28)
    fake_filename = f"{DIR_OUT}/lambda_{LAMBDA}_fake.png"
    save_image(fake_imgs, fake_filename, nrow=nrows)

    # real images interpolation
    if LAT_DIM == 2:
        a = np.array([-randn_mult,-randn_mult])
        b = np.array([randn_mult,-randn_mult])
        c = np.array([-randn_mult,randn_mult])
        d = np.array([randn_mult,randn_mult])
    else:
        a = np.random.randn(1, LAT_DIM)*randn_mult
        b = np.random.randn(1, LAT_DIM)*randn_mult
        c = np.random.randn(1, LAT_DIM)*randn_mult
        d = np.random.randn(1, LAT_DIM)*randn_mult
    r1 = net.latent(real)[0]
    r1 = r1.cpu().detach().numpy()
    r2 = net.latent(real)[6]
    r2 = r2.cpu().detach().numpy()
    interp_in = torch.from_numpy(interp_vectors(r1, r2, nrows)).to(device).float()
    interp_out = net.decoder(interp_in)
    interp_out = interp_out.reshape(-1, 1, 28, 28)
    interp_filename = f"{DIR_OUT}/lambda_{LAMBDA}_interp.png"
    save_image(interp_out, interp_filename, nrow=8)

generateImg(net=net)

# %%
