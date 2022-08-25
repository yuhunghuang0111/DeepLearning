"""
Code for the VAE (Section 2 of the problem)
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
image_path = 'VAE_dataset/anime_faces'
img = []
for i in range(10000):
    img_path = f'VAE_dataset/anime_faces\{i+1}.png'
    image_ori = image.imread(img_path)
    img.append(image_ori)
img = np.array(img) # (10000, 64, 64, 3)
"""
plt.axis(("off"))
plt.imshow(img[1])
plt.show()
"""


#%%
batch_size = 100
class TrainDataset(Dataset):
    
    def __init__(self, image):
        self.image = image
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        image = self.transform((self.image[idx])).float()
        return image.to(device)
train_data = TrainDataset(img)
#%%
print(len(train_data))
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_data, batch_size=batch_size)
len(train_dataloader)
# %%
class VAE(nn.Module):
    def __init__(self, latent=2):
        super(VAE, self).__init__()
        self.conv_layer = nn.Sequential(
            # (in_channels, out_channels, kernel_size, stride, padding)
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten()
            )   # (batch, 64*8*8)
        
        self.fc1 = nn.Sequential(
            nn.Linear(64*8*8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            )
        self.fc2_mean = nn.Linear(64, latent)
        self.fc2_logv = nn.Linear(64, latent)
        self.fc3 = nn.Sequential(
            nn.Linear(latent, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64*8*8),
            nn.ReLU()
            )

        self.deconv_layer = nn.Sequential(
            # (in_c, out_c, kernel_size, stride, padding, output_padding)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, 2, 1, 1),
            nn.Tanh()
            )

    def encoder(self, x):

        x = self.conv_layer(x)
        x = self.fc1(x)
        mean = self.fc2_mean(x)
        logv = self.fc2_logv(x)
        return mean, logv

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        z = torch.randn(size = (mu.size(0),mu.size(1)))
        z= z.type_as(mu)
        return mu + std * z

    def decoder(self, x):
        
        x = self.fc3(x)
        x = x.view(-1, 64, 8, 8)
        x = self.deconv_layer(x)
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        kl_loss = (-0.5*(1 + logVar - mu**2 - torch.exp(logVar)).sum(dim=1)).mean(dim=0)
        return out, kl_loss
    
    
    def latent(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z
# %%

LAMBDA = 100
learning_rate = 1e-3
def train(epochs):
    train_loss_list=[]
    net = VAE().to(device)
    criterion = nn.MSELoss(reduction='sum')
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

            
        avg_loss = training_loss/(10000)
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
net = train(80) # train 100 epochs

# %%
from torchvision.utils import save_image
from torchvision.utils import make_grid
LAT_DIM = 2
DIR_OUT = "results/anime_faces"
def interp_vectors(start, stop, ncols):
    steps = (1.0/(ncols-1)) * (stop - start)
    return np.array([(start + steps*x) for x in range(ncols)])

def generateImg(net, randn_mult=4, nrows=8):
    # real samples
    real = [train_data[i] for i in range(64)]
    real = torch.stack(real)
    save_image(real, f"{DIR_OUT}/lambda_{LAMBDA}_real.png", nrow=8)
    
    # resconstruction
    real_cons = net.forward(real)[0]
    
    save_image(real_cons, f"{DIR_OUT}/lambda_{LAMBDA}_real_construction.png", nrow=8)
    
    # fake images
    fake_in = np.random.randn(nrows**2, LAT_DIM)*randn_mult
    fake_in = torch.from_numpy(fake_in).to(device).float()
    fake_imgs = net.decoder(fake_in)
    fake_imgs = fake_imgs.reshape(-1, 3, 64, 64)
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
    interp_out = interp_out.reshape(-1, 3, 64, 64)
    interp_filename = f"{DIR_OUT}/lambda_{LAMBDA}_interp.png"
    save_image(interp_out, interp_filename, nrow=8)

generateImg(net=net)

# %%
