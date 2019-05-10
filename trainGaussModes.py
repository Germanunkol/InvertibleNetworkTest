import torch
from torch import nn, optim
import os, sys, random, shutil
from torch.utils.data import DataLoader
import torch_two_sample
from tensorboardX import SummaryWriter

from GaussModes import GaussModesDataset
from InvertibleNetwork import InvertibleNetwork, InvertibleBlock, inverseTest

import torchvision

outDir = "out"
if len(sys.argv) > 1:
    outDir = sys.argv[1]
print("Output directory:", outDir)
if not os.path.exists( outDir ):
    os.makedirs( outDir )
for f in os.listdir("."):
    if f.endswith(".py"):
        shutil.copyfile( f, os.path.join(outDir, f) )

logger = SummaryWriter(outDir)

batch_size = 200
modeLabels = [ 0, 0, 1, 1, 1 ]
dataset = GaussModesDataset( modeLabels=modeLabels, num=1e6 )
dataloader = DataLoader( dataset, shuffle=True, batch_size=batch_size, num_workers=8)
#dataset.toImage( os.path.join( outDir, "dataset.png" ) )
datasetTest = GaussModesDataset( modeLabels=modeLabels, num=1e3 )
dataloaderTest = DataLoader( datasetTest, shuffle=False, batch_size=batch_size, num_workers=8)
datasetTest.toImage( os.path.join( outDir, "test_dataset.png" ) )

numModes = dataset.numModes
numLabels = dataset.numLabels

numBlocks = 3
numChannels = 16
f = nn.ModuleList()
for i in range(numBlocks):
    c = numChannels//2
    s1 = nn.Sequential(
            nn.Linear( c, c ),
            nn.LeakyReLU(),
            nn.Linear( c, c ),
            nn.LeakyReLU(),
            nn.Linear( c, c ) )
    t1 = nn.Sequential(
            nn.Linear( c, c ),
            nn.LeakyReLU(),
            nn.Linear( c, c ),
            nn.LeakyReLU(),
            nn.Linear( c, c ) )
    s2 = nn.Sequential(
            nn.Linear( c, c ),
            nn.LeakyReLU(),
            nn.Linear( c, c ),
            nn.LeakyReLU(),
            nn.Linear( c, c ) )
    t2 = nn.Sequential(
            nn.Linear( c, c ),
            nn.LeakyReLU(),
            nn.Linear( c, c ),
            nn.LeakyReLU(),
            nn.Linear( c, c ) )
    f.append( InvertibleBlock(s1,t1,s2,t2,numChannels) )
net = InvertibleNetwork( f ).cuda()

inverseTest( net, numChannels )

MSE = nn.MSELoss()

if batch_size > 1:
    MMDStat = torch_two_sample.MMDStatistic( batch_size, batch_size )

alphas = [0.0001, 0.00001, 0.01]   # Works with fullZ
#alphas = [0.001, 0.01]
#alphas = [0.001, 0.01, 0.01]
#alphas = [0.000001, 0.0000001, 0.001, 0.001]

def MMD( a, b ):
    global MMDStat,alphas
    #a = a.view( batch_size, -1 )
    #b = b.view( batch_size, -1 )
    a = a.view( batch_size, -1 )
    b = b.view( batch_size, -1 )
    return MMDStat( a, b, alphas )

optimizer = optim.Adam(net.parameters(), lr = 1e-4)
#scheduler = optim.lr_scheduler.StepLR(optimizer,5000,gamma=0.75)

def padded( x, randomize=False ):
    if randomize == False:
        padded = torch.zeros( batch_size, numChannels ).cuda()
    else:
        padded = torch.randn_like( batch_size, numChannels ).cuda()
    padded[:,0:x.shape[1]] = x
    return padded

def testNet():
    samples = []
    for pos,lbl in dataloaderTest:
        pos = pos.cuda()
        pos = padded( pos )
        predForward = net(pos)
        for i in range(0,batch_size):
            sample = (pos[i,:], predForward[i,0:numLabels])
            samples.append( sample )
    datasetTest.samplesToImage( samples, os.path.join( outDir, "test.png" ) )

iteration = 0
numEpochs = 1000
for epoch in range(0, numEpochs):
    print("Epoch:", epoch)
    for pos,lbl in dataloader:

        optimizer.zero_grad()

        pos = pos.cuda()
        lbl = lbl.cuda()

        pos_padded = padded( pos )
        lbl_padded = padded( lbl )


        # Forward Pass:
        forward = net(pos_padded) 
        y = forward[:,0:numLabels]
        z = forward[:,numLabels:numLabels+2]

        # Sample a random z from N(0,1):
        zRnd = torch.randn_like(z)

        # Forward losses:
        Ly = MSE( y.squeeze(), lbl )
        #Lz = MMD( torch.cat((y.detach(),z),dim=1), torch.cat((lbl,zRnd),dim=1) )
        Lz = torch.Tensor((0,)).cuda()

        # Backward pass with random z:
        

        # Ensure padding goes towards zero:
        Lzero = torch.mean( forward[:,2:] ** 2 )

        loss = Ly + Lz + Lzero
        loss.backward()

        optimizer.step()
        #scheduler.step()

        if iteration % 1000 == 0:
            print("Iteration:", iteration)
            print("Loss: {:f} (Ly: {:f}, Lz: {:f}, Lzero: {:f})".format(loss.item(), Ly.item(), Lz.item(), Lzero.item()))
            print("\tz {:f} {:f} ({:f} - {:f})".format( z.mean().item(), z.std().item(), z.min().item(), z.max().item()))
            print("\tzRnd {:f} {:f} ({:f} - {:f}):".format( zRnd.mean().item(), zRnd.std().item(), zRnd.min().item(), zRnd.max().item() ) )
            print("Test:", iteration)
            testNet()

        logger.add_scalar('loss', loss.item(), iteration)
        logger.add_scalar('lossLy', Ly.item(), iteration)
        logger.add_scalar('lossLz', Lz.item(), iteration)
        iteration += 1


