import torch
from torch import nn
from torch import optim
from torchvision import transforms
#from MMD import MMD
import torch_two_sample
import os, sys, random
from tensorboardX import SummaryWriter

from InvertibleNetwork import InvertibleNetwork, InvertibleBlock, inverseTest

from Toy import ToyDataset
from torch.utils.data import DataLoader

import torchvision
import shutil

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

dataset = ToyDataset(1000000)
batch_size = 200
dataloader = DataLoader( dataset, shuffle=True, batch_size=batch_size, num_workers=8)
datasetTest = ToyDataset( num=2 )
testDataloader = DataLoader( datasetTest, shuffle=False, batch_size=1, num_workers=8)

numEpochs = 10

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        print(input.shape)
        shape = input.shape[0:2] + self.shape
        print(shape)
        return input.view(shape)

numBlocks = 5
numChannels = 10
f = nn.ModuleList()
for i in range(numBlocks):
    c = numChannels//2
    ks = 9
    p = ks//2
    imHeight = 7
    imWidth = 3
    s1 = nn.Sequential(
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign(),
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign(),
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign() )
    t1 = nn.Sequential(
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign(),
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign(),
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign() )
    s2 = nn.Sequential(
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign(),
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign(),
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign() )
    t2 = nn.Sequential(
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign(),
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign(),
            nn.Linear( c*imWidth*imHeight, c*imWidth*imHeight ),
            nn.Softsign() )
    f.append( InvertibleBlock(s1,t1,s2,t2,numChannels) )

f.append( InvertibleBlock(s1,t1,s2,t2,numChannels) )
net = InvertibleNetwork( f ).cuda()

# Test net inverse:
#inverseTest( net, numChannels, 9 )

#def display( ims ):
    #global plotIm
    #im = torch.cat(ims,dim=2)[0].squeeze(0).transpose(0,1)
    #if plotIm == None:
        #plotIm = plt.imshow( im )
    #else:
        #plotIm.set_data( im )
        #plt.draw()
    #plt.imshow( torch.cat(ims,dim=2)[0].squeeze(0).transpose(0,1) )
    #plt.pause(0.0005)

def blowUp( im, numChannels, noise=False ):
    if noise == False:
        padding = torch.zeros( im.shape[0], numChannels - im.shape[1], im.shape[2], im.shape[3] ).cuda()
    else:
        padding = torch.randn( im.shape[0], numChannels - im.shape[1], im.shape[2], im.shape[3] ).cuda()
    return torch.cat( [im, padding], dim=1 )

MSE = nn.MSELoss()

MMDStat = torch_two_sample.MMDStatistic( batch_size, batch_size )
#alphas = [0.0001, 0.00001, 0.01]   # Works with fullZ
alphas = [0.001, 0.01]
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
scheduler = optim.lr_scheduler.StepLR(optimizer,5000,gamma=0.75)

iteration = 0
for epoch in range(0, numEpochs):
    for x,lbl in dataloader:

        optimizer.zero_grad()

        x = x.cuda()
        lbl = lbl.cuda()

        lblPadded = torch.zeros_like(x)
        lblPadded[:,:,:,0:1] = lbl

        padded = blowUp(x, numChannels)
        output = net(padded)
        y = output[:,0:1,:,0:1]
        zFullLayer = output[:,1:2,:,:]
        z = output[:,1:2,:,0:2].contiguous()
        #zPadded = torch.zeros_like(lbl).cuda()
        #zPadded[:,0:1,:,:] = z

        yPadded = torch.zeros_like(x)
        yPadded[:,:,:,0:1] = y

        # Target loss:
        lossFFit = MSE( y, lbl )

        zRnd = torch.randn_like( z )

        # Distribution loss on z and y:
        #outputDist = torch.cat( (y.detach(), z), dim=1 )        # Block gradient w.r.t. label
        #lblDist = torch.cat( (lbl, zRnd), dim=1 )        # Block gradient w.r.t. label

        #lossDist = MMD2( outputDist.view(batch_size,-1), lblDist.view(batch_size,-1), alphas )
        #lossDist = MMD( outputDist, lblDist )
        lossFMMD = MMD( torch.cat((y.detach(),z),dim=3), torch.cat((lbl,zRnd),dim=3) )
        #lossDistTmp = MMD2( z.view(batch_size,-1), zRnd.view(batch_size,-1), alphas )

        inv = net.inverse( output )
        xInvTrue = inv[:,0:1,:,:]

        #yInv = y.detach() + torch.randn_like( y )*0.01
        zInv = torch.zeros_like(x)
        zInv[:,:,:,0:2] = z# + torch.randn_like( z )*0.001
        inpInv = torch.cat( (lblPadded,zInv), dim = 1 )
        inpInv += torch.randn_like( inpInv )*0.1
        paddedInv = blowUp( inpInv, numChannels )
        inv = net.inverse( paddedInv )
        xInv = inv[:,0:1,:,:]

        lossBFit = MSE( x, xInv )
        xInvLbl = dataset.im2Lbl( xInv, True )
        xInvLblPadded = torch.zeros_like(x)
        xInvLblPadded[:,:,:,0:1] = xInvLbl

        x2,lbl2 = dataset[random.randint(0,len(dataset))]
        lbl2Padded = torch.zeros_like(x)
        lbl2Padded[:,:,:,0:1] = lbl

        zRnd = torch.zeros_like(lbl2Padded)
        zRnd[:,:,:,0:2] = torch.randn_like( z )
        inpRnd = torch.cat( (lbl2Padded,zRnd), dim = 1 )
        paddedRnd = blowUp( inpRnd, numChannels )
        inv = net.inverse( paddedRnd )
        xFromRnd = inv[:,0:1,:,:]
        xFromRndLbl = dataset.im2Lbl( xFromRnd )
        xFromRndLblPadded = torch.zeros_like(x)
        xFromRndLblPadded[:,:,:,0:1] = xFromRndLbl

        lossBMMD = MMD( xFromRnd, x )

        zInvNoised = torch.randn_like(x)
        zInvNoised[:,:,:,0:2] = z# + torch.randn_like( z )*0.001
        lblPaddedNoised = torch.randn_like(x)
        lblPaddedNoised[:,:,:,0:1] = lbl        # or y?
        inpInvNoised = torch.cat( (lblPaddedNoised,zInvNoised), dim = 1 )
        paddedInvNoised = blowUp( inpInvNoised, numChannels, noise=True )
        invNoised = net.inverse( paddedInvNoised )
        xInvNoised = invNoised[:,0:1,:,:]

        lossNoisedB = MSE( x, xInvNoised )

        #lossDistInv = MMD2( x.view(batch_size,-1), predInv.view(batch_size,-1), alphas )
        #lossDistInv = MMD( x, predInv )

        paddingF = torch.cat( [output[:,2:,:,:].contiguous().view( -1 ), output[:,1,:,2:].contiguous().view(-1)] )
        lossZeroF = MSE( paddingF, torch.zeros_like( paddingF ) )
        paddingB = inv[:,2:,:,:].contiguous().view( -1 )
        lossZeroB = MSE( paddingB, torch.zeros_like( paddingB ) )

        loss = 10*lossFFit + 10*lossBFit + lossFMMD + lossBMMD + lossZeroF + lossZeroB #+ lossNoisedB
        #loss = lossDist #+ lossPadding
        #loss = lossDist#+ lossDistribution #+ lossPadding
        loss.backward()
        #net.printGrads()
        #lossTgt.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 20 == 0:
            #print("iter: {:d}, loss: {:f} ({:f}, {:f}, {:f})".format( iteration, loss.item(), lossTgt.item(), lossLatent.item(), lossDistribution.item() ) )
            print("iter: {:d}, loss: {:f} ({:f}, {:f}, {:f}, {:f})".format( iteration, loss.item(), lossFFit.item(), lossFMMD.item(), lossBFit.item(), lossBMMD.item() ) )
            #print("iter: {:d}, loss: {:f}".format( iteration, lossTgt.item() ) )
            #c = torch.cat([im,lbl,pred,z,predInv,zInv], dim=0)
            c = torch.cat([x,lblPadded,yPadded,zFullLayer/6+0.5,xInvTrue,zInv/6+0.5,xInv,xInvLblPadded,xInvNoised,lbl2Padded,zRnd/6+0.5,xFromRnd,xFromRndLblPadded], dim=0)
            print("\tz {:f} {:f} ({:f} - {:f})".format( z.mean().item(), z.std().item(), z.min().item(), z.max().item()))
            print("\tzRnd {:f} {:f} ({:f} - {:f}):".format( zRnd.mean().item(), zRnd.std().item(), zRnd.min().item(), zRnd.max().item() ) )
            torchvision.utils.save_image( c, os.path.join(outDir, "train.png"), nrow=batch_size, pad_value = 0.01, padding=1, scale_each=True )
            #print("Sanity check:")
            #print( MMD2( outputDist.view(batch_size,-1), outputDist.view(batch_size,-1), alphas ).item() )
            #print( MMD2( lblDist.view(batch_size,-1), lblDist.view(batch_size,-1), alphas ).item() )
            #print( MMD2( outputDist.view(batch_size,-1), lblDist.view(batch_size,-1), alphas ).item() )
        iteration += 1

        logger.add_scalar('loss', loss.item(), iteration)
        logger.add_scalar('lossFFit', lossFFit.item(), iteration)
        logger.add_scalar('lossFMMD', lossFMMD.item(), iteration)
        logger.add_scalar('lossBFit', lossBFit.item(), iteration)
        logger.add_scalar('lossBMMD', lossBMMD.item(), iteration)
        logger.add_scalar('lossNoisedB', lossNoisedB.item(), iteration)


    # Test:
    for x,lbl in testDataloader:
        x = x.cuda()
        lbl = lbl.cuda()
        predictions = []
        z = []
        for i in range( 0, batch_size ):
            z_ = torch.randn_like( x )
            z_[:,:,:,2:] = 0
            lblPadded = torch.zeros_like(x)
            lblPadded[:,:,:,0:1] = lbl
            inp = torch.cat( (lblPadded,z_), dim = 1 )
            padded = blowUp( inp, numChannels )
            output = net.inverse(padded)
            predictions.append( output[:,0,:,:].unsqueeze(1) )
            z.append( z_ )

        predictions = torch.cat( predictions, dim = 0 )
        z = torch.cat( z, dim = 0 )
        x = x.repeat(batch_size, 1,1,1)
        lblPadded = lblPadded.repeat(batch_size, 1,1,1)
        lblPrediction = dataset.im2Lbl( predictions )
        lblPredictionPadded = torch.zeros_like(x)
        lblPredictionPadded[:,:,:,0:1] = lblPrediction
        c = torch.cat([lblPadded,z/6+0.5,predictions,lblPredictionPadded,x], dim=0)
        torchvision.utils.save_image( c, os.path.join(outDir,"test.png"), nrow=batch_size, pad_value = 0.2, padding=1, scale_each=True )
        break


