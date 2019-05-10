from torch.utils.data import Dataset
import random, math
import torch, numpy
import torchvision

def label2Color( lbl ):
    col = (255,255,255)
    if lbl == 0:
        col = (128,0,0)
    elif lbl == 1:
        col = (0,128,0)
    elif lbl == 2:
        col = (0,0,128)
    elif lbl == 3:
        col = (128,128,0)
    elif lbl == 4:
        col = (0,128,128)
    elif lbl == 5:
        col = (128,0,128)
    elif lbl == 6:
        col = (0,128,255)
    elif lbl == 7:
        col = (255,128,0)
    elif lbl == 8:
        col = (128,255,0)
    return torch.Tensor( col )

class GaussModesDataset( Dataset ):

    def __init__( self, modeLabels = [0,0,1], num = 1000 ):
        super(GaussModesDataset, self).__init__()
 
        self.num = int(num)
        self.radius = 7
        self.modeRadius = 1

        self.numModes = len(modeLabels)
        self.numLabels = len(torch.unique(torch.Tensor(modeLabels)))
        self.modeLabels = modeLabels
        print("Created dataset with {:d} modes ({:d} unique labels)".format(self.numModes, self.numLabels))

    def __len__( self ):
        return self.num

    def __getitem__( self, i ):

        mode = i % self.numModes

        r = self.radius

        cx = r*math.sin(mode/self.numModes*math.pi*2)
        cy = r*math.cos(mode/self.numModes*math.pi*2)

        rx = torch.randn(1)*self.modeRadius
        ry = torch.randn(1)*self.modeRadius

        px = (cx+rx)
        py = (cy+ry)

        pos = torch.Tensor( (px,py) )

        lbl = torch.zeros( self.numLabels )
        lbl[self.modeLabels[mode]] = 1
        return pos,lbl

    def samplesToImage( self, samples, filename ):

        t = torch.zeros( 3, 100, 100 )

        for i in range(0,len(samples)):
            pos, lbl = samples[i]
            x = pos[0]*5 + 50
            y = pos[1]*5 + 50
            if x < 100 and y < 100 and x >= 0 and y >= 0:
                lblID = torch.argmax( lbl )
                col = label2Color(lblID)
                t[:,int(x),int(y)] = col

        print("img:", torch.min(t).item(), torch.max(t).item(), t.shape)

        torchvision.utils.save_image( t, filename )

    def toImage( self, filename ):

        samples = []
        for i in range(0, len(self)):
            samples.append( self[i] )

        self.samplesToImage( samples, filename )

if __name__ == "__main__":

    d = GaussModesDataset( modeLabels=[0,0,1,1,0,2,3], num=1000 )
    d.toImage( "data_samples.png" )


