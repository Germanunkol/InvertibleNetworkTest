from torch.utils.data import Dataset
import random
import torch

class ToyDataset( Dataset ):

    def __init__( self, num = 1000 ):
        super(ToyDataset, self).__init__()

        self.num = num
        self.height = 7
        self.width = 3

        #self.blur = torch.ones( 1, 1, 5, 5 )/25

    def __len__( self ):
        return self.num

    def im2Lbl( self, im, verbose=False ):

        w = self.width
        h = self.height
        if im.dim() == 3:
            lbl = torch.zeros_like(im)
            lbl[0,:,0] = im.sum(dim=2)
            return lbl[0:1,:,0:1]
        else:
            lbl = torch.zeros_like(im)
            lbl[:,0:1,:,0] = im.sum(dim=3)
            return lbl[:,0:1,:,0:1]

    def __getitem__( self, i ):

        w = self.width
        h = self.height
        im = torch.zeros( 1, w, h )

#        for x in range(0,self.width):
#            im[0,0,x] = 0.2
#            im[0,self.height-1,x] = 0.2
#
#        for y in range(0,self.height):
#            im[0,y,0] = 0.2
#            im[0,y,self.width-1] = 0.2

        for i in range(0,3):
            x = random.randint( 0, w-1 )
            y = random.randint( 1, h-1 )
            im[0,x,y] = 0.1

        lbl = self.im2Lbl( im )

        return im, lbl

