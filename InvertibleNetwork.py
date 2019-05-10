import torch
import torch.nn as nn

class InvertibleBlock( nn.Module ):
    def __init__(self, S1, T1, S2, T2, numChannels):
        super(InvertibleBlock, self).__init__()
        self.T1 = T1
        self.S1 = S1
        self.T2 = T2
        self.S2 = S2

        self.perm = torch.randperm(numChannels)
        self.permInverse = torch.argsort(self.perm)

        #self.extreme = torch.FloatTensor( [10] ).cuda()
        self.extreme = torch.FloatTensor( [1e2] ).cuda()

    def forward(self, u):
        bs = u.shape[0]
        c = u.shape[1]//2
        u = u[:,self.perm]
        u1,u2 = torch.split( u, c, dim=1 )
        v1 = u1*torch.exp(torch.min( self.S2(u2), self.extreme)) + self.T2(u2)
        v2 = u2*torch.exp(torch.min( self.S1(v1), self.extreme)) + self.T1(v1) 
        v = torch.cat( (v1,v2), dim=1 )
        v = v[:,self.permInverse]
        return v

    def inverse(self, v):
        bs = v.shape[0]
        c = v.shape[1]//2
        v = v[:,self.perm]
        v1,v2 = torch.split( v, v.shape[1]//2, dim=1 )
        u2 = (v2 - self.T1(v1))*torch.exp( torch.min(-self.S1(v1),self.extreme))
        u1 = (v1 - self.T2(u2))*torch.exp( torch.min(-self.S2(u2),self.extreme))
        u = torch.cat( (u1,u2), dim=1 )
        u = u[:,self.permInverse]
        return u


class InvertibleNetwork( nn.Module ):
    def __init__(self, blocks):
        super(InvertibleNetwork, self).__init__()
        self.blocks = blocks

    def forward(self,u):
        u = u/10
        for m in self.blocks:
            u = m(u)
        return u*10

    def inverse(self,u):
        u = u/10
        for m in reversed(self.blocks):
            u = m.inverse(u)
            #print(torch.max(u).item(), "inv")
        return u*10

    def printGrads(self):
        for b in self.blocks:
            #print("block")
            maximum = 0
            for m in b.modules():
                for p in m.parameters():
                    mm = torch.max(p).item()
                    if maximum < mm:
                        maximum = mm
            #print(maximum)


def inverseTest( net, numChannels ):
    print("Network inverse test:")
    batch_size = 20
    d = [batch_size, numChannels]
    r = torch.randn( d ).cuda()
    print(r.shape)
    res = net(r)
    inv = net.inverse(res)
    diff = torch.max( torch.abs( inv -r ) )
    print("\tMax Diff:", diff.item())

if __name__ == "__main__":
    # Invertible Block test:

    c0 = torch.nn.Sequential( torch.nn.Conv2d(2,2,3,padding=1), torch.nn.Softsign() )
    c1 = torch.nn.Sequential( torch.nn.Conv2d(2,2,5,padding=2), torch.nn.Softsign() )
    c2 = torch.nn.Sequential( torch.nn.Conv2d(2,2,7,padding=3), torch.nn.Softsign() )
    c3 = torch.nn.Sequential( torch.nn.Conv2d(2,2,1,padding=0), torch.nn.Softsign() )

    ib1 = InvertibleBlock(c0,c0,c1,c2,4)
    ib2 = InvertibleBlock(c2,c3,c1,c0,4)
    ib3 = InvertibleBlock(c2,c3,c1,c1,4)
    ib4 = InvertibleBlock(c2,c3,c0,c2,4)
    ib5 = InvertibleBlock(c2,c3,c1,c2,4)
    ib6 = InvertibleBlock(c3,c3,c1,c2,4)

    net = InvertibleNetwork( [ib1,ib2,ib3,ib4,ib5,ib6] )

    u = torch.randn( 1, 4, 11, 11 )

    v = net(u)

    inverse = net.inverse(v)

    print(u[0,0,0,0].item())
    print(v[0,0,0,0].item())
    print(inverse[0,0,0,0].item())
    print(torch.mean(u-inverse).item())

