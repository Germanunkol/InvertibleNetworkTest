import torch
import torch_two_sample

def MMD( x, y, verbose=False ):
    #x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
    #y = y.view(y.size(0), y.size(1) * y.size(2) * y.size(3))

    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
    #if verbose:
        #print(torch.mean(xx).item(), torch.mean(yy).item(), torch.mean(zz).item())
        #print(torch.sum(xx).item(), torch.sum(yy).item(), torch.sum(zz).item())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    #if verbose:
        #print(torch.sum(rx).item(), torch.sum(ry).item())

    alpha = 0.001
    K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(- alpha * (rx.t() + ry - 2*zz))
    if verbose:
        print("K,L,P", torch.sum(K).item(), torch.sum(L).item(), torch.sum(P).item())


    B = x.shape[0]

    beta = (1./(B*(B-1)))
    gamma = (2./(B*B)) 
    return beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)


if __name__ == "__main__":
    # MMD unit test:

    print("MMD")
    for i in range(0,10):
        r = torch.randn( 10,100 )
        r2 = torch.randn( 10,100 )

        print("r,r", MMD(r,r, False).item())
        print("r,r2", MMD(r,r2, False).item())
    #print("r2,r2", MMD(r2,r2))
    #print("r3,r3", MMD(r3,r3))

    # MMD torch_two_sample
    print("MMD torch_two_sample")
    MMD2 = torch_two_sample.MMDStatistic( 10, 10 )
    for i in range(0,10):
        r = torch.randn( 10,100 )
        r2 = torch.randn( 10,100 )

        alphas = [0.01]
        dSame = MMD2(r,r,alphas).item()
        dSame2 = MMD2(r2,r2,alphas).item()
        dDiff = MMD2(r,r2,alphas).item()
        print(dSame, dSame2, dDiff)
        print( dSame < dDiff, dSame2 < dDiff, dSame < dSame2 )
 
