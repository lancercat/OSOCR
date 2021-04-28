
import torch;
from torch import nn;
from neko_sdk.thirdparty.PositionalEncoding2D.positionalembedding2d import positionalencoding2d;

class neko_sin_se(nn.Module):
    def __init__(this,dim=32):
        super(neko_sin_se, this).__init__()
        this.sew=-1;
        this.seh=-1;
        this.se=None;
        this.dim=dim;
        this.devinc=nn.Parameter(torch.tensor([0]),requires_grad=False);
    def forward(this,w,h):
        if(w != this.sew or h!=this.seh):
            this.seh=h;
            this.sew=w;
            this.se=positionalencoding2d(this.dim,h,w).to(this.devinc.device);
        return this.se.to(this.devinc.device);

# if __name__ == '__main__':
#     neko_sin_se(32)(12,12)