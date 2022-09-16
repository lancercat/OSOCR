import torch;
import numpy as np;
class neko_att_ap(torch.nn.Module):
    def __init__(this,nchannel):
        super(neko_att_ap,this).__init__();
        this.score = torch.nn.Sequential(
            torch.nn.Conv2d(nchannel, 12, 1),
        );
        this.nchannel=nchannel;
        this.frac=np.sqrt(nchannel);
    def forward(this,feature):
        scores=this.score(feature).max(dim=1,keepdim=True)[0]/this.frac;
        shp=scores.shape;
        t=feature.view(shp[0],this.nchannel,-1)*torch.nn.functional.softmax(scores.view(shp[0],1,-1),dim=-1);
        return t.sum(-1)
