from torch import nn;
import torch;
import torch.nn.functional as trnf;


class neko_dense_calc(nn.Module):
    def __init__(this,channel):
        super(neko_dense_calc, this).__init__();
        ## 996-251. I am not on purpose.
        ## The sigmoid perserves topology.
        this.dkern=torch.nn.Sequential(
            nn.Conv2d(int(channel),2,5,1),
            torch.nn.Sigmoid());
    def forward(this,feat,lw,lh):
        if(this.training):
            dmap = this.dkern(feat/(feat.norm(dim=1,keepdim=True)+0.00000009));
        else:
            norm=feat.norm(dim=1,keepdim=True);
            dmap = this.dkern(feat/(norm+0.00000009)*(norm>0.09));
            # During large-batch evaluation, numeric errors seems to get much larger than eps,
            # causing sever performance loss, hence we commence this hot-fix.

        ndmap = trnf.interpolate(dmap, [lh + 2, lw + 2]);
        return ndmap;

class neko_dense_calcnn(nn.Module):
    def __init__(this,channel):
        super(neko_dense_calcnn, this).__init__();
        ## 996-251. I am not on purpose.
        ## The sigmoid perserves topology.
        this.dkern=torch.nn.Sequential(
            nn.Conv2d(int(channel),2,5,1),
            torch.nn.Sigmoid());
    def forward(this,feat,lw,lh):
        dmap = this.dkern(feat);
        ndmap = trnf.interpolate(dmap, [lh + 2, lw + 2]);
        return ndmap;


def neko_dense_norm(ndmap):
    [h__, w__] = ndmap.split([1, 1], 1);
    sumh=torch.sum(h__, dim=2, keepdim=True);
    sumw=torch.sum(w__,dim=3,keepdim=True);
    h_ = h__ / sumh;
    w_ = w__ / sumw;
    h = torch.cumsum(h_, dim=2);
    w = torch.cumsum(w_, dim=3);
    nidx = torch.cat([ w[:, :, 1:-1, 1:-1],h[:, :, 1:-1, 1:-1]], dim=1)* 2 - 1;
    return nidx;

def neko_sample(feat,grid,dw,dh):
    dst = trnf.grid_sample(feat, grid.permute(0, 2, 3, 1),mode="bilinear");
    return trnf.adaptive_avg_pool2d(dst,[dh,dw]);

def vis_lenses(img,lenses):
    oups=[img];
    for lens in lenses:
        dmap=trnf.interpolate(lens, [img.shape[-2],img.shape[-1]])
        grid=neko_dense_norm(dmap);
        img=neko_sample(img,grid,img.shape[3],img.shape[2])
        oups.append(img);
    return oups;

class neko_lens(nn.Module):
    DENSE=neko_dense_calc
    def __init__(this,channel,pw,ph,hardness=2,dbg=False):
        super(neko_lens, this).__init__();

        this.pw=pw;
        this.ph=ph;
        this.dbg=dbg;
        this.hardness=hardness;
        this.dkern=this.DENSE(channel)


    def forward(this,feat):
        dw = feat.shape[3] // this.pw;
        dh = feat.shape[2] // this.ph;
        dmap=this.dkern(feat,dw,dh);
        dmap += 0.0009;  # sigmoid can be clipped to 0!
        dmap[:, :, 1:-1, 1:-1] += this.hardness;

        grid=neko_dense_norm(dmap);
        dst=neko_sample(feat,grid,dw,dh);
        if(not this.dbg):
            return dst,dmap.detach();
        else:
            return dst,dmap.detach();
class neko_lensnn(neko_lens):
    DENSE=neko_dense_calcnn

class neko_lens_w_mask(nn.Module):
    def __init__(this,channel,pw,ph,hardness=2,dbg=False):
        super(neko_lens_w_mask, this).__init__();

        this.pw=pw;
        this.ph=ph;
        this.dbg=dbg;
        this.hardness=hardness;
        this.dkern=neko_dense_calc(channel)


    def forward(this,feat,mask):
        dw = feat.shape[3] // this.pw;
        dh = feat.shape[2] // this.ph;
        dmap=this.dkern(feat,dw,dh);
        dmap += 0.0009;  # sigmoid can be clipped to 0!
        dmap[:, :, 1:-1, 1:-1] += this.hardness;

        grid=neko_dense_norm(dmap);
        dst=neko_sample(feat,grid,dw,dh);
        with torch.no_grad():
            dmsk=neko_sample(mask,grid,dw,dh);
        if(not this.dbg):
            return dst,dmsk,dmap.detach();
        else:
            return dst,dmsk,dmap.detach();

class neko_lens_self(nn.Module):
    def __init__(this,ich,channel,pw,ph,hardness=2,dbg=False):
        super(neko_lens_self, this).__init__();

        this.pw=pw;
        this.ph=ph;
        this.dbg=dbg;
        this.hardness=hardness;
        this.ekern=torch.nn.Sequential(
            nn.Conv2d(ich,channel,3,1,1),
            nn.BatchNorm2d(channel),
            nn.ReLU6()
        )
        this.dkern=neko_dense_calc(channel)


    def forward(this,feat):
        dw = feat.shape[3] // this.pw;
        dh = feat.shape[2] // this.ph;
        lfeat=this.ekern(feat)
        dmap=this.dkern(lfeat,dw,dh);
        dmap += 0.0009;  # sigmoid can be clipped to 0!
        dmap[:, :, 1:-1, 1:-1] += this.hardness;
        grid=neko_dense_norm(dmap);
        dst=neko_sample(feat,grid,dw,dh);
        if(not this.dbg):
            return dst,dmap.detach();
        else:
            return dst,dmap.detach();


class neko_lens_fuse(nn.Module):
    def __init__(this,channel,hardness=0.5,dbg=False):
        super(neko_lens_fuse, this).__init__();
        this.hidense=neko_dense_calc(channel);
        this.lodense=neko_dense_calc(channel);
        # allow it to oversample


        this.dbg=dbg;
        this.hardness=hardness;


    def forward(this,hifeat,lofeat):
        dw = lofeat.shape[3];
        dh = lofeat.shape[2];
        lw=hifeat.shape[3];
        lh = hifeat.shape[2];
        hidmap=this.hidense(hifeat,lw,lh);
        lodmap=this.lodense(lofeat,lw,lh);
        dmap=(hidmap+lodmap)/2;
        dmap += 0.0009;  # sigmoid can be clipped to 0!
        dmap[:, :, 1:-1, 1:-1] += this.hardness;
        grid=neko_dense_norm(dmap);
        his=neko_sample(hifeat,grid,dw,dh);
        los=neko_sample(lofeat,grid,dw,dh);
        return his+los,dmap.detach();
