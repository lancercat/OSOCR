import torch.nn as nn
import math
from neko_sdk.encoders.blocks import neko_LensBlock,BasicBlockNoLens
import torch;
from neko_sdk.spatial_embeddings.sinemb import neko_sin_se;


class neko_reslayer(nn.Module):
    def __init__(this,in_planes, planes, blocks=1, stride=1):
        super(neko_reslayer, this).__init__()
        this.in_planes=in_planes
        this.downsample = None
        if stride != 1 or in_planes != planes * BasicBlockNoLens.expansion:
            this.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * BasicBlockNoLens.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlockNoLens.expansion),
            )

        this.layers = []
        this.layers.append(BasicBlockNoLens(this.in_planes, planes, stride, this.downsample))
        this.add_module("blk" + "init", this.layers[-1]);
        in_planes = planes * BasicBlockNoLens.expansion
        for i in range(1, blocks):
            this.layers.append(BasicBlockNoLens(in_planes, planes))
            this.add_module("blk"+str(i),this.layers[-1]);
        this.out_planes=planes;

    def forward(this, input):
        fields=[];
        feat=input;
        for l in  this.layers:
            feat,f=l(feat);
            if(f is not None):
                fields.append(f);
        return feat,fields;

class neko_se_reslayer(nn.Module):
    def __init__(this,in_planes, planes, blocks, stride=1):
        super(neko_se_reslayer, this).__init__()
        hes=in_planes//4;
        this.se = neko_sin_se(hes)

        this.in_planes=in_planes+hes
        this.downsample = None
        if stride != 1 or in_planes != planes * BasicBlockNoLens.expansion:
            this.downsample = nn.Sequential(
                nn.Conv2d(this.in_planes, planes * BasicBlockNoLens.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlockNoLens.expansion),
            )

        this.layers = []
        this.layers.append(BasicBlockNoLens(this.in_planes, planes, stride, this.downsample))
        this.add_module("blk" + "init", this.layers[-1]);
        in_planes = planes * BasicBlockNoLens.expansion
        for i in range(1, blocks):
            this.layers.append(BasicBlockNoLens(in_planes, planes))
            this.add_module("blk"+str(i),this.layers[-1]);
        this.out_planes=planes;

    def forward(this, input):
        fields=[];
        feat=input;
        se=this.se(feat.shape[-1],feat.shape[-2]).unsqueeze(0).repeat(feat.shape[0],1,1,1);
        feat=torch.cat([feat,se],dim=1);
        for l in  this.layers:
            feat,f=l(feat);
            if(f is not None):
                fields.append(f);
        return feat,fields;
