import torch.nn as nn
import math
from neko_sdk.AOF.neko_lens import neko_lens;
from neko_sdk.AOF.neko_reslayers import neko_reslayer;


class dan_ResNetv2(nn.Module):
    LAYER=neko_reslayer;
    def __init__(self, layers, strides,layertype,hardness, compress_layer=True,inpch=1,oupch=512,expf=1.0):
        self.inplanes = int(32*expf)
        super(dan_ResNetv2, self).__init__()
        self.conv1 = nn.Conv2d(inpch, int(32*expf), kernel_size=3, stride=strides[0], padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(int(32*expf))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.LAYER( int(32*expf),int(64*expf), layers[0],stride=strides[1])
        self.deformation=neko_lens(int(64*expf),1,1,hardness);
        self.layer2 = self.LAYER( int(64*expf),int(128*expf), layers[1], stride=strides[2])
        self.layer3 = self.LAYER( int(128*expf),int(256*expf), layers[2], stride=strides[3])
        self.layer4 = self.LAYER(int(256*expf),int(512*expf), layers[3], stride=strides[4])
        if(compress_layer):
            self.layer5 = self.LAYER( int(512*expf),int(512*expf), layers[4], stride=strides[5])
        else:
            self.layer5 = self.LAYER(int(512*expf), oupch, layers[4], stride=strides[5])
        self.compress_layer = compress_layer        
        if compress_layer:
            # for handwritten
            self.layer6 = nn.Sequential(
                nn.Conv2d(int(512*expf), oupch, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1)),
                nn.BatchNorm2d(oupch),
                nn.ReLU(inplace = True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def pre_layers(self,x):
        x1 = self.conv1(x)
        x3 = self.bn1(x1)
        x4 = self.relu(x3)
        x5,grid = self.layer1(x4);
        x6, lens = self.deformation(x5);
        return x6,lens;

    def forward(self, x, multiscale = False):
        out_features = []
        grids = []

        tmp_shape = x.size()[2:];
        x,lens=self.pre_layers(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        grids.append(lens);

        x,grid = self.layer2(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x,grid = self.layer3(x)
        grids+=grid;
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x,grid = self.layer4(x)
        grids+=grid;
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x,grid = self.layer5(x);
        grids+=grid;
        if not self.compress_layer:
            out_features.append(x)
        else:
            if x.size()[2:] != tmp_shape:
                tmp_shape = x.size()[2:]
                out_features.append(x)
            x = self.layer6(x)
            out_features.append(x)

        return out_features,grids

def res_naive_lens45v2(strides, compress_layer,hardness,inpch=1,oupch=512):
    model = dan_ResNetv2( [3, 4, 6, 6, 3], strides,None,hardness, compress_layer,inpch=inpch,oupch=oupch)
    return model
def res_naive_lens45_thiccv2(strides, compress_layer,hardness,inpch=1,oupch=512):
    model = dan_ResNetv2( [3, 4, 6, 6, 3], strides,None,hardness, compress_layer,inpch=inpch,oupch=oupch,expf=1.5)
    return model

