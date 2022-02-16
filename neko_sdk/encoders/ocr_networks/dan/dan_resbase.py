import torch.nn as nn
import math
from neko_sdk.AOF.neko_lens import neko_lens,neko_lensnn,vis_lenses;
from neko_sdk.AOF.neko_reslayers import neko_reslayer,neko_se_reslayer;


class dan_ResNet(nn.Module):
    LAYER=neko_reslayer;
    LENS=neko_lens;
    def freezebn(this):
        for m in this.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if this.freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def unfreezebn(this):
        for m in this.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train();
                if this.freeze_bn_affine:
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
    def setup(self, layers, strides,layertype,hardness, compress_layer=True,inpch=1,oupch=512,expf=1.0):

        self.inplanes = int(32*expf)
        self.freeze_bn_affine=False;
        self.conv1 = nn.Conv2d(inpch, int(32*expf), kernel_size=3, stride=strides[0], padding=1,
                               bias=False)
        # self.deformation=self.LENS(int(32*expf),1,1,hardness);
        self.bn1 = nn.BatchNorm2d(int(32*expf))
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self.LAYER( int(32*expf),int(64*expf), layers[0],stride=strides[1])
        self.layer2 = self.LAYER( int(64*expf),int(128*expf), layers[1], stride=strides[2])
        self.layer3 = self.LAYER( int(oupch//4*expf),int(oupch//2*expf), layers[2], stride=strides[3])
        self.layer4 = self.LAYER(int(oupch//2*expf),int(oupch*expf), layers[3], stride=strides[4])
        if(compress_layer):
            self.layer5 = self.LAYER( int(oupch*expf),int(oupch*expf), layers[4], stride=strides[5])
        else:
            self.layer5 = self.LAYER(int(oupch*expf), oupch, layers[4], stride=strides[5])
        self.compress_layer = compress_layer
        if compress_layer:
            # for handwritten
            self.layer6 = nn.Sequential(
                nn.Conv2d(int(512*expf), oupch, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1)),
                nn.BatchNorm2d(oupch),
                nn.ReLU(inplace = False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __init__(self, layers, strides,layertype,hardness, compress_layer=True,inpch=1,oupch=512,expf=1.0):
        super(dan_ResNet, self).__init__()
        self.setup(layers, strides,layertype,hardness, compress_layer=compress_layer,inpch=inpch,oupch=oupch,expf=expf)
    def pre_layers(self,x):
        x1 = self.conv1(x)
        # x2, lens = self.deformation(x1);
        x3 = self.bn1(x1)
        x4 = self.relu(x3)
        return x4,None;

    def forward(self, x, multiscale = False):
        out_features = []
        grids = []

        tmp_shape = x.size()[2:];
        x,lens=self.pre_layers(x);
        # grids.append(lens);

        x,grid = self.layer1(x);
        grids+=grid;
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        grids+=grid;
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


def res_base(strides, compress_layer,hardness,inpch=1,oupch=512,expf=1):
    model = dan_ResNet( [3, 4, 6, 6, 3], strides,None,hardness, compress_layer,inpch=inpch,oupch=oupch,expf=expf)
    return model
def res_base45_thicc(strides, compress_layer,hardness,inpch=1,oupch=512):
    model = dan_ResNet( [3, 4, 6, 6, 3], strides,None,hardness, compress_layer,inpch=inpch,oupch=oupch,expf=1.5)
    return model

