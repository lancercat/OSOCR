import torch.nn
from torch import nn
from neko_sdk.NDK.coattention import MultiHeadCoAttention
from neko_sdk.spatial_embeddings.sinemb import neko_sin_se;
'''
Convolutional Alignment Module
'''
# Current version only supports input whose size is a power of 2, such as 32, 64, 128 etc.
# You can adapt it to any input size by changing the padding or stride.
class stacked_coattention(nn.Module):
    def __init__(this,d_qkv,num_layers):
        super(stacked_coattention, this).__init__()
        attns=[];
        this.num_layers=num_layers;
        for i in range(num_layers):
            attns.append(MultiHeadCoAttention(1, d_qkv, d_qkv, d_qkv))
        this.attns=torch.nn.Sequential(*attns);
        pass;
    def forward(this,sidea,sideb):
        for i in range(this.num_layers):
            sidea,_,_,_=this.attns[i](sidea,sideb)
        return sidea,sideb;

class neko_transformer_CAM(nn.Module):
    def __init__(this, scales, maxT, depth, num_channels):
        super(neko_transformer_CAM, this).__init__()
        # cascade multiscale features
        fpn = []
        this.maxT=maxT;
        this.keys=torch.nn.Parameter(torch.rand([1,maxT,num_channels+32])-0.5);
        this.attms=stacked_coattention(num_channels+32,3);
        this.se=neko_sin_se(32)
        for i in range(1, len(scales)):
            assert not (scales[i-1][1] / scales[i][1]) % 1, 'layers scale error, from {} to {}'.format(i-1, i)
            assert not (scales[i-1][2] / scales[i][2]) % 1, 'layers scale error, from {} to {}'.format(i-1, i)
            ksize = [3,3,5] # if downsampling ratio >= 3, the kernel size is 5, else 3
            r_h, r_w = int(scales[i-1][1] / scales[i][1]), int(scales[i-1][2] / scales[i][2])
            ksize_h = 1 if scales[i-1][1] == 1 else ksize[r_h-1]
            ksize_w = 1 if scales[i-1][2] == 1 else ksize[r_w-1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i-1][0], scales[i][0],
                                              (ksize_h, ksize_w),
                                              (r_h, r_w),
                                              (int((ksize_h - 1)/2), int((ksize_w - 1)/2))),
                                     nn.BatchNorm2d(scales[i][0]),
                                     nn.ReLU(True)))
        this.fpn = nn.Sequential(*fpn)
        # convolutional alignment
        # convs
        assert depth % 2 == 0, 'the depth of CAM must be a even number.'
        in_shape = scales[-1]
        strides = []
        conv_ksizes = []
        deconv_ksizes = []
        h, w = in_shape[1], in_shape[2]
        for i in range(0, int(depth / 2)):
            stride = [2] if 2 ** (depth/2 - i) <= h else [1]
            stride = stride + [2] if 2 ** (depth/2 - i) <= w else stride + [1]
            strides.append(stride)
            conv_ksizes.append([3, 3])
            deconv_ksizes.append([_ ** 2 for _ in stride])
        convs = [nn.Sequential(nn.Conv2d(in_shape[0], num_channels,
                                        tuple(conv_ksizes[0]),
                                        tuple(strides[0]),
                                        (int((conv_ksizes[0][0] - 1)/2), int((conv_ksizes[0][1] - 1)/2))),
                               nn.BatchNorm2d(num_channels),
                               nn.ReLU(True))]
        for i in range(1, int(depth / 2)):
            convs.append(nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                                tuple(conv_ksizes[i]),
                                                tuple(strides[i]),
                                                (int((conv_ksizes[i][0] - 1)/2), int((conv_ksizes[i][1] - 1)/2))),
                                       nn.BatchNorm2d(num_channels),
                                       nn.ReLU(True)))
        this.convs = nn.Sequential(*convs)
        # deconvs
        deconvs = []
        for i in range(1, int(depth / 2)):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
                                                           tuple(deconv_ksizes[int(depth/2)-i]),
                                                           tuple(strides[int(depth/2)-i]),
                                                           (int(deconv_ksizes[int(depth/2)-i][0]/4.), int(deconv_ksizes[int(depth/2)-i][1]/4.))),
                                         nn.BatchNorm2d(num_channels),
                                         nn.ReLU(True)))
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
                                                       tuple(deconv_ksizes[0]),
                                                       tuple(strides[0]),
                                                       (int(deconv_ksizes[0][0]/4.), int(deconv_ksizes[0][1]/4.))),
                                     nn.BatchNorm2d(num_channels),
                                     nn.ReLU(True))
                       )
        this.deconvs = nn.Sequential(*deconvs)
    def forward(this, input):
        x = input[0]
        for i in range(0, len(this.fpn)):
            x = this.fpn[i](x) + input[i+1]
        conv_feats = []
        for i in range(0, len(this.convs)):
            x = this.convs[i](x)
            conv_feats.append(x)
        for i in range(0, len(this.deconvs) - 1):
            x = this.deconvs[i](x)
            f=conv_feats[len(conv_feats) - 2 - i]
            x = x[:,:,:f.shape[2],:f.shape[3]] + f
        x = this.deconvs[-1](x)

        BS,CS,H,W=x.shape;
        emb=this.se(x.shape[3],x.shape[2]).repeat(BS,1,1,1);
        x=torch.cat([x,emb],dim=1)
        BS,CS,H,W=x.shape;
        keys,x=this.attms(this.keys.repeat([BS,1,1]),x.permute(0,2,3,1).reshape(BS,-1,CS))
        a=torch.softmax(keys.matmul(x.permute(0,2,1)),dim=2);
        return a.reshape(BS,this.maxT,H,W)
