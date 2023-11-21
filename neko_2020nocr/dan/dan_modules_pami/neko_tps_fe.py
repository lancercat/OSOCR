import torch
import torch.nn as nn
import neko_sdk.encoders.ocr_networks.dan.dan_resnet as resnet
from neko_sdk.ocr_modules.transformations.tps import GridGenerator,LocalizationNetwork
import torch.nn.functional as trnf
import neko_sdk.encoders.ocr_networks.dan.dan_resbase as rescco

class neko_tps(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(this, F, irsize, I_channel_num=1):
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(neko_tps, this).__init__()
        this.F = F
        this.I_r_size = irsize;
        this.I_channel_num = I_channel_num
        this.LocalizationNetwork = LocalizationNetwork(this.F, this.I_channel_num)
        this.GridGenerator = GridGenerator(this.F, this.I_r_size)

    def forward(this, batch_I):
        batch_C_prime = this.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = this.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), this.I_r_size[0], this.I_r_size[1], 2])

        if torch.__version__ > "1.2.0":
            batch_I_r = trnf.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border', align_corners=True)
        else:
            batch_I_r = trnf.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

        return batch_I_r


'''
Lens_Feature_Extractor
'''
class neko_tps_Feature_Extractor(nn.Module):
    def __init__(this, strides, compress_layer, input_shape,oupch=512,expf=1):
        super(neko_tps_Feature_Extractor, this).__init__()
        this.tps=neko_tps(8,input_shape[1:],input_shape[-3]);
        this.model = resnet.resnet45(strides, compress_layer,oupch=oupch,inpch=input_shape[0],expf=expf)
        this.input_shape = input_shape

    def forward(this, input,debug=False):
        rip=this.tps(input);
        features = this.model(rip)
        if debug:
            return features;
        return features

    def Iwantshapes(this):
        pseudo_input = torch.rand(1, this.input_shape[0], this.input_shape[1], this.input_shape[2])
        features = this.forward(pseudo_input)
        return [feat.size()[1:] for feat in features]

class neko_tps_Feature_ExtractorF(nn.Module):
    def __init__(this, strides, compress_layer, input_shape, oupch=512, expf=1):
        super(neko_tps_Feature_ExtractorF, this).__init__()
        this.tps = neko_tps(8, input_shape[1:], input_shape[-3]);
        this.model =rescco.res_base(strides, compress_layer,None,oupch=oupch,inpch=input_shape[0],expf=expf)
        this.input_shape = input_shape

    def forward(this, input, debug=False):
        rip = this.tps(input);
        features,_ = this.model(rip)
        if debug:
            return features;
        return features

    def Iwantshapes(this):
        pseudo_input = torch.rand(1, this.input_shape[0], this.input_shape[1], this.input_shape[2])
        features= this.forward(pseudo_input)
        return [feat.size()[1:] for feat in features]

