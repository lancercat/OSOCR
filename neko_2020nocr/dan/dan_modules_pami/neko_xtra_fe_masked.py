import torch
import torch.nn as nn
import neko_sdk.encoders.ocr_networks.dan.dan_reslens_naive_masked as rescco

class neko_cco_Feature_Extractor_masked(nn.Module):
    def __init__(self, strides, compress_layer, input_shape,hardness=2,oupch=512):
        super(neko_cco_Feature_Extractor_masked, self).__init__()
        self.model = rescco.res_naive_lens45_masked(strides, compress_layer,hardness,oupch=oupch,inpch=input_shape[0])
        self.input_shape = input_shape

    def forward(self, input,mask,debug=False):
        features,mask,grid = self.model(input,mask)
        if debug:
            return features,mask,grid;
        return features,mask

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        pseudo_mask = torch.ones(1, 1, self.input_shape[1], self.input_shape[2])
        features,_,_ = self.model(pseudo_input,pseudo_mask)
        return [feat.size()[1:] for feat in features]
