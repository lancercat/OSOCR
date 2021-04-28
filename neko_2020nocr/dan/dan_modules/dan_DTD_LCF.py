from torch import nn
import torch
from torch.nn import Parameter
from torch.nn import functional as F
'''
Decoupled Text Decoder
'''
class DTDLCF(nn.Module):
    # LSTM DTD
    def __init__(self, nclass, nchannel, dropout = 0.3):
        super(DTDLCF, self).__init__()
        self.nclass = nclass
        self.nchannel = nchannel
        self.generator = nn.Sequential(
                            nn.Linear(nchannel, nclass)
                        )
        self.char_embeddings = Parameter(torch.randn(nclass, nchannel))

    def forward(self, feature, A, text, text_length, test = False):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        # Normalize
        A = A / A.view(nB, nT, -1).sum(2).view(nB,nT,1,1)
        # weighted sum
        # C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
        # C = C.view(nB,nT,nC,-1).sum(3).transpose(1,0)
        C= feature.view(nB,nC,nW*nH).matmul(A.permute(0,2,3,1).reshape(nB,nH*nW,nT))
        C = F.dropout(C, p = 0.3, training=self.training,inplace=True).permute(2,0,1)
        out_res_=self.generator(C);
        if not test:
            lenText = int(text_length.sum())
            out_res = torch.zeros(lenText, self.nclass).type_as(feature.data)
            start = 0
            for i in range(0, nB):
                cur_length = int(text_length[i])
                out_res[start: start + cur_length] = out_res_[0: cur_length, i, :]
                start += cur_length

            return out_res, None;
        else:
            out_res=out_res_;
            nsteps = nT
            out_length = torch.zeros(nB)
            now_step = 0
            while 0 in out_length and now_step < nsteps:
                tmp_result = out_res[now_step].topk(1)[1].squeeze()
                for j in range(nB):
                    if out_length[j] == 0 and tmp_result[j] == 0:
                        out_length[j] = now_step + 1
                now_step += 1
            for j in range(0, nB):
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps
            start = 0
            output = torch.zeros(int(out_length.sum()), self.nclass).type_as(feature.data)
            for i in range(0, nB):
                cur_length = int(out_length[i])
                output[start : start + cur_length] = out_res[0: cur_length,i,:]
                start += cur_length
            return output, out_length
