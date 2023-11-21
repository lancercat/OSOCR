from torch import nn;
from neko_sdk.ocr_modules.neko_interprinter import neko_visual_only_interprinter;
from neko_sdk.ocr_modules.neko_score_merging import scatter_cvt;
import torch;
from torch.nn import functional as F;


'''
Decoupled Text Decoder
'''
class neko_os_DTD_mk4(nn.Module):
    # LSTM DTD

    def normed_init(this,nchannel):
        input=torch.rand(1,nchannel)
        return input/torch.norm(input,dim=-1,keepdim=True);
    def setup_modules(this, dropout = 0.3):
        pass;

    def __init__(this, nchannel, dropout = 0.3,xtraparams=None):
        super(neko_os_DTD_mk4,this).__init__();
        this.nchannel = nchannel;
        this.dropout=dropout;
        this.xtraparams=xtraparams;
        this.setup_modules();
        this.baseline=0;



    def h1stt(this,nB):
        return this.STA.index_select(0, torch.zeros(nB).long().to(this.STA.device))

    def pick(this, tens, emb,plabel, hype, i):
        chemb = torch.cat([emb, this.UNK]);
        if (hype is None):
            t1=tens.topk(1)[1].squeeze()
            ssel = chemb.index_select(0, t1);
            ret = ssel;
        else:
            ret = chemb.index_select(0, hype[:, i]);
        return ret;

    def loop(this, C, proto,semb,plabel, nsteps, nB, hype ):
        out_res_ = torch.zeros(nsteps, nB, proto.shape[0] + 1).type_as(C.data) + this.UNK_SCR;
        gru_res = torch.zeros(C.size()).type_as(C.data)
        hidden = torch.zeros(nB, this.nchannel).type_as(C.data)
        prev_emb = [this.h1stt(nB)];
        for i in range(0, nsteps):
            hidden = this.rnn(torch.cat((C[i, :, :], prev_emb[-1]), dim=1),
                              hidden)
            gru_res[i, :, :] = hidden
            raw_scores = torch.matmul(hidden, proto.t());
            tens = torch.cat([raw_scores * this.ALPHA + this.BIAS, this.UNK_SCR.repeat(nB, 1)], dim=-1);
            out_res_[i, :, :] = tens;
            tens = scatter_cvt(tens,plabel);
            # Soft selection
            prev_emb.append(this.pick(tens,semb,plabel,hype,i))
        return out_res_;


    def prob_length(this,out_res,nsteps,nB):
        out_length = torch.zeros(nB)
        for i in range(0, nsteps):
            tens=out_res[i, :, :];
            tmp_result = tens.topk(1)[1].squeeze(-1)
            for j in range(nB):
                if out_length[j].item() == 0 and tmp_result[j] == 0:
                    out_length[j] = i + 1
        for j in range(nB):
            if out_length[j] == 0:
                out_length[j] = nsteps + 1
        return out_length

    def pred(this,out_res_,label,out_length,nB,nT):
        scores = scatter_cvt(out_res_, label);
        start = 0
        output = torch.zeros(int(out_length.sum()), label.max().item()+1).type_as(out_res_.data)
        for i in range(0, nB):
            cur_length = int(out_length[i])
            cur_length_=cur_length
            if(cur_length_>nT):
                cur_length_=nT;
            output[start: start + cur_length_] = scores[0: cur_length_, i, :]
            # if(scores[cur_length_-1, i, :].argmax().item()!=0):
            #     print("???")
            start += cur_length
        return output;


    def forward_test(this,proto,semb,label,nB,C,nT):
        # unknown is a fact, not a similarity.
        out_res_= this.loop(C, proto, semb,label,nT, nB, None);
        out_length=this.prob_length(out_res_,nT,nB);
        output=this.pred(out_res_,label,out_length,nB,nT);
        return output, out_length

    def out_attns(this,text_length,A,nB,nH,nW):
        lenText = int(text_length.sum())
        start = 0
        out_attns = torch.zeros(lenText, nH, nW).type_as(A.data)
        for i in range(0, nB):
            cur_length = int(text_length[i])
            out_attns[start: start + cur_length] = A[i, 0:cur_length, :, :]
            start += cur_length
        return out_attns;


    def randemb(this,emb):
        rweight=torch.softmax(torch.rand([1,emb.shape[0]]),dim=1).to(emb.device);
        return rweight.matmul(emb);

    def forward_train_hypeless(this,proto,semb,label,nB,C,nT,text_length,A,nW,nH):
        nsteps = int(text_length.max())
        # unknown is a fact, not a similarity.
        out_res_= this.loop(C, proto,semb,label, nsteps, nB, None)
        out_attns=this.out_attns(text_length,A,nB,nH,nW);
        output=this.pred(out_res_,label,text_length,nB,nT);
        return output, out_attns

    def forward_train_hyped(this, proto,semb, label, nB, C, nT, text_length, A, nW, nH,hype):
        nsteps = int(text_length.max())
        # unknown is a fact, not a similarity.
        out_res_ = this.loop(C, proto,semb,label, nsteps, nB, hype);
        out_attns = this.out_attns(text_length, A, nB, nH, nW);
        output = this.pred(out_res_, label, text_length, nB, nT);
        return output, out_attns

    def getC(this, feature, A, nB, nC, nH, nW, nT):
        C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
        C = C.view(nB, nT, nC, -1).sum(3).transpose(1, 0);
        return C;

    def sample(this,feature,A):
        return None,None



    def forward(this, feature,protos,semb,labels, A, hype, text_length, test = False):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        A,C=this.sample(feature,A);
        C = torch.nn.functional.dropout(C, p=0.3, training=this.training)
        if not test:
            if(hype is not None):
                return this.forward_train_hyped(protos,semb,labels,nB,C,nT,text_length,A,nW,nH,hype);
            else:
                return this.forward_train_hypeless(protos,semb,labels,nB,C,nT,text_length,A,nW,nH);
        else:
            out,outl=this.forward_test(protos,semb,labels,nB,C,nT)
            return out,outl,A;


