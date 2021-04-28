from neko_sdk.ocr_modules.neko_score_merging import scatter_cvt;
import torch;
from neko_2020nocr.dan.dan_modules.neko_os_DTD_mk4 import neko_os_DTD_mk4;

class neko_os_DTD_mk4_cclcf(neko_os_DTD_mk4):
    # LSTM DTD

    def setup_modules(this, dropout = 0.3):
        this.STA = torch.nn.Parameter(this.normed_init(this.nchannel));
        this.UNK = torch.nn.Parameter(this.normed_init(this.nchannel));
        this.UNK_SCR = torch.nn.Parameter(torch.zeros([1, 1]), requires_grad=True);
        this.ALPHA = torch.nn.Parameter(torch.ones([1, 1]) , requires_grad=True);
        this.context_free_pred = torch.nn.Linear(this.nchannel, this.nchannel);

        this.register_parameter("STA", this.STA);
        this.register_parameter("UNK", this.UNK);
        this.register_parameter("UNK_SCR", this.UNK_SCR);
        this.register_parameter("ALPHA", this.ALPHA);

    def loop(this, C, proto,semb,plabel, nsteps, nB, hype ):
        # this.UNK_SCR=torch.nn.Parameter(torch.zeros_like(this.UNK_SCR)-100.)
        out_res_cf = torch.zeros(nsteps, nB, proto.shape[0] + 1).type_as(C.data) + this.UNK_SCR;
        sim_score = torch.zeros(nsteps, nB, proto.shape[0] + 1).type_as(C.data) + this.UNK_SCR;
        # hidden=C;
        hidden=this.context_free_pred(C);
        cfpred=hidden.matmul(proto.t());
        # beforsum=(hidden.unsqueeze(-1)*proto.t().unsqueeze(0).unsqueeze(0)).permute(0,1,3,2);
        # cfpred=beforsum.reshape([beforsum.shape[0], beforsum.shape[1], beforsum.shape[2], 4, beforsum.shape[3] // 4]).min(3)[
        #     0].sum(3);

        cfcos=cfpred/(hidden.norm(dim=-1,keepdim=True)+0.0009);

        out_res_cf[:nsteps, :, :]=torch.cat([cfpred[:nsteps,:,:]*this.ALPHA, this.UNK_SCR.repeat(nsteps,nB, 1)], dim=-1)
        sim_score[:nsteps, :, :-1]=cfcos[:nsteps,:,:];

            # Soft selection
        return out_res_cf,sim_score;


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
        out_res_cls,_= this.loop(C, proto, semb,label,nT, nB, None);
        out_length=this.prob_length(out_res_cls,nT,nB);
        output=this.pred(out_res_cls,label,out_length,nB,nT);
        return output, out_length

    def forward_train_hypeless(this,proto,semb,label,nB,C,nT,text_length,A,nW,nH):
        nsteps = int(text_length.max())
        # unknown is a fact, not a similarity.
        out_res_cls,out_res_cos= this.loop(C, proto,semb,label, nsteps, nB, None)
        # out_attns=this.out_attns(text_length,A,nB,nH,nW);
        output_cls=this.pred(out_res_cls,label,text_length,nB,nT);
        output_cos=this.pred(out_res_cos,label,text_length,nB,nT);
        return output_cls, output_cos

    def forward_train_hyped(this, proto,semb, label, nB, C, nT, text_length, A, nW, nH,hype):
        nsteps = int(text_length.max())
        # unknown is a fact, not a similarity.
        out_res_cls,out_res_cos = this.loop(C, proto,semb,label, nsteps, nB, hype);
        # out_attns = this.out_attns(text_length, A, nB, nH, nW);
        output_cls = this.pred(out_res_cls, label, text_length, nB, nT);
        output_cos = this.pred(out_res_cos, label, text_length, nB, nT);
        return output_cls, output_cos
    def sample(this,feature,A):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        # Normalize
        A = A / A.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1)
        # weighted sum
        C = this.getC(feature, A, nB, nC, nH, nW, nT);
        return A, C;
        pass;
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
