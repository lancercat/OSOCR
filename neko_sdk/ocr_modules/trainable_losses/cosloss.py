from torch import nn;
import torch;

class neko_cos_loss(nn.Module):
    def __init__(this):
        super(neko_cos_loss, this).__init__()
        pass;
    def forward(this,pred,gt):
        oh=torch.zeros_like(pred).scatter_(1, gt.unsqueeze(-1), 1)[:,:-1];
        noh=1-oh;
        pred_=pred[:,:-1];
        pwl = torch.nn.functional.smooth_l1_loss(pred_, oh, reduction="none");
        nl = torch.sum(noh * pwl) / (torch.sum(noh) + 0.009);
        pl = torch.sum(oh * pwl) / (torch.sum(oh) + 0.009)
        return (nl+pl)/2;

class neko_cos_loss2(nn.Module):
    def __init__(this):
        super(neko_cos_loss2, this).__init__()
        pass;
    def forward(this,pred,gt):
        oh=torch.zeros_like(pred).scatter_(1, gt.unsqueeze(-1), 1)[:,:-1];
        noh=1-oh;
        pred_=pred[:,:-1];
        corr=oh*pred_;

        ### Only classes too close to each other should be pushed.
        ### That said if they are far we don't need to align them
        ### Again 0.14 =cos(spatial angluar on 50k evenly distributed prototype)
        wrong=torch.nn.functional.relu(noh*pred_-0.14)
        nl=torch.sum(wrong)/(torch.sum(noh)+0.009);
        pl=1-torch.sum(corr)/(torch.sum(oh)+0.009)
        return (nl+pl)/2;

