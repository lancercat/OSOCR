from torch_scatter import scatter_max,scatter_mean;
def id_cvt(pred,label):
    return pred;
def scatter_cvt_d(pred,label,dim=-1):
    dev=pred.device;
    label=label.long().to(dev);
    pred=pred.cpu();
    label=label.cpu();
    return scatter_max(pred,label,dim)[0].cuda();

def scatter_cvt(pred, label, dim=-1):
    ###return scatter_cvt_d(pred,label,dim)
    dev = pred.device;
    label = label.long().to(dev);
    return scatter_max(pred,label,dim)[0];
