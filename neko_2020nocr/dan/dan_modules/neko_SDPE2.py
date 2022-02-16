import torch;
from torch.nn import functional as trnf
from neko_sdk.ocr_modules.prototypers.neko_nonsemantical_prototyper_core import neko_nonsematical_prototype_core_basic
import regex
class neko_SDPE2(torch.nn.Module):
    def setupcore(this,backbone=None,val_frac=0.8):
        try:
            meta = torch.load(this.meta_path);
        except:
            meta=None;
            print("meta loading failed")
        this.dwcore = neko_nonsematical_prototype_core_basic(this.nchannel, meta, backbone, None,
                                                {"master_share": not this.case_sensitive,
                                                 "max_batch_size": 512,
                                                 "val_frac": val_frac,
                                                 "neg_servant" :True
                                                 },dropout=0.3);

    def __init__(this, meta_path, nchannel, case_sensitive,backbone=None,val_frac=0.8):
        super(neko_SDPE2, this).__init__();
        this.EOS = 0;
        this.nchannel = nchannel;
        this.case_sensitive = case_sensitive;
        this.meta_path=meta_path;
        this.setupcore(backbone,val_frac);


    def dump_all(this):
        return this.dwcore.dump_all();
    def sample_tr(this,text_batch):
        # if(not this.case_sensitive):
        #     label_batch=[l.lower() for l in text_batch]
        return this.dwcore.sample_charset_by_text(text_batch);
    def encode_fn_naive(this,tdict,label_batch):
        max_len = max([len(regex.findall(r'\X', s, regex.U)) for s in label_batch])
        out = torch.zeros(len(label_batch), max_len + 1).long() + this.EOS
        for i in range(0, len(label_batch)):
            cur_encoded = torch.tensor([tdict[char] if char in tdict else tdict["[UNK]"]
                                        for char in regex.findall(r'\X', label_batch[i], regex.U)])
            out[i][0:len(cur_encoded)] = cur_encoded
        return out
    def encode(this,proto,plabel,tdict,label_batch):
        if(not this.case_sensitive):
            label_batch=[l.lower() for l in label_batch]
        return this.encode_fn_naive(tdict,label_batch)

    def decode(this, net_out, length,protos,labels,tdict):
    # decoding prediction into text with geometric-mean probability
    # the probability is used to select the more realiable prediction when using bi-directional decoders
        out = []
        out_prob = []
        net_out = trnf.softmax(net_out, dim = 1)
        for i in range(0, length.shape[0]):
            current_idx_list = net_out[int(length[:i].sum()) : int(length[:i].sum() + length[i])].topk(1)[1][:,0].tolist()
            current_text = ''.join([tdict[_] if _ > 0 and _ <= len(tdict) else '' for _ in current_idx_list])
            current_probability = net_out[int(length[:i].sum()) : int(length[:i].sum() + length[i])].topk(1)[0][:,0]
            current_probability = torch.exp(torch.log(current_probability).sum() / current_probability.size()[0])
            out.append(current_text)
            out_prob.append(current_probability)
        return (out, out_prob)

