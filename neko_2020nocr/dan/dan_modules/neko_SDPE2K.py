import torch;
from torch.nn import functional as trnf
from neko_sdk.ocr_modules.prototypers.neko_nonsemantical_prototyper_core_xtra import neko_nonsematical_prototype_core_basic,\
    neko_nonsematical_prototype_core_inv,neko_nonsematical_prototype_core_HD,neko_nonsematical_prototype_core_castle
import regex
class neko_SDPE2(torch.nn.Module):
    def setupcore(this):
        meta = torch.load(this.meta_path);
        this.dwcore = neko_nonsematical_prototype_core_basic(this.nchannel, meta, None, None,
                                                {"master_share": not this.case_sensitive,
                                                 "max_batch_size": 512,
                                                 "val_frac": 0.8,
                                                 "neg_servant" :True
                                                 },dropout=0.3);

    def __init__(this, meta_path, nchannel, case_sensitive):
        super(neko_SDPE2, this).__init__();
        this.EOS = 0;
        this.nchannel = nchannel;
        this.case_sensitive = case_sensitive;
        this.meta_path=meta_path;
        this.setupcore();


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

class neko_SDPE2S(neko_SDPE2):

    def sample_tr(this,text_batch):
        # if(not this.case_sensitive):
        #     label_batch=[l.lower() for l in text_batch]
        return this.dwcore.sample_charset_by_text_both(text_batch);

    def encode_semi(this,proto,plabel,tdict,label_batch):
        calb=[l.lower() for l in label_batch];
        cslb=label_batch;
        lab=[calb,cslb];
        ret=[this.encode_fn_naive(tdict[i],lab[i]) for i in [0,1]];
        return ret;
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

class neko_SDPE2I(torch.nn.Module):
    def setupcore(this):
        meta = torch.load(this.meta_path);
        this.dwcore = neko_nonsematical_prototype_core_inv(this.nchannel, meta, None, None,
                                                {"master_share": not this.case_sensitive,
                                                 "max_batch_size": 512,
                                                 "val_frac": 0.8,
                                                 "neg_servant" :True
                                                 },dropout=0.3);

    def __init__(this, meta_path, nchannel, case_sensitive):
        super(neko_SDPE2I, this).__init__();
        this.EOS = 0;
        this.nchannel = nchannel;
        this.case_sensitive = case_sensitive;
        this.meta_path=meta_path;
        this.setupcore();


    def dump_all(this):
        return this.dwcore.dump_all();
    def sample_tr(this,text_batch):
        # if(not this.case_sensitive):
        #     label_batch=[l.lower() for l in text_batch]
        return this.dwcore.sample_charset_by_text(text_batch);

    def encode(this,proto,plabel,tdict,label_batch):
        if(not this.case_sensitive):
            label_batch=[l.lower() for l in label_batch]
        max_len = max([len(regex.findall(r'\X', s, regex.U) ) for s in label_batch])
        out = torch.zeros(len(label_batch), max_len+1).long()+this.EOS
        for i in range(0, len(label_batch)):
            cur_encoded = torch.tensor([tdict[char] if char in tdict else tdict["[UNK]"]
                                     for char in regex.findall(r'\X', label_batch[i], regex.U) ])
            out[i][0:len(cur_encoded)] = cur_encoded
        return out

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


class neko_SDPE2C(neko_SDPE2):
    def setupcore(this):
        meta = torch.load(this.meta_path);
        this.dwcore = neko_nonsematical_prototype_core_castle(this.nchannel, meta, None, None,
                                                {"master_share": not this.case_sensitive,
                                                 "max_batch_size": 512,
                                                 "val_frac": 0.8,
                                                 "neg_servant": True
                                                 },dropout=0.3);




class neko_SDPE2K(neko_SDPE2):
    def setupcore(this):
        meta = torch.load(this.meta_path);
        this.dwcore = neko_nonsematical_prototype_core_basic(this.nchannel, meta, None, None,
                                                {"master_share": not this.case_sensitive,
                                                 "max_batch_size": 512,
                                                 "val_frac": 0.8,
                                                 "neg_servant": False
                                                 },dropout=0.3);
class neko_SDPE2H(torch.nn.Module):
    def setupcore(this):
        meta = torch.load(this.meta_path);
        this.dwcore = neko_nonsematical_prototype_core_HD(this.nchannel//4, meta, None, None,
                                                {"master_share": not this.case_sensitive,
                                                 "max_batch_size": 512,
                                                 "val_frac": 0.8,
                                                 "neg_servant": True
                                                 },dropout=0.3);

    def __init__(this, meta_path, nchannel, case_sensitive):
        super(neko_SDPE2H, this).__init__();
        this.EOS = 0;
        this.nchannel = nchannel;
        this.case_sensitive = case_sensitive;
        this.meta_path=meta_path;
        this.setupcore();


    def dump_all(this):
        return this.dwcore.dump_all();
    def sample_tr(this,text_batch):
        return this.dwcore.sample_charset_by_text(text_batch);

    def encode(this,proto,plabel,tdict,label_batch):
        max_len = max([len(regex.findall(r'\X', s, regex.U) ) for s in label_batch])
        out = torch.zeros(len(label_batch), max_len+1).long()+this.EOS
        for i in range(0, len(label_batch)):
            cur_encoded = torch.tensor([tdict[char] if char in tdict else tdict["[UNK]"]
                                     for char in regex.findall(r'\X', label_batch[i], regex.U) ])
            out[i][0:len(cur_encoded)] = cur_encoded
        return out

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
