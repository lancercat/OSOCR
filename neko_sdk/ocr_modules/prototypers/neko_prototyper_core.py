import torch;
from torch import nn;
import random;
from neko_sdk.ocr_modules.neko_interprinter import neko_visual_only_interprinter,\
    neko_structural_visual_only_interprinter,neko_weird_visual_only_interprinter,\
    neko_visual_only_interprinterR34
from neko_sdk.ocr_modules.prototypers.neko_visual_center_prototyper import neko_abstract_visual_center_prototyper;
import numpy as np;
from collections import deque as python_queue;
# this class defines how samples are sampled ^_^



class neko_prototype_core_basic(nn.Module):
    # every thing does not involve sampling
    PROTOENGINE=neko_visual_only_interprinter;
    def make_proto_engine(this,meta,backbone,preload_tensor=None):
        this.proto_engine = this.PROTOENGINE(this.output_channel, backbone);
        this.prototype_cnt = -1;
        # handles Capitalize like problem.
        # In some annotations they have same labels, while in others not.
        # i.e, 'A' and 'a' can have the same label 'a',
        # '茴','回','囘' and '囬' can have the same label '回'
        if(meta is not None):
            this.masters = meta["master"];

            # Foes includes the characters looks like each other
            # but never share labels (They may actually have linguistic relationships...
            # Like yanderes in a broken relationship[x]).
            # This set helps implement ohem like minibatching on the huge labelset.
            # e.g. 'u' and 'ü'
            this.foes = meta["foes"];
            this.servants=meta["servants"];
            # union set of friend, harem and foe.
            this.related_proto_ids = meta["relationships"];

        this.sp_protos = torch.nn.Parameter(torch.rand([
            this.sp_cnt, this.output_channel]).float() * 2 - 1);
        this.register_parameter("sp_proto", this.sp_protos);




    def setup_common(this,output_channel,
                 meta,
                 backbone=None,
                 preload_tensor=None,
                dropout=None):
        this.output_channel = output_channel;
        # Generally, CUDA devices works well with dynamic batch size.
        # However, for Rocm devices it bites as it compiles kernel
        # everytime we change batch size it's nightmare.
        this.dev_ind = torch.nn.Parameter(torch.rand([1]));
        list_character = list(meta["chars"]);
        this.aligned_characters=meta["achars"];
        # characters without shape is generally what you do now want to sample.
        this.shaped_characters = set(meta["chars"])
        # UNK is not a sp_token as it is centerless.
        this.character = list(meta["sp_tokens"]) + list_character;
        this.label_dict = meta["label_dict"];
        this.shaped_ids=set([this.label_dict[i] for i in this.shaped_characters]);
        this.sp_cnt = len(meta["sp_tokens"]);
        this.sp_tokens=meta["sp_tokens"];
        if(dropout is not None):
            this.drop=torch.nn.Dropout(p=0.3);
        else:
            this.drop=None;
        unk = this.label_dict["[UNK]"];
        # if the dict does not provide an specific unk token, set it to -1;
        for i, char in enumerate(this.character):
            # print(i, char)
            this.label_dict[char] = i;
        # shapeless unk shall be excluded
        if (unk < 0):
            this.label_set = set(this.label_dict.values()) - {unk};
        else:
            this.label_set = set(this.label_dict.values());

        if (preload_tensor is None):
            this.norm_protos = meta["protos"][this.sp_cnt:];
        else:
            this.norm_protos = torch.load(preload_tensor);

        for i in range(len(this.norm_protos)):
            if this.norm_protos[i] is not None and this.norm_protos[i].max() > 20:
                this.norm_protos[i] = (this.norm_protos[i] - 127.5) / 128;

        this.make_proto_engine(meta, backbone, preload_tensor=None);
    # defines sampler
    def setup_sampler(this,sampler_args):
        if sampler_args is None:
            masters_share = False;
            max_match_size=512;
            val_frac=0.8;
            neg_servant=True;
        else:
            masters_share = sampler_args["master_share"];
            max_match_size = sampler_args["max_batch_size"];
            val_frac=sampler_args["val_frac"];
            neg_servant=sampler_args["neg_servant"];
        this.masters_share=masters_share;
        this.max_batch_size=max_match_size;
        this.val_frac=val_frac;
        this.neg_servant=neg_servant;


    def __init__(this,
                 output_channel,
                 meta,
                 backbone=None,
                 preload_tensor=None,
                 sampler_args=None,
                 dropout=None,
                 ):
        print("DEBUG-SDFGASDFGSDGASFGSD",dropout);
        super(neko_prototype_core_basic, this).__init__();
        this.setup_common(output_channel,meta,backbone,preload_tensor,dropout);
        this.setup_sampler(sampler_args);


    def debug(this,normpids,labels):
        normprotos = [this.norm_protos[i - this.sp_cnt] for i in normpids];
        protos=((torch.cat(normprotos, dim=-1).squeeze(0).squeeze(0)+1)*127.5).detach().cpu().numpy().astype(np.uint8);
        import cv2;
        cv2.imshow(labels,protos[:,:32*32])
        cv2.waitKey(0);

    def get_protos(this,sppids,normpids):
        normprotos=[this.norm_protos[i-this.sp_cnt] for i in normpids];
        spprotos=[this.sp_protos[i].unsqueeze(0) for i in sppids];
        normprotos=this.proto_engine(torch.cat(normprotos).repeat(1,3,1,1).to(this.dev_ind.device));
        allproto=torch.cat(spprotos+[normprotos]);
        if(this.drop):
            allproto=this.drop(allproto);

        return allproto/torch.norm(allproto,dim=-1,keepdim=True);
    def get_plabel_and_dict(this,sappids,normpids):
        all_ids=sappids.union(normpids);
        new_id=0;
        plabels=[];
        labmap={};
        bidict={}
        for i in all_ids:
            cha=this.aligned_characters[i];
            if(this.masters_share):
                vlab=this.masters[i];
            else:
                vlab=i;
            if(vlab not in labmap):
                labmap[vlab]=new_id;
                # A new label
                new_id+=1;
            alab=labmap[vlab];
            plabels.append(alab);
            bidict[alab]=cha;
            bidict[cha]=alab;
        plabels.append(new_id)
        bidict["[UNK]"]=new_id;
        bidict[new_id]="⑨";

        return torch.tensor(plabels),bidict;
    def grab_cluster(this,ch):
        chid=this.label_dict[ch];
        ret={chid};
        if this.masters_share:
            ret.add(this.masters[chid]);
            ret=ret.union(this.servants[this.masters[chid]]);
        return ret;

    def get_sampled_ids(this,plain_chars_in_data):
        cntval = int(len(plain_chars_in_data) * this.val_frac);
        cntval = min(this.max_batch_size - this.sp_cnt, cntval);
        trchs=set();
        related_chars_in_data=set();
        random.shuffle(plain_chars_in_data);
        # make sure no missing centers--
        # or it may enforce "A" to look like "a" encoded by proto CNN
        remaining = cntval;
        for ch in plain_chars_in_data:
            if(ch not in this.label_dict):
                continue;
            new=this.grab_cluster(ch);
            ns=trchs.union(new);
            related_chars_in_data=related_chars_in_data.union(new);
            delta=len(ns)-len(trchs);
            if(delta<=remaining):
                trchs=ns;
                remaining-=delta;
        remaining=this.max_batch_size-this.sp_cnt-len(trchs);
        plain_charid_not_in_data=list(this.shaped_ids-related_chars_in_data);
        random.shuffle(plain_charid_not_in_data);
        for chid in plain_charid_not_in_data:
            if chid not in trchs:
                if (remaining == 0):
                    break;
                if (this.neg_servant==False and this.masters[chid]!=chid):
                    continue;
                remaining-=1;
                trchs.add(chid);

        trsps=set([this.label_dict[i] for i in this.sp_tokens]);
        return trsps,trchs;

    def sample_charset_by_text(this,text_batch):
        b="";
        for _ in text_batch: b+=_;
        plain_chars_in_data=list(set(regex.findall(r'\X', b, regex.U)));
        trsps,trchs=this.get_sampled_ids(plain_chars_in_data);
        protos=this.get_protos(trsps,trchs);
        plabels,tdicts=this.get_plabel_and_dict(trsps,trchs)
        return protos,plabels,tdicts;

    def dump_all(this):
        trsps=set([this.label_dict[i] for i in this.sp_tokens]);
        trchs=set([this.label_dict[i] for i in this.shaped_characters]);

        protos=this.get_protos(trsps,trchs);
        plabels,tdicts=this.get_plabel_and_dict(trsps,trchs)
        return protos,plabels,tdicts;

class neko_prototype_core_basic_shared(nn.Module):

    def make_proto_engine(this, meta, core):
        this.proto_engine = core.proto_engine;
        this.prototype_cnt = -1;
        # handles Capitalize like problem.
        # In some annotations they have same labels, while in others not.
        # i.e, 'A' and 'a' can have the same label 'a',
        # '茴','回','囘' and '囬' can have the same label '回'
        this.masters = meta["master"];
        # Foes includes the characters looks like each other
        # but never share labels (They may actually have linguistic relationships...
        # Like yanderes in a broken relationship[x]).
        # This set helps implement ohem like minibatching on the huge labelset.
        # e.g. 'u' and 'ü'
        this.foes = meta["foes"];
        this.servants = meta["servants"];
        # union set of friend, harem and foe.
        this.related_proto_ids = meta["relationships"];

        this.sp_protos = core.sp_protos;
    def setup_common(this,
                     meta,
                     core):
        this.masters_share=True;
        this.output_channel = core.output_channel;
        # Generally, CUDA devices works well with dynamic batch size.
        # However, for Rocm devices it bites as it compiles kernel
        # everytime we change batch size it's nightmare.
        this.dev_ind = torch.nn.Parameter(torch.rand([1]));
        list_character = list(meta["chars"]);
        this.aligned_characters = meta["achars"];
        # characters without shape is generally what you do now want to sample.
        this.shaped_characters = set(meta["chars"])
        # UNK is not a sp_token as it is centerless.
        this.character = list(meta["sp_tokens"]) + list_character;
        this.label_dict = meta["label_dict"];
        this.shaped_ids = set([this.label_dict[i] for i in this.shaped_characters]);
        this.sp_cnt = len(meta["sp_tokens"]);
        this.sp_tokens = meta["sp_tokens"];
        this.drop=core.drop;
        unk = this.label_dict["[UNK]"];
        # if the dict does not provide an specific unk token, set it to -1;
        for i, char in enumerate(this.character):
            # print(i, char)
            this.label_dict[char] = i;
        # shapeless unk shall be excluded
        if (unk < 0):
            this.label_set = set(this.label_dict.values()) - {unk};
        else:
            this.label_set = set(this.label_dict.values());

        this.norm_protos = meta["protos"][this.sp_cnt:];

        for i in range(len(this.norm_protos)):
            if this.norm_protos[i] is not None and this.norm_protos[i].max() > 20:
                this.norm_protos[i] = (this.norm_protos[i] - 127.5) / 128;

        this.make_proto_engine(meta, core);


    def __init__(this,
                 meta,
                 core
                  ):
        super(neko_prototype_core_basic_shared, this).__init__();
        this.setup_common(meta,core);

    def debug(this, normpids, labels):
        normprotos = [this.norm_protos[i - this.sp_cnt] for i in normpids];
        protos = ((torch.cat(normprotos, dim=-1).unsqueeze(0).unsqueeze(1) + 1) * 127.5).detach().cpu().numpy().astype(
            np.uint8);
        import cv2;
        cv2.imshow("protos", protos)
        cv2.waitKey(0);

    def get_protos(this, sppids, normpids):
        normprotos = [this.norm_protos[i - this.sp_cnt] for i in normpids];
        spprotos = [this.sp_protos[i].unsqueeze(0) for i in sppids];
        normprotos = this.proto_engine(torch.cat(normprotos).repeat(1, 3, 1, 1).to(this.dev_ind.device));
        allproto = torch.cat(spprotos + [normprotos]);
        if (this.drop):
            allproto = this.drop(allproto);

        return allproto / torch.norm(allproto, dim=-1, keepdim=True);

    def get_plabel_and_dict(this, sappids, normpids):
        all_ids = sappids.union(normpids);
        new_id = 0;
        plabels = [];
        labmap = {};
        bidict = {}
        for i in all_ids:
            cha = this.aligned_characters[i];
            if (this.masters_share):
                vlab = this.masters[i];
            else:
                vlab = i;
            if (vlab not in labmap):
                labmap[vlab] = new_id;
                # A new label
                new_id += 1;
            alab = labmap[vlab];
            plabels.append(alab);
            bidict[alab] = cha;
            bidict[cha] = alab;
        plabels.append(new_id)
        bidict["[UNK]"] = new_id;
        bidict[new_id] = "⑨";

        return torch.tensor(plabels), bidict;

    def dump_all(this):
        trsps = set([this.label_dict[i] for i in this.sp_tokens]);
        trchs = set([this.label_dict[i] for i in this.shaped_characters]);

        protos = this.get_protos(trsps, trchs);
        plabels, tdicts = this.get_plabel_and_dict(trsps, trchs)
        return protos, plabels, tdicts;


class neko_prototype_core_structural(neko_prototype_core_basic):
    PROTOENGINE = neko_structural_visual_only_interprinter;

class neko_prototype_core_structuralR34(neko_prototype_core_basic):
    PROTOENGINE = neko_visual_only_interprinterR34

class neko_prototype_core_weird(neko_prototype_core_basic):
    PROTOENGINE=neko_weird_visual_only_interprinter;


    def get_protos(this, sppids, normpids):
        normprotos = [this.norm_protos[i - this.sp_cnt] for i in normpids];
        spprotos = [this.sp_protos[i].unsqueeze(0) for i in sppids];
        normprotos = this.proto_engine(torch.cat(normprotos).to(this.dev_ind.device));
        allproto = torch.cat(spprotos + [normprotos]);
        if (this.drop):
            allproto = this.drop(allproto);

        return allproto / torch.norm(allproto, dim=-1, keepdim=True);

