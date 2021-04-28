import torch;
from torch import nn;
import random;
from neko_sdk.ocr_modules.neko_interprinter import neko_visual_only_interprinter,neko_visual_only_interprinterHD,neko_visual_only_interprinter_inv,neko_visual_only_interprinterR34;
from neko_sdk.ocr_modules.prototypers.neko_visual_center_prototyper import neko_abstract_visual_center_prototyper;
import numpy as np;
from collections import deque as python_queue;
# this class defines how samples are sampled ^_^
from neko_sdk.ocr_modules.prototypers.neko_prototyper_core import neko_prototype_core_basic
import torch_scatter;

import regex
class neko_nonsematical_prototype_core_basic(neko_prototype_core_basic):
    # every thing does not involve sampling
    PROTOENGINE=neko_visual_only_interprinter;

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
        # characters without shape is generally what you always want to keep.
        this.shaped_characters = sorted(set(meta["chars"]))
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

    def get_plabel_and_dict_core(this, sappids, normpids,masters_share):
        all_ids = sappids + normpids;
        new_id = 0;
        plabels = [];
        labmap = {};
        bidict = {}
        sembs = []
        for i in all_ids:
            cha = this.aligned_characters[i];
            if (masters_share):
                vlab = this.masters[i];
            else:
                vlab = i;
            if (vlab not in labmap):
                labmap[vlab] = new_id;
                # A new label
                new_id += 1;
                # sembs.append(this.semantic_embedding[vlab]);
            alab = labmap[vlab];
            plabels.append(alab);
            bidict[alab] = cha;
            bidict[cha] = alab;
        plabels.append(new_id)
        bidict["[UNK]"] = new_id;
        # Well it has to be something --- at least ⑨ is alignment friendly and not likely to appear in the dataset
        bidict[new_id] = "⑨";
        return torch.tensor(plabels),None,bidict;

    def get_plabel_and_dict(this,sappids,normpids):
        return this.get_plabel_and_dict_core(sappids,normpids,this.masters_share);



    def sample_charset_by_text_both(this,text_batch):
        b="";
        for _ in text_batch: b+=_;

        plain_chars_in_data=list(set(regex.findall(r'\X', b, regex.U)));
        trsps,trchs=this.get_sampled_ids(plain_chars_in_data);
        trchs=list(trchs);
        trsps=list(trsps);
        protos=this.get_protos(trsps,trchs);
        plabels_cased,sembs_cased,tdicts_cased=this.get_plabel_and_dict_core(trsps,trchs,False)
        plabels_uncased, sembs_uncased, tdicts_uncased = this.get_plabel_and_dict_core(trsps, trchs,True);

        # this.debug(trchs,"meow");
        return protos,[sembs_uncased,sembs_cased],[plabels_uncased,plabels_cased],[tdicts_uncased,tdicts_cased];


    def sample_charset_by_text(this,text_batch):
        b="";
        for _ in text_batch: b+=_;

        plain_chars_in_data=list(set(regex.findall(r'\X', b, regex.U)));
        trsps,trchs=this.get_sampled_ids(plain_chars_in_data);
        trchs=list(trchs);
        trsps=list(trsps);
        protos=this.get_protos(trsps,trchs);
        plabels,sembs,tdicts=this.get_plabel_and_dict(trsps,trchs)
        # this.debug(trchs,"meow");
        return protos,sembs,plabels,tdicts;



    def dump_all(this):
        trsps=[this.label_dict[i] for i in this.sp_tokens];
        trchs=[this.label_dict[i] for i in this.shaped_characters];

        protos=this.get_protos(trsps,trchs);
        plabels,sembs,tdicts=this.get_plabel_and_dict(trsps,trchs)
        return protos,sembs,plabels,tdicts;

