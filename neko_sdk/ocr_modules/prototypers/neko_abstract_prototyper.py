
from torch import nn;
import torch;
from neko_sdk.ocr_modules.neko_interprinter import neko_visual_only_interprinter;

class neko_abstract_prototyper(nn.Module):

    def make_proto_engine(this,meta,backbone,preload_tensor=None):
        pass;

    def __init__(this,output_channel,meta,backbone=None,preload_tensor=None,max_batch_size=1024):
        super(neko_abstract_prototyper, this).__init__();
        this.output_channel = output_channel;
        this.max_batch_size=max_batch_size;
        # Generally, CUDA devices works well with dynamic batch size.
        # However, for Rocm devices it bites as it compiles kernel
        # everytime we change batch size it's nightmare.
        this.has_dumped=False;
        this.dumped_protos=None;
        this.dev_ind=torch.nn.Parameter(torch.rand([1]));
        list_character = list(meta["chars"]);
        this.character = list(meta["sp_tokens"]) + list_character;
        this.spcnt=len( list(meta["sp_tokens"]));

        this.label_dict = meta["label_dict"];

        unk=this.label_dict["[UNK]"];
        # if the sp_tokens does not provide an specific unk token, set it to -1;

        for i, char in enumerate(this.character):
            # print(i, char)
            this.label_dict[char] = i;

        # shapeless unk shall be excluded
        if(unk<0):
            this.label_set= set(this.label_dict.values()) - {unk};
        else:
            this.label_set = set(this.label_dict.values());
        this.need_compression=True;
        if(this.max_batch_size>len(this.label_set)):
            this.max_batch_size=len(this.label_set);
        else:
            this.need_compression=False;

        this.make_proto_engine(meta,backbone,preload_tensor=None);

    def encoded(this, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        pass
        # length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        # batch_max_length += 1
        # # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        # batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        # from dataset_adaptors.mltcvt.mltmeta_kai2 import chset;
        # chset.add('[s]');
        # for i, t in enumerate(text):
        #     text = list(t)
        #     text.append('[s]')
        #     nt=[];
        #     for c in text:
        #         if(c in chset):
        #             nt.append(this.label_dict[c])
        #         else:
        #             nt.append(0)
        #     text = nt;
        #     batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        # return (batch_text, torch.IntTensor(length))
    def encode(this, text, batch_max_length=25,label_dict=None):
        if(label_dict is None):
            label_dict=this.label_dict;
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            try:
                text = [label_dict[char] for char in text];
            except:
                # print(text);
                nt=[];
                for ch in text:
                    try:
                        nt.append(this.label_dict[ch]);
                    except:
                        # print("     unknown",ch);
                        nt.append(this.label_dict["[UNK]"]);
                text = nt;
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text, torch.IntTensor(length))

    def decode(this, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = '';
            for i in text_index[index, :]:
                if(this.character[i]=="[s]"):
                    break;
                elif(i<this.spcnt):
                    continue;


                text+=this.character[i];
            texts.append(text)
        return texts

    def label_proto_compress(this, label, labels_in_task, nidx, prototypes):
        compact_prototype, (norm_chars, norm_weights)=this.proto_compress(labels_in_task);
        compact_ids=this.label_compress(label,labels_in_task,nidx);
        # print(compact_prototype.norm(dim=1));
        return compact_ids,compact_prototype,(norm_chars,norm_weights);


    def label_compress(this, label, labels_in_task, nidx):
        return None,None;

    def encode_text_compressed(this,text,nidx,labels_in_task,batch_max_length):
        label,length=this.encode(text,batch_max_length);
        clabel=this.label_compress(label,labels_in_task,nidx);
        return clabel,length;
    def encode_text_uncompressed(this,text,nidx,labels_in_task,batch_max_length):
        label,length=this.encode(text,batch_max_length);
        return label,length;



    def label_decompress(this, compressed_ids, valid, index):
        return None;

    def get_proto(this,valid,nidx):
        return None;

    #uni loss takes batch
    def encode_compressed(this,text,batch_max_length):
        pass;

        # label,length=this.encode(text,batch_max_length);
        # valid,nidx=label.unique().sort();
        # clabel,cproto=this.label_compress(label,valid,nidx);
        # return label,length,valid,nidx,clabel,cproto,None;

    def decode_compressed(this,ids,valid,nidx,length):
        did=this.label_decompress(ids,valid,nidx);
        texts=this.decode(did,length);
        return texts;

    # uni loss takes batch
    def encode_uncompressed(this, text, batch_max_length):

        label, length = this.encode(text, batch_max_length);
        valid, nidx = None,None;
        clabel=label;
        cproto=this.dumped_protos;
        return label, length, valid, nidx, clabel, cproto,None;

    def decode_uncompressed(this, ids, valid, nidx, length):
        texts = this.decode(ids, length);
        return texts;

    def undump_protos(this):
        this.has_dumped=False;
        if(this.dumped_protos is not None):
            del this.dumped_protos;
            this.dumped_protos=None;
    def dump_protos(this,charset):
        pass;


# static prototypes, one per class. Equiv to FC layer.
class neko_baseline_prototyper(neko_abstract_prototyper):

    def make_proto_engine(this,meta,backbone,preload_tensor=None):
        this.prototype_cnt = len(meta["sp_tokens"] + meta["chars"]);
        tensor=torch.rand([this.prototype_cnt, this.output_channel])*2-1;

        # this.prototype_cnt = len(meta["sp_tokens"] + meta["chars"])+30000;
        # tensor = torch.rand([this.prototype_cnt , this.output_channel]);
        # this.character+=['@' for _ in range(30000)];


        # torch.nn.init.xavier_uniform_(tensor, gain=1.0);
        this.prototypes = torch.nn.Parameter(tensor);

        this.register_parameter("prototypes", this.prototypes);

    def dump_protos(this,eval_meta):
        #Baseline has no conv proto maker
        this.has_dumped = True;
        this.dumped_protos=this.prototypes;

    def encode_compressed(this,text,batch_max_length):
        label, length = this.encode(text, batch_max_length);
        clabel, cproto, cvproto =label,this.prototypes,None;
        #(this.prototypes.norm(dim=-1,keepdim=True)+0.000001),None;
        nidx=torch.tensor(range(this.prototypes.shape[0]));
        batch_proto_id=nidx;
        return label, length, batch_proto_id, None, clabel, cproto, cvproto;
        pass;
    def label_proto_compress(this, label, labels_in_task, nidx,prototypes):
        compact_ids=label;
        compact_prototype=this.prototypes;
        # compact_ids=ids.clone();
        # for i in range(len(valid)):
        #     compact_ids[compact_ids==valid[i]]=index[i];
        # compact_prototype=prototypes[valid];
        return compact_ids,compact_prototype,None;

    def label_decompress(this, compressed_ids, valid, index):
        original_id=compressed_ids;
        # original_id=ids.clone();
        # for i in range(len(valid)):
        #     original_id[original_id == index[i]] =valid[i];
        return original_id;


    def forward(this,text):
        return this.prototypes;

# produces prototypes from standard visual clue
class neko_visual_only_prototyper(neko_abstract_prototyper):

    def make_proto_engine(this,meta,backbone,preload_tensor=None):
        this.proto_engine=neko_visual_only_interprinter(this.output_channel);
        this.prototype_cnt = len(meta["sp_tokens"] + meta["chars"]);
        this.sp_cnt=len(meta["sp_tokens"]);
        this.sp_protos=torch.nn.Parameter(torch.rand([
            this.sp_cnt,this.output_channel]).float()*2-1);
        this.register_parameter("sp_proto",this.sp_protos);
        if(preload_tensor is None):
            this.norm_protos= meta["protos"][this.sp_cnt:];
        else:
            this.norm_protos=torch.load(preload_tensor);
        for i in range(len(this.norm_protos)):
            if this.norm_protos[i].max()>20:
                this.norm_protos[i]=(this.norm_protos[i] - 127.5) / 128;
    def pick(this,text):
        pass;

    def dump_protos(this,eval_meta):
        #Baseline has no conv proto maker
        this.has_dumped = True;
        with torch.no_grad():
            protos=[this.sp_protos];
            for i in range(0,len(this.norm_protos),50):
                protos.append(this.proto_engine(torch.cat(this.norm_protos[i:i+50],0).repeat([1,3,1,1]).to(this.dev_ind.device)).detach());
            this.dumped_protos=torch.cat(protos)


    def label_proto_compress(this, label, labels_in_task, nidx,prototypes):
        norm_valid= labels_in_task[labels_in_task >= this.sp_cnt] - this.sp_cnt;
        norm_chars=torch.stack([this.norm_protos[i][0] for i in norm_valid ],0).repeat([1,3,1,1]).contiguous();
        norm_weights=this.proto_engine(norm_chars.to(this.dev_ind.device));
        spproto=this.sp_protos/this.sp_protos.norm(dim=-1,keepdim=True);
        compact_prototype=torch.cat([spproto,norm_weights],dim=0);
        # print(compact_prototype.norm(dim=1));
        compact_ids=label.clone();
        for i in range(len(labels_in_task)):
            compact_ids[label == labels_in_task[i]]=nidx[i];
        return compact_ids,compact_prototype,(norm_chars,norm_weights);

    def label_decompress(this, compressed_ids, valid, index):
        original_id=valid[compressed_ids];
        # original_id=ids.clone();
        # for i in range(len(valid)):
        #     original_id[original_id == index[i]] =valid[i];
        return original_id;


    def forward(this,text):
        valid=this.pick(text);
        this.forward(valid);

# produces different forms from standard visual clue
# That means, protos are not friendless! you can get either proto within the group,
# for example, [a and A] can be counted as an effective hit of a
# this is for the sake where the label dose not distinguish different forms of a character.
