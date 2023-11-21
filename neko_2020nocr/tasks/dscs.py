from neko_sdk.lmdb_wrappers.ocr_lmdb_reader import neko_ocr_lmdb_mgmt;
from neko_2020nocr.tools.renderlite.metafy import render_lite;
from neko_2020nocr.tools.addfffh import refactor_meta,add_masters,finalize

import torch;
import regex
def get_ds(root):
    charset = {};
    db=neko_ocr_lmdb_mgmt(root,False,1000);
    for i in range(len(db)):
        _,t=db.getitem_encoded_im(i);
        for c in regex.findall(r'\X', t, regex.U):
            if(c not in charset):
                charset[c]=0
            charset[c]+=1;
        if(i%300==0):
            print(i, "of" , len(db),"ds",root)
    return charset;

#
# servants="QWERTYUIOPASDFGHJKLZXCVBNM";
# masters="qwertyuiopasdfghjklzxcvbnm";
# another example to make it fancier(multiple centers)
# servants=["qf1","wf2",'qf2','wf1'];
# masters="qwqw";

def makept(dataset,font,protodst,xdst,blacklist):
    chrset=list(set(xdst.union(get_ds(dataset))).difference(blacklist));
    engine = render_lite(os=84,fos=32);
    font_ids=[0 for c in chrset];
    meta=engine.render_core(chrset,['[s]'],font,font_ids,False);
    meta=refactor_meta(meta,unk=len(chrset)+len(['[s]']));
    # inject a shapeless UNK.
    servants="QWERTYUIOPASDFGHJKLZXCVBNM";
    masters="qwertyuiopasdfghjklzxcvbnm";
    meta["protos"].append(None)
    meta["achars"].append("[UNK]")
    add_masters(meta,servants,masters);
    # add_masters(meta,servants,masters);
    meta=finalize(meta);
    torch.save(meta,protodst);
    return chrset

from glob import glob
import os
def scanfolder_and_add_pt(root,font,xdst,blacklist):
    dslist=glob(os.path.join(root,"*"));
    for data in dslist:
        makept(data,font,os.path.join(data,"dict.pt"),xdst,blacklist);