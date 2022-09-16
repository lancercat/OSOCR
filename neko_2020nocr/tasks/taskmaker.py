from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper
from neko_sdk.lmdb_wrappers.ocr_lmdb_reader import neko_ocr_lmdb_mgmt
import os



def disjoint(setable_a,setable_b):
    return len(set(setable_a).intersection(set(setable_b)))==0;

def split_labelset(shuffled_labellist, traincnt, evalcnt, valcnt):
    if (valcnt == -1):
        valcnt=max(0, len(shuffled_labellist) - traincnt - evalcnt);
    if(len(shuffled_labellist)<traincnt+evalcnt):
        return None,None,None
    train= shuffled_labellist[:traincnt]
    evalval= shuffled_labellist[traincnt:][::-1]
    eval=evalval[:evalcnt];
    val=evalval[evalcnt:evalcnt+valcnt];
    if(not disjoint(train,eval)):
        return None,None,None;
    if (not disjoint(eval,val)):
        return None, None, None;
    if (not disjoint(val, train)):
        return None, None, None;
    return train,eval,val;

# for GFSL/GZSL, src for training and evaluation separates, so we need 2 fns.
def build_testing_dataset_by_label(srcdbtes,dstroot,val_list,
                                   eval_list,imkeys,annks,
                                   labkey="label",
                                   evalname="eval",valname="val"):
    evaldst = os.path.join(dstroot, evalname);
    valdst = os.path.join(dstroot, valname);
    evaldb = im_lmdb_wrapper(evaldst);
    if (len(val_list)):
        valdb= im_lmdb_wrapper(valdst);

    for srcdbte in srcdbtes:
        srctedb = neko_ocr_lmdb_mgmt(srcdbte, False, 1000);
        for i in range(len(srctedb)):
            im, t = srctedb.getitem_encoded_kv(i, imkeys, annks);
            td = srctedb.parse_to_dict(annks, t);
            rd = srctedb.parse_to_dict(imkeys, im);
            t=td[labkey]

            if t in val_list:
                valdb.adddata_kv({}, td, rd);
            elif t in eval_list:
                evaldb.adddata_kv({}, td, rd);
        return;

def build_training_dataset_by_label(srcdbtrs,dstroot,train_list,imkeys,annks,labkey="label",name="train"):

    traindst = os.path.join(dstroot, name);
    traindb = im_lmdb_wrapper(traindst);
    for srcdbtr in srcdbtrs:
        srctrdb=neko_ocr_lmdb_mgmt(srcdbtr,True,1000);
        for i in range(len(srctrdb)):
            im,t=srctrdb.getitem_encoded_kv(i,imkeys,annks);
            td=srctrdb.parse_to_dict(annks,t);
            rd=srctrdb.parse_to_dict(imkeys,im);
            t=td[labkey]
            if t in train_list:
                traindb.adddata_kv({},td,rd);

