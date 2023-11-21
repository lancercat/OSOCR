
from neko_2020nocr.tasks.taskmaker import build_training_dataset_by_label,build_testing_dataset_by_label,split_labelset;
from neko_sdk.lmdbcvt.hwdbcvt import make_hwdb
from neko_2020nocr.tasks.dscs import scanfolder_and_add_pt
from neko_2020nocr.tasks.hwdb_fslchr.split import t1shuflist
import os
import torch

def splithwdbtr(trfulldbdst,tefulldbdst,name,
                dstroot,train_cnt,evalcnt,valcnt):
    # os.makedirs(dstroot,exist_ok=True);
    train,eval,val=split_labelset(t1shuflist,train_cnt,evalcnt,valcnt);
    build_training_dataset_by_label([trfulldbdst,tefulldbdst], dstroot,train, ["image"], ["label", "lang", "wrid"], "label",name);
    # torch.save([train,val,eval],os.path.join(dstroot,"split.pt"));
def splithwdb_test(trfulldbdst, tefulldbdst, compdbdst,name,
                   dstroot, train_cnt, evalcnt, valcnt):
    dstroot=os.path.join(dstroot,name);
    train,eval,val=split_labelset(t1shuflist,train_cnt,evalcnt,valcnt);

    build_testing_dataset_by_label([tefulldbdst], dstroot, val, eval, ["image"], ["label", "lang", "wrid"], "label","cuws_eval"+name,"cuws_val"+name);
    build_testing_dataset_by_label([compdbdst], dstroot, val, eval, ["image"], ["label", "lang", "wrid"], "label","cuwu_eval"+name,"cuwu_val"+name);


    # build_testing_dataset_by_label([compdbdst], dstroot, val, eval, ["image"], ["label", "lang", "wrid"], "label","cuwu_eval","cuwu_val");
    # build_training_dataset_by_label([compdbdst], dstroot, train, ["image"], ["label", "lang", "wrid"], "label", "cswu_eval");


def buildhwdb(trfulldb,tefulldb,compdb,droot):
    splithwdbtr(trfulldb,tefulldb,"hwdbfsl_train_20",droot,2000,1000,100);
    splithwdbtr(trfulldb,tefulldb,"hwdbfsl_train_15",droot,
                1500,1000,100);
    splithwdbtr(trfulldb,tefulldb,"hwdbfsl_train_10",droot,
                1000,1000,100);
    splithwdbtr(trfulldb,tefulldb,"hwdbfsl_train_5",droot,
                500,1000,100);

    splithwdb_test(trfulldb, tefulldb, compdb, "hwdbfsl_10_1",
                   droot, 2000, 1000, 100);




# it seems to be a tradition using 1.0-1.2 testing set for training.
# and as you can see, the validation set and evaluation set are always the same
# so we manually removed redundant
if __name__ == '__main__':
    ROOT="/media/lasercat/backup/"
    trgnt=ROOT+"deploy/hwdb/train/"
    tegnt=ROOT+"deploy/hwdb/test/"
    i13gnt=ROOT+"deploy/hwdb/comp/"

    trfulldbdst = ROOT+"deployedlmdbs/HWDB/hwdbtr";
    tefulldbdst = ROOT+"deployedlmdbs/HWDB/hwdbte";
    i13fulldbdst= ROOT+"deployedlmdbs/HWDB/hwdbco"
    fsltsks=ROOT+"deployedlmdbs/HWDB/pami_ch_fsl_hwdb";
    fnts=[ROOT+"deploy/NotoSansCJK-Regular.ttc"]
    make_hwdb(trgnt, trfulldbdst);
    make_hwdb(tegnt, tefulldbdst);
    make_hwdb(i13gnt, i13fulldbdst);
    buildhwdb(trfulldbdst,tefulldbdst,i13fulldbdst,fsltsks);
    scanfolder_and_add_pt(fsltsks,fnts,set(),set())