from neko_2020nocr.tasks.taskmaker import build_training_dataset_by_label,build_testing_dataset_by_label,split_labelset;
from neko_2020nocr.tasks.dscs import scanfolder_and_add_pt;
from neko_2020nocr.tasks.ctw_fslchr.split import ctwshuflist

from neko_sdk.lmdbcvt.ctwchcvt import make_ctwch


import os
import torch;


def splitctwfsl_tr(srcdb,dstroot,train_cnt,name,evalcnt=500,valcnt=100):
    train,eval,val=split_labelset(ctwshuflist,train_cnt,evalcnt,valcnt);
    build_training_dataset_by_label(srcdb, dstroot,train, ["image"], ["label", "lang", "attr"], "label",name=name);

def splitctwfsl_te(srcdb, dstroot, train_cnt, evalname,valname, evalcnt=500, valcnt=100):
    train,eval,val=split_labelset(ctwshuflist,train_cnt,evalcnt,valcnt);
    build_testing_dataset_by_label(srcdb, dstroot, val, eval, ["image"], ["label", "lang", "attr"], "label",evalname=evalname,valname=valname);

def buildctws(src,droot,fntpath):
    splitctwfsl_tr([src],droot,500,"ctwfsl5_train",500,100);
    splitctwfsl_tr([src],droot,1000,"ctwfsl10_train",500,100);
    splitctwfsl_tr([src],droot,1500,"ctwfsl15_train",500,100);
    splitctwfsl_tr([src],droot,2000,"ctwfsl20_train",500,100);
    splitctwfsl_te([src], droot, 2000, "ctwfsl_5_1eval","ctwfsl_5_1val", 500, 100);


if __name__ == '__main__':
    ROOT="/media/lasercat/backup"
    trgtpath = ROOT+"/deploy/ctw/gtar/train.jsonl";
    trjpgpath = ROOT+"/deploy/ctw/jpgs";
    chfulldbdst = ROOT+"/deploy/ctwch";
    fsltsks=ROOT+"/deployedlmdbs/ctwch";
    fntpath=ROOT+"/deploy/NotoSansCJK-Regular.ttc"
    # make_ctwch(trgtpath, trjpgpath,chfulldbdst);
    # buildctws(chfulldbdst, fsltsks,fntpath);
    scanfolder_and_add_pt(fsltsks,[fntpath],set(),set());
