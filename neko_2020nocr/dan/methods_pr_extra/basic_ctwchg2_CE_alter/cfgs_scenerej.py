# coding:utf-8
from neko_2020nocr.dan.dataloaders.dataset_scene import *
from neko_2020nocr.dan.configs.global_cfg import get_test_cfg,get_save_cfgs,get_train_cfg;
from neko_2020nocr.dan.configs.optimizers import get_dos_optim

from neko_sdk.root import find_data_root;
from neko_2020nocr.dan.configs.nets.pr_extras import get_dos_basic_g2rand_cco as get_net;
from neko_2020nocr.dan.methods_review_extra.CTWS_cfg import *
from neko_2020nocr.dan.methods_pr.loss_cfg import cls_emb as loss;
from cfgs_scene20_g2 import scene_cfg
cnt=2000

prefix="basic"+"_"+DSPRFIX+"_"+loss[0]+"_"+"alter_"+str(cnt);
print(prefix);
DSROOT=find_data_root();
assert(os.path.basename(os.getcwd())+"_"+str(cnt)==prefix);


class scene_cfg_te(scene_cfg):
    def __init__(this,pathoveride,datasetname="dictrej500.pt"):
        dataset = DSCFG(cnt, T,datasetname , DSROOT);
        PDICT = dataset["dict_dir"];
        this.net_cfgs = get_net(PDICT, prefix, None, maxT=T);
        this.optimizer_cfgs = get_dos_optim()
        this.saving_cfgs = get_save_cfgs(prefix,pathoveride)
        this.dataset_cfgs = dataset;
        this.loss_weight=loss[1];
        this.global_cfgs = get_test_cfg();
        this.net_cfgs = get_net(PDICT,prefix,"E9",maxT=T,root_override=pathoveride);
