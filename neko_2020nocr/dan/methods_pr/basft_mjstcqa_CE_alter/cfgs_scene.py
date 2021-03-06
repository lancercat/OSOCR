# coding:utf-8
from neko_2020nocr.dan.dataloaders.dataset_scene import *
from neko_2020nocr.dan.configs.global_cfg import get_test_cfg,get_save_cfgs,get_train_cfg;
from neko_2020nocr.dan.configs.optimizers import get_dos_optim

from neko_sdk.root import find_data_root;
from neko_2020nocr.dan.configs.nets.pamithicc import get_dos_base_f_rgb_thicc as get_net;
from neko_2020nocr.dan.stdbenchcfgs.MJSTCQA_cfg import *
from neko_2020nocr.dan.methods_pr.loss_cfg import cls_emb as loss;
prefix="basft"+"_"+DSPRFIX+"_"+loss[0]+"_alter";

DSROOT=find_data_root();
assert(os.path.basename(os.getcwd())==prefix);

class scene_cfg:
    def __init__(this,root_override=None):
        this.global_cfgs = get_train_cfg();
        this.dataset_cfgs = DSCFG(T,DSROOT);
        this.net_cfgs = get_net(PDICT,prefix,None,maxT=T,root_override=root_override);
        this.optimizer_cfgs =get_dos_optim()
        this.saving_cfgs = get_save_cfgs(prefix)
        this.loss_weight=loss[1];
    def mkdir(this,path_):
        paths = path_.split('/')
        command_str = 'mkdir '
        for i in range(0, len(paths) - 1):
            command_str = command_str + paths[i] + '/'
        command_str = command_str[0:-1]
        os.system(command_str)

    def showcfgs(this,s):
        for key in s.keys():
            print(key , s[key])
        print('')


class scene_cfg_te(scene_cfg):
    def __init__(this,root_override=None):
        this.global_cfgs = get_test_cfg();
        this.dataset_cfgs =  get_test_all_uncased_dsrgb(root=DSROOT);
        this.net_cfgs = get_net(PDICT,prefix,"E4",maxT=T, root_override=root_override);
        this.optimizer_cfgs = get_dos_optim()
        this.saving_cfgs = get_save_cfgs(prefix)
        this.loss_weight = loss[1];
