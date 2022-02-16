# coding:utf-8
from neko_2020nocr.dan.configs.global_cfg import get_test_cfg,get_save_cfgs;
from neko_2020nocr.dan.configs.optimizers import get_dos_optim

from neko_sdk.root import find_data_root;
from neko_2020nocr.dan.configs.nets.pamithicc import get_dos_basic_ccorgb_thicc as get_net;
from neko_2020nocr.dan.methods_pami.CHSHSCQA_cfg import *
from neko_2020nocr.dan.methods_pami.loss_cfg import cls_emb as loss;
prefix="basict"+"_"+DSPRFIX+"_"+loss[0]+"_alter";

class scene_cfg_open_test:
    def __init__(this,pdict,save_root_override=None):
        this.global_cfgs = get_test_cfg();
        this.net_cfgs = get_net(pdict, prefix, "E4", maxT=T,root_override=save_root_override);
        this.optimizer_cfgs =get_dos_optim()
        this.saving_cfgs = get_save_cfgs(prefix,save_root_override)
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

