# coding:utf-8
import os.path

from neko_2020nocr.dan.dataloaders.dataset_scene import *
from neko_2020nocr.dan.configs.global_cfg import get_test_cfg,get_save_cfgs,get_train_cfg;
from neko_2020nocr.dan.configs.optimizers import get_dos_optim

from neko_sdk.root import find_data_root;
from neko_2020nocr.dan.configs.nets.pami import get_baseline_lcf_cco as get_net;
from neko_2020nocr.dan.methods_pami.CHSHS_cfg import *
from neko_2020nocr.dan.methods_pami.loss_cfg import cls_emb as loss;
import os;
prefix="conventional"+"_"+DSPRFIX+"_"+loss[0]+"_alter";
print(prefix);

DSROOT=find_data_root();
assert(os.path.basename(os.getcwd())==prefix);

DICT=os.path.join(find_data_root(),"dicts/dic_3791.txt");
CC=3791+2;

class scene_cfg:
    def __init__(this,root_override=None):
        this.global_cfgs = get_train_cfg();
        this.dataset_cfgs = DSCFG(T, DICT, DSROOT);
        this.net_cfgs = get_net(CC, prefix, None, maxT=T,root_override=root_override);
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


class scene_cfg_tejp(scene_cfg):
    def __init__(this,root_override=None):
        this.global_cfgs = get_test_cfg();
        this.dataset_cfgs =   get_jap_test(T,DICT,DSROOT);
        this.net_cfgs = get_net(CC,prefix,"E4",maxT=T, root_override=root_override);
        this.optimizer_cfgs = get_dos_optim()
        this.saving_cfgs = get_save_cfgs(prefix)
        this.loss_weight = loss[1];

