# coding:utf-8
from neko_2020nocr.dan.dataloaders.dataset_scene import *
from neko_2020nocr.dan.configs.global_cfg import get_test_cfg,get_save_cfgs,get_train_cfg;
from neko_2020nocr.dan.configs.optimizers import get_dos_optim

from neko_sdk.root import find_data_root;
from neko_2020nocr.dan.configs.nets.pamithicc import get_dos_basic_ccorgb_thicc as get_net;
from neko_2020nocr.dan.methods_pami.CHSHSCQA_cfg import *
from neko_2020nocr.dan.methods_pami.loss_cfg import cls_emb as loss;
prefix="basict"+"_"+DSPRFIX+"_"+loss[0]+"_alter";
print(prefix);
DSROOT=find_data_root();
assert(os.path.basename(os.getcwd())==prefix);

class scene_cfg:
    global_cfgs = get_train_cfg();
    dataset_cfgs = DSCFG(T,None,DSROOT);
    net_cfgs = get_net(pdict_trchs(DSROOT),prefix,None,maxT=T);
    optimizer_cfgs =get_dos_optim()
    saving_cfgs = get_save_cfgs(prefix)
    loss_weight=loss[1];
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
    global_cfgs = get_test_cfg();
    net_cfgs = get_net(pdict_evaljap(DSROOT),prefix,"E4",maxT=T);
    dataset_cfgs = get_test_jap_rgb(T,DICT,DSROOT);

