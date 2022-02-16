# coding:utf-8
from __future__ import print_function

from cfgs_scene import scene_cfg_te;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    # root="/run/media/lasercat/ssddata/mjstbad"
    root=None;
    cfgs=scene_cfg_te(root_override="/run/media/lasercat/ssddata/pamidump/fixabl/")

    # cfgs.dataset_cfgs["te_case_sensitive"]=True;
    # cfgs.dataset_cfgs["case_sensitive"] = True;
    # cfgs.net_cfgs["PE_args"]["case_sensitive"] = True;

    runner=HDOS2C(cfgs);

    runner.test_all(root);

