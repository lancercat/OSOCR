# coding:utf-8
from __future__ import print_function

from cfgs_scene import scene_cfg_te;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    cfgs=scene_cfg_te()
    runner=HDOS2C(cfgs);
    runner.test_all("/home/prir1005/cat/cute_tpt");

