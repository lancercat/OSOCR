# coding:utf-8
from __future__ import print_function

from cfgs_scene import scene_cfg;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    cfgs=scene_cfg()
    runner=HDOS2C(cfgs);
    runner.run();

