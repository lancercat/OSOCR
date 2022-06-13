# coding:utf-8
from __future__ import print_function
import sys

from cfgs_scene20 import scene_cfg;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C_cossim2f;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    cfgs=scene_cfg(root_override="/run/media/lasercat/ssddata/pr_review_extra/")
    runner=HDOS2C_cossim2f(cfgs);
    runner.run();

