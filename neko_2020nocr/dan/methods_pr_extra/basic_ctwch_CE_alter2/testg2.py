# coding:utf-8
from __future__ import print_function

import sys

from cfgs_scene20x import scene_cfg_te;

from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;

#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
import os;

if __name__ == '__main__':
    cfgs=scene_cfg_te(root_override="/run/media/lasercat/data/chdump/")


    # When you test for visualization.
    cfgs.global_cfgs["test_miter"]=100000000; # 100*160 images. Or it will get really long
    # tmpfolder="/run/media/lasercat/disks/ctwchres/basic_ce"
    # root=os.path.join(tmpfolder,sys.argv[1])
    # os.makedirs(root)

    runner=HDOS2C(cfgs);
    runner.runtest(miter=100000,protoidx=0);
