# coding:utf-8
from __future__ import print_function

import sys



# from cfgs_scene5 import scene_cfg_te as scene_cfg_te_5;
# from cfgs_scene10 import scene_cfg_te as scene_cfg_te_10;
# from cfgs_scene15 import scene_cfg_te as scene_cfg_te_15;
from cfgs_scene20_g2 import scene_cfg_te as scene_cfg_te_20;
DICT={
    "2000": scene_cfg_te_20,
}
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    for k in DICT:
        cfgs = DICT[k]("/run/media/lasercat/data/chdump/")
        runner = HDOS2C(cfgs);
        runner.runtest(miter=1000000000,measure_rej=False,protoidx=1);
        print(k,"Done")
