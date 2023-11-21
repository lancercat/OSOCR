# coding:utf-8
from __future__ import print_function

import sys



from cfgs_scenerej import scene_cfg_te

from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    for k in ["dictrej250.pt","dictrej200.pt","dictrej100.pt","dictrej50.pt"]:
        cfgs = scene_cfg_te("/home/lasercat/ssddata/pamidump/mdch/ctw/",k)
        runner = HDOS2C(cfgs);
        runner.run(measure_rej=True);
        print(k,"Done")
