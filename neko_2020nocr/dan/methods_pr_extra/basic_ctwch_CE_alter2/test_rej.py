# coding:utf-8
from __future__ import print_function

import sys

from cfgs_scenerej import scene_cfg_te;
id="20";

from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;

#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
import os;

if __name__ == '__main__':
    for k in ["dictrej250.pt","dictrej200.pt","dictrej100.pt","dictrej50.pt"]:
        cfgs = scene_cfg_te("/run/media/lasercat/data/chdump/",k)
        runner = HDOS2C(cfgs);
        runner.runtest(measure_rej=True);
        print(k,"Done")

