# coding:utf-8
from __future__ import print_function

import sys



from cfgs_scenerej import scene_cfg_te

from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    fnts=["serif","sans"];
    for protoidx in [0,1]:
        print("proto fnt",fnts[protoidx]);
        for k in ["dictrej2502g.pt","dictrej2002g.pt","dictrej1002g.pt","dictrej502g.pt"]:
            cfgs = scene_cfg_te("/run/media/lasercat/data/chdump/", k)
            runner = HDOS2C(cfgs);
            runner.runtest(miter=1000000000,measure_rej=True,protoidx=protoidx);
            print(k,"Done")
