# coding:utf-8
from __future__ import print_function

import sys



from cfgs_scenerej20x import scene_cfg_te

from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    for k in ["dictrej2502g.pt","dictrej2002g.pt","dictrej1002g.pt","dictrej502g.pt"]:
        cfgs = scene_cfg_te("/run/media/lasercat/data/chdump/",k)
        runner = HDOS2C(cfgs);
        runner.runtest(measure_rej=True,protoidx=0);
        print(k,"Done")
