# coding:utf-8
from __future__ import print_function
import sys
try:
    if sys.argv[1]=="5":
        from cfgs_scene5 import scene_cfg;
    elif sys.argv[1]=="10":
        from cfgs_scene10 import scene_cfg;
    elif sys.argv[1]=="15":
        from cfgs_scene15 import scene_cfg;
    elif sys.argv[1]=="20":
        from cfgs_scene20 import scene_cfg;
    else:
        from cfgs_scene10 import scene_cfg;
except:
    from cfgs_scene5 import scene_cfg;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    cfgs=scene_cfg()
    runner=HDOS2C(cfgs);
    runner.run();

