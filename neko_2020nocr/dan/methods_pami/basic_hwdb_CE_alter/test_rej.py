# coding:utf-8
from __future__ import print_function

import sys
try:
    if sys.argv[1]=="5":
        from cfgs_scene5 import scene_cfg_te;
    if sys.argv[1]=="10":
        from cfgs_scene10 import scene_cfg_te;
    if sys.argv[1]=="15":
        from cfgs_scene15 import scene_cfg_te;
    if sys.argv[1]=="20":
        from cfgs_scene20 import scene_cfg_te;
    id=sys.argv[1]
except:
    id="20"
    from cfgs_scenerej import scene_cfg_te;

from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    import os;
    cfgs=scene_cfg_te()
    runner=HDOS2C(cfgs);
    root=os.path.join("/run/media/lasercat/ssddata/hwdbcedump/","2000")
    os.makedirs(root,exist_ok=True);
    runner.run(root);

