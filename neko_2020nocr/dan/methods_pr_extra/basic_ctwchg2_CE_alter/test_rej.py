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
        from cfgs_scenerej import scene_cfg_te;

    id=sys.argv[1];
except:
    from cfgs_scenerej import scene_cfg_te;
    id="20";

from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;

#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
import os;

if __name__ == '__main__':
    cfgs=scene_cfg_te()

    # when you test for performance
    root="/home/lasercat/ssddata/ctwchres/"+id;


    # When you test for visualization.
    # cfgs.global_cfgs["test_miter"]=1*160 #images. Or it will get really long
    # tmpfolder="/run/media/lasercat/disks/ctwchres/basic_ce"
    # root=os.path.join(tmpfolder,sys.argv[1])
    # os.makedirs(root)

    runner=HDOS2C(cfgs);
    runner.run(root);

