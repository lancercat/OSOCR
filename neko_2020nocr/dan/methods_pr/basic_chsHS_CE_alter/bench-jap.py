from __future__ import print_function


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------

from cfgs_scene import scene_cfg_tejp;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    root="/run/media/lasercat/ssddata/chs-jap/"
    root=None
    cfgs=scene_cfg_tejp()
    runner=HDOS2C(cfgs);
    runner.run();

