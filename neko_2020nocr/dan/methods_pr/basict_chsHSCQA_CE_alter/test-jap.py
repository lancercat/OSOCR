from __future__ import print_function

from cfgs_scene import scene_cfg_tejp;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    root="/run/media/lasercat/20615BC32265B955/prfinal/chs-japxl"
    cfgs=scene_cfg_tejp( root_override="/run/media/lasercat/20615BC32265B955/prfinal/")
    runner=HDOS2C(cfgs);
    runner.run(dbgpath=root);




