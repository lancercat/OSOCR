from __future__ import print_function

from cfgs_scene import scene_cfg_tejp;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    root="/run/media/lasercat/75f830a6-9ccb-48bd-b779-2662b84fb036/home/lasercat/results/"
    cfgs=scene_cfg_tejp( root_override="/run/media/lasercat/f3a1698e-80ad-4473-8fc6-4df8c81c3831/osocr-weight/prfinal/")
    runner=HDOS2C(cfgs);
    runner.run(dbgpath=root);




