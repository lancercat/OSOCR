from __future__ import print_function

from cfgs_scene import scene_cfg_tejp;
from neko_2020nocr.dan.danframework.baseline import baseline;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    cfgs=scene_cfg_tejp(root_override="/run/media/lasercat/data/chdump/")
    runner=baseline(cfgs);
    runner.run();

