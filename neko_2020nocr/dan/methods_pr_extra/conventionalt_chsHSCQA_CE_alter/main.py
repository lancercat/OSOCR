# coding:utf-8
from __future__ import print_function

from cfgs_scene import scene_cfg;
from neko_2020nocr.dan.danframework.baseline import baseline;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    cfgs=scene_cfg(root_override="/run/media/lasercat/ssddata/pr_review_extra")
    runner=baseline(cfgs);
    runner.run();

