from __future__ import print_function

from cfgs_scene import scene_cfg_tejp;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------

#####OSR
# if __name__ == '__main__':
#     cfgs=scene_cfg_tejp("/run/media/lasercat/ssddata/dicts/dabjpmltch_seen.pt",root_override="/run/media/lasercat/ssddata/pamidump/trained_models/")
#     runner=HDOS2C(cfgs);
#     runner.run("/run/media/lasercat/ssddata/pamiremake/basict_chjaprej/",True);
#####GOSR
if __name__ == '__main__':
    cfgs=scene_cfg_tejp("/run/media/lasercat/ssddata/dicts/dabjpmltch_nohirakata.pt",
                        root_override="/run/media/lasercat/ssddata/pamidump/ablchs_md_scene/")
    runner=HDOS2C(cfgs);
    runner.run("/run/media/lasercat/ssddata/pamiremake/basict_chjaprej/",measure_rej=True);



