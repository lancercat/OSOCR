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
splitdict={
    "OSR":"/home/lasercat/ssddata/dicts/dabjpmltch_seen.pt",
    "GOSR":"/home/lasercat/ssddata/dicts/dabjpmltch_nohirakata.pt",
    "OSTR":"/home/lasercat/ssddata/dicts/dabjpmltch_kanji.pt",
}

if __name__ == '__main__':
    for k in splitdict:
        cfgs=scene_cfg_tejp(splitdict[k],
                            root_override="/run/media/lasercat/f3a1698e-80ad-4473-8fc6-4df8c81c3831/osocr-weight/prfinal/")
        runner=HDOS2C(cfgs);
        print(k);
        runner.run(None,measure_rej=True);





