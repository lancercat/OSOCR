# coding:utf-8
from __future__ import print_function

import glob
import os.path

from cfgs_scene_open import scene_cfg_open_test;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
# Cmon'man, It's 2020 and we still need lexicon?
import cv2;
import torch;
import numpy as np
import pylcs;
from neko_sdk.ocr_modules.img_eval import keepratio_resize
from neko_2020nocr.result_renderer import render_word


def img_test(img,runner,args,rgb=True):
    img=cv2.imread(img)
    if(rgb):
        imgr=keepratio_resize(img,32,128,rgb)[0]/255.
        res=runner.test_im(torch.tensor(imgr).float().permute(2,0,1).unsqueeze(0).cuda(),args);

    else:
        imgr=keepratio_resize(img,32,128,rgb)/255.
        res=runner.test_im(torch.tensor(imgr).float().unsqueeze(0).unsqueeze(0).cuda(),args);
    return  res[0];



def run_folder(ptfile,sfolder,dfolder,save_root):
    cfgs = scene_cfg_open_test(ptfile,save_root)
    runner = HDOS2C(cfgs);
    args = runner.testready();
    mdict = torch.load(ptfile)

    files=glob.glob(os.path.join(sfolder,"*.jpg"));
    for i in range(len(files)):
        res = img_test(files[i],
                              runner,
                              args);
        base=os.path.basename(files[i]);
        dstt=os.path.join(dfolder,base.replace("jpg","txt"));
        dsti=os.path.join(dfolder,base);
        dim,_=render_word(mdict,{},cv2.imread(files[i]),None,res,0);
        cv2.imwrite(dsti,dim);
        print(res);
        with open(dstt,"w+") as fp:
            fp.writelines(res);



if __name__ == '__main__':
    run_folder("/run/media/lasercat/ssddata/dicts/dabrusnum.pt",
               "/run/media/lasercat/ssddata/SIW-13/Russian/","/run/media/lasercat/ssddata/pamidump/rusputin-zero/greek/","/run/media/lasercat/ssddata/pamidump/ablchs_md_scene/");

    # run_folder("/run/media/lasercat/ssddata/dicts/dabgreeknumaf.pt",
    #            "/run/media/lasercat/ssddata/SIW-13/Greek/","/run/media/lasercat/ssddata/pamidump/rusputin-zero/greek/","/run/media/lasercat/ssddata/pamidump/ablchs_md_scene/");
    #
