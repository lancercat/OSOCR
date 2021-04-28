# coding:utf-8
from __future__ import print_function

from cfgs_scene import scene_cfg_te;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
# Cmon'man, It's 2020 and we still need lexicon?
import cv2;
import torch;
import numpy as np
import pylcs;

def keepratio_resize( img,h,w):
    cur_ratio = img.shape[1] / float(img.shape[0])
    target_ratio=w/ float(h)
    mask_height = h
    mask_width = w
    img = np.array(img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if cur_ratio > target_ratio:
        cur_target_height = h
        # print("if", cur_ratio, self.target_ratio)
        cur_target_width = w
    else:
        cur_target_height = h
        # print("else",cur_ratio,self.target_ratio)
        cur_target_width = int(h * cur_ratio)
    img = cv2.resize(img, (cur_target_width, cur_target_height))
    start_x = int((mask_height - img.shape[0]) / 2)
    start_y = int((mask_width - img.shape[1]) / 2)
    mask = np.zeros([mask_height, mask_width]).astype(np.uint8)
    mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
    img = mask
    return img


def img_test(img,runner,args):
    img=cv2.imread(img)
    imgr=keepratio_resize(img,32,128)/255.
    res=runner.test_im(torch.tensor(imgr).float().unsqueeze(0).unsqueeze(0).cuda(),args);
    return  res[0];
def dicted_test(img,runner,args,lex):
    res=img_test(img,
                runner,
                args);
    mind=9999;
    p=res;
    lns=[]
    with open(lex,"r") as fp:
        lns=[i.strip() for i in fp];
    lex=lns[0].split(",");
    iss=[];
    for i in lex:
        if(len(i)==0):
            continue;
        e=pylcs.edit_distance(res.lower(),i.lower())
        if(e<mind):
            mind=e;
            p=i
            iss.append(i)
    if(p==""):
        print("???")
    if(p.lower()!=res.lower()):
        print(res.lower(),"->",p.lower()," at ",img);

    return p
def dictfree_test(img,runner,args,lex):
    res=img_test(img,
                runner,
                args);
    mind=9999;
    p=res;
    lns=[]
    with open(lex,"r") as fp:
        lns=[i.strip() for i in fp];
    lex=lns[0].split(",");
    iss=[];
    for i in lex:
        if(len(i)==0):
            continue;
        e=pylcs.edit_distance(res.lower(),i.lower())
        if(e<mind):
            mind=e;
            p=i
            iss.append(i)
    if(p==""):
        print("???")
    return res

def rundictsvt():
    cfgs = scene_cfg_te()
    runner = HDOS2C(cfgs);
    args = runner.testready();
    err = 0;
    for i in range(647):

        with open("/run/media/lasercat/ssddata/svtdicted/%d.gt" % i, "r") as fp:
            gts = [k.strip() for k in fp];
        try:
            res = dicted_test("/run/media/lasercat/ssddata/svtdicted/%d.jpg" % i,
                              runner,
                              args, "/run/media/lasercat/ssddata/svtdicted/%d.lex" % i);
            if (res != gts[0]):
                err += 1
                print(gts, "vs", res, "at", i);
        except:
            err += 1
    print(1 - err / 647)
def rundictfreesvt():
    cfgs = scene_cfg_te()
    runner = HDOS2C(cfgs);
    args = runner.testready();
    err = 0;
    for i in range(647):

        with open("/run/media/lasercat/ssddata/svtdicted/%d.gt" % i, "r") as fp:
            gts = [k.strip() for k in fp];
        try:
            res = dictfree_test("/run/media/lasercat/ssddata/svtdicted/%d.jpg" % i,
                              runner,
                              args, "/run/media/lasercat/ssddata/svtdicted/%d.lex" % i);
            if (res.lower() != gts[0].lower()):
                err += 1
                print(gts, "vs", res, "at", i);
        except:
            err += 1
    print(1 - err / 647)
def runic03():
    cfgs = scene_cfg_te()
    runner = HDOS2C(cfgs);
    args = runner.testready();
    err = 0;
    for i in range(1107):

        with open("/home/lasercat/ssddata/ic03dicted/%d.gt" % i, "r") as fp:
            gts = [k.strip() for k in fp];
        try:
            res = dicted_test("/home/lasercat/ssddata/ic03dicted/%d.jpg" % i,
                              runner,
                              args, "/home/lasercat/ssddata/ic03dicted/%d.lex" % i);
            if (res.lower() != gts[0].lower()):
                err += 1
                print(gts, "vs", res, "at", i);
        except:
            err += 1
    print(1 - err / 1107)


if __name__ == '__main__':
    rundictsvt()