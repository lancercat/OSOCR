import glob
import os
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755;
from neko_sdk.ocr_modules.charset.etc_cset import latin62;

from neko_sdk.ocr_modules.charset.jap_cset import hira,kata,Kyoiku_kanji,Joyo_kanji;
import pylcs
def getres(file):
    with open(file,"r") as fp:
        [gt,pr]=[i.strip() for i in fp ];
        return gt, pr;
def accrfolder(root):
    files=glob.glob(os.path.join(root,"*res.txt"));
    tot =0;
    corr=0;
    tned=0;

    tlen=0
    for f in files:
        gt,pr=getres(f);
        tlen+=len(gt);

        if(len(set(gt).intersection(hira.union(kata)))==0):
            continue;
        # if(len(set(gt).intersection(t1_3755.union(latin62)))==len(set(gt))):
        #     continue;
        tned+=1-pylcs.edit_distance(gt,pr)/len(gt)
        if (gt==pr):
            corr+=1;
        tot+=1;
    print(corr/tot,tned/tot, corr,tot,tlen /tot)

accrfolder("/run/media/lasercat/ssddata/chs-jap-kt/");
