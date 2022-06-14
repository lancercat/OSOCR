
import glob
import os
import shutil
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755;
from neko_sdk.ocr_modules.charset.etc_cset import latin62;
import editdistance as ed
from neko_2020nocr.dan.utils import Loss_counter,neko_os_Attention_AR_counter

from neko_sdk.ocr_modules.charset.jap_cset import hira,kata,Kyoiku_kanji,Joyo_kanji;
def with_hirakata(gt):
    return len(set(gt).intersection(hira.union(kata)))>0;
def wo_hirakata(gt):
    return len(set(gt).intersection(hira.union(kata))) ==0;
def seen(gt):
     return len(set(gt).intersection(t1_3755.union(latin62)))==len(set(gt))
def ukanji(gt):
    return wo_hirakata(gt) and not seen(gt);
def all_words(gt):
    return True;
filters={
    "Overall": all_words,
    "Seen":seen,
    "Unique Kanji": ukanji,
    "All Kanji": wo_hirakata,
    "Kana": with_hirakata
}
import pylcs
def getres(file):
    with open(file,"r") as fp:
        [gt,pr]=[i.strip() for i in fp ][:2];
        return gt, pr;
def accrfolder(root,filter,case_sensitive=False):
    files=glob.glob(os.path.join(root,"*.txt"));
    tot =0;
    corr=0;
    tned=0;
    arcntr=neko_os_Attention_AR_counter(root,case_sensitive=case_sensitive);
    tlen=0
    for f in files:
        gt,pr=getres(f);
        if(not case_sensitive):
            gt=gt.lower();
            pr=pr.lower();
        tlen+=len(gt);
        if(not filter(gt)):
            continue;
        arcntr.add_iter([pr],[gt],[gt])
        ned=1-pylcs.edit_distance(gt, pr) / len(gt)
        tned+=ned
        if (gt==pr):
            corr+=1;
        else:
            # print(gt,pr);
            pass;
        tot+=1;
    arcntr.show();
    return corr / tot, tned / tot, corr, tot, tlen / tot;
def maketex(root,name,ks):
    rd={};
    tex=name
    for k in ks:
        dst=os.path.join(root,k)
        try:
            shutil.rmtree(dst);
        except:
            pass;
        os.makedirs(os.path.join(root,k),exist_ok=True);

        if(k not in filters):
            tex+="&";
            continue;
        acr,ccr,corr,tot,alen=accrfolder(root,filters[k]);
        print(k, ccr,acr);
        tex+="&"+"{:.2f}".format(ccr*100)+"/"+"{:.2f}".format(acr*100);
    tex+="\\\\";
    print(tex);
def makejp(root):
    KS=    "Overall","Speed","Seen","Unique Kanji", "All Kanji","Kana";
    maketex(root,"Ours",KS);

if __name__ == '__main__':
    makejp("/run/media/lasercat/20615BC32265B955/prfinal/chs-japxl/");
