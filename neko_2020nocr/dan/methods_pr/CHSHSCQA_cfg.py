import os
from neko_2020nocr.dan.methods_pr.pami_openset_word_dataset import get_chs_wrctwHKSCQA,get_test_jap_rgb
T=30;
DICT=None;
DSCFG=get_chs_wrctwHKSCQA
def pdict_trchs(root):
    return os.path.join(root,"dicts","dab3791MC.pt");
def pdict_evaljap(root):
    return os.path.join(root,"dicts","dabjpmlt.pt");
DSPRFIX="chsHSCQA"