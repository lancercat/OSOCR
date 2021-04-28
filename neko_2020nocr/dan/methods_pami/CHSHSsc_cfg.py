import os
from neko_2020nocr.dan.methods_pami.pami_openset_word_dataset import get_chs_wrctwHKS,get_jap_test
T=30;
DICT=None;
DSCFG=get_chs_wrctwHKS
def pdict_trchs(root):
    return os.path.join(root,"dicts","dab3791SC.pt");
def pdict_evaljap(root):
    return os.path.join(root,"dicts","dabjpmlt.pt");
DSPRFIX="chsHSsc"