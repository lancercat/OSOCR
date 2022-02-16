import os
def get_stdhwdbtr(train_cnt,root):
    name="hwdbfsl_train_"+str(train_cnt//100);
    return os.path.join(root, "pami_ch_fsl_hwdb", name),\
           os.path.join(root, "pami_ch_fsl_hwdb", name,"dict.pt");

def get_stdhwdbte(root,dict_name):
    return os.path.join(root, "pami_ch_fsl_hwdb", "cuwu_evalhwdbfsl_10_1"), \
           os.path.join(root, "pami_ch_fsl_hwdb", "cuwu_evalhwdbfsl_10_1",dict_name);
def get_full_orahwdb(root):
    return os.path.join(root, "oraclehw"), \
           os.path.join(root, "oraclehw","meta.pt");
def get_hnd_ch74kdb(root):
    return os.path.join(root, "char74k","char74k_hnd_good"),os.path.join(root,"char74k","char74k_hnd_good","dabchar74khnd.pt")

def get_stdctwchtr(train_cnt,root):
    name="ctwfsl"+str(train_cnt//100)+"_train"
    return os.path.join(root,"ctwch",name), \
           os.path.join(root, "ctwch", name,"dict.pt");
def get_stdctwchte(root,dict_name):
    return os.path.join(root,"ctwch","ctwfsl_5_1eval"), \
           os.path.join(root, "ctwch", "ctwfsl_5_1eval",dict_name);
def get_stdctwchterej(root):
    return os.path.join(root,"ctwch","ctwfsl_5_1eval"), \
           os.path.join(root, "ctwch", "ctwfsl_5_1eval","dictrej.pt");

def get_synchtr(root):
    return os.path.join(root,"ch3755synth");

def get_artK_path(root):
    return os.path.join(root,"artdb_seen");
def get_mlt_chlatK_path(root):
    return os.path.join(root,"mlttrchlat_seen");
def get_mlt_krK_path(root):
    return os.path.join(root,"mltkrdb_seen");
def get_lsvtK_path(root):
    return os.path.join(root,"lsvtdb_seen");
def get_ctwK_path(root):
    return os.path.join(root,"ctwdb_seen");
def get_rctwK_path(root):
    return os.path.join(root,"rctwtrdb_seen");

def get_mltjp_path(root):
    return os.path.join(root,"mlttrjp_hori");
def get_mltkr_path(root):
    return os.path.join(root,"mlttrkr_hori");

def get_mltbe_path(root):
    return os.path.join(root,"mlttrbe_hori");

def get_mlthn_path(root):
    return os.path.join(root,"mlttrhn_hori");


def get_monkey_path(root,lang):
    return os.path.join(root,"monkey",lang);
