from neko_2020nocr.dan.DAN import Feature_Extractor,CAM;
from neko_2020nocr.dan.dan_modules_pami.neko_tps_fe import neko_tps_Feature_Extractor,neko_tps_Feature_ExtractorF
from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe import neko_cco_Feature_Extractor

from neko_2020nocr.dan.dan_modules.neko_SDPE2 import neko_SDPE2,neko_SDPE3_rand;
from neko_2020nocr.dan.dan_modules.neko_os_DTD_mk4_cclcf import neko_os_DTD_mk4_cclcf,\
     neko_os_DTD_mk4_scosine_cclcf;
from neko_2020nocr.dan.dan_modules.dan_DTD_LCF import DTDLCF;
from neko_2020nocr.dan.configs.pipelines_pami import get_dos,get_dos_cco,get_baseline_cco,get_baseline_cco_rgb;
from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe_xl import neko_cco_Feature_Extractor_thicc;


def get_baseline_lcf_ccorgb_thicc(nclass,prefix,token,maxT=25,root_override=None):
    return get_baseline_cco_rgb(neko_cco_Feature_Extractor_thicc,CAM,DTDLCF,nclass,prefix,token,0.5,maxT,root_override=root_override);

def get_dos_basic_cco_scosine(ptfile,prefix,token,maxT=25,root_override=None,val_frac_override=0.8):
    return get_dos_cco(neko_cco_Feature_Extractor,CAM,neko_os_DTD_mk4_scosine_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT,root_override=root_override,val_frac_override=val_frac_override);

def get_dos_basic_g2rand_cco(ptfile,prefix,token,maxT=25,root_override=None,val_frac_override=0.8):
    return get_dos_cco(neko_cco_Feature_Extractor,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE3_rand,ptfile,prefix,token,0.5,maxT,root_override=root_override,val_frac_override=val_frac_override);
