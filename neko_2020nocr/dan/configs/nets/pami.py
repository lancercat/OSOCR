from neko_2020nocr.dan.DAN import Feature_Extractor,CAM;
from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe import neko_cco_Feature_Extractor
from neko_2020nocr.dan.dan_modules.neko_SDPE2 import neko_SDPE2;
from neko_2020nocr.dan.dan_modules.neko_os_DTD_mk4_cclcf import neko_os_DTD_mk4_cclcf;
from neko_2020nocr.dan.dan_modules.dan_DTD_LCF import DTDLCF;
from neko_2020nocr.dan.configs.pipelines_pami import get_dos,get_dos_cco,get_baseline_cco;

def get_baseline_lcf_cco(nclass,prefix,token,maxT=25):
    return get_baseline_cco(neko_cco_Feature_Extractor,CAM,DTDLCF,nclass,prefix,token,0.5,maxT);
def get_dos_basic(ptfile,prefix,token,maxT=25):
    return get_dos(Feature_Extractor,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,maxT);
def get_dos_basic_cco(ptfile,prefix,token,maxT=25):
    return get_dos_cco(neko_cco_Feature_Extractor,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);
