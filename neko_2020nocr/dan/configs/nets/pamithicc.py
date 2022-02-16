from neko_2020nocr.dan.DAN import Feature_Extractor,CAM;
from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe_xl import neko_cco_Feature_Extractor_thicc,neko_Feature_Extractor_thicc;
from neko_2020nocr.dan.dan_modules.neko_SDPE2 import neko_SDPE2;
from neko_2020nocr.dan.dan_modules.neko_os_DTD_mk4_cclcf import neko_os_DTD_mk4_cclcf;
from neko_2020nocr.dan.configs.pipelines_pami import get_dos,get_dos_cco
from neko_2020nocr.dan.configs.pipelines_pamixl import get_dos_rgb,get_dos_cco_rgb
from neko_2020nocr.dan.dan_modules_pami.neko_base_f import neko_base_Feature_ExtractorF,neko_base_Feature_ExtractorF_thicc;

def get_dos_basic_cco_thicc(ptfile,prefix,token,maxT=25,root_override=None):
    return get_dos_cco(neko_cco_Feature_Extractor_thicc,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT,root_override=root_override);
def get_dos_basic_ccorgb_thicc(ptfile,prefix,token,maxT=25,root_override=None,val_frac_override=0.8):
    return get_dos_cco_rgb(neko_cco_Feature_Extractor_thicc,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT,root_override=root_override,val_frac_override=val_frac_override);
def get_dos_basic_rgb_thicc(ptfile,prefix,token,maxT=25,root_override=None):
    return get_dos_rgb(neko_Feature_Extractor_thicc,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,maxT,root_override=root_override);
def get_dos_base_f_rgb_thicc(ptfile,prefix,token,maxT=25,root_override=None,val_frac_override=0.8):
    return get_dos_cco_rgb(neko_base_Feature_ExtractorF_thicc,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT,root_override=root_override,val_frac_override=val_frac_override);
