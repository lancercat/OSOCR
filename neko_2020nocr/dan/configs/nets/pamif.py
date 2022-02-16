from neko_2020nocr.dan.dan_modules_pami.neko_base_f import neko_base_Feature_ExtractorF,neko_base_Feature_ExtractorF_thicc;
from neko_2020nocr.dan.DAN import CAM;

from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fef import neko_cco_Feature_ExtractorF,neko_cco_Feature_ExtractorF_thicc

from neko_2020nocr.dan.dan_modules.neko_SDPE2 import neko_SDPE2;
from neko_2020nocr.dan.dan_modules.neko_os_DTD_mk4_cclcf import neko_os_DTD_mk4_cclcf;
from neko_2020nocr.dan.dan_modules.dan_DTD_LCF import DTDLCF;
from neko_2020nocr.dan.configs.pipelines_pami import get_dos,get_dos_cco,get_baseline_cco;



def get_dos_base_f(ptfile,prefix,token,maxT=25,root_override=None,val_frac_override=0.8):
    return get_dos_cco(neko_base_Feature_ExtractorF,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT,root_override=root_override,val_frac_override=val_frac_override);
def get_dos_base_f_thicc(ptfile,prefix,token,maxT=25,root_override=None,val_frac_override=0.8):
    return get_dos_cco(neko_base_Feature_ExtractorF_thicc,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT,root_override=root_override,val_frac_override=val_frac_override);

def get_dos_basic_ccof_thicc(ptfile,prefix,token,maxT=25,root_override=None,val_frac_override=0.8):
    return get_dos_cco(neko_cco_Feature_ExtractorF_thicc,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT,root_override=root_override,val_frac_override=val_frac_override);
# Literally do nothing.
def get_dos_basic_ccof(ptfile,prefix,token,maxT=25,root_override=None,val_frac_override=0.8):
    return get_dos_cco(neko_cco_Feature_ExtractorF,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT,root_override=root_override,val_frac_override=val_frac_override);
# Literally do nothing.
def get_dos_basic_ccofb(ptfile,prefix,token,maxT=25,root_override=None,val_frac_override=0.8):
    return get_dos_cco(neko_cco_Feature_ExtractorF,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,9999,maxT,root_override=root_override,val_frac_override=val_frac_override);

def get_dos_basic_ccofx(ptfile,prefix,token,maxT=25,root_override=None,val_frac_override=0.8):
    return get_dos_cco(neko_cco_Feature_ExtractorF,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.01,maxT,root_override=root_override,val_frac_override=val_frac_override);
