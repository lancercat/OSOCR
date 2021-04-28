from neko_2020nocr.dan.DAN import Feature_Extractor,CAM;
from neko_2020nocr.dan.dan_modules_pami.neko_transformer_cam import neko_transformer_CAM
from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe import neko_cco_Feature_Extractor

from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe_xl import \
    neko_cco_se_Feature_Extractor,neko_cco_Feature_Extractor_thicc,neko_cco_Feature_Extractor_Thicc;
from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe2 import \
    neko_cco_Feature_Extractor2,neko_cco_Feature_Extractor2_thicc;
from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe2f import \
    neko_cco_Feature_Extractor2F_thicc;

from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe_masked import \
    neko_cco_Feature_Extractor_masked
from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe_nonorm import \
    neko_cco_Feature_Extractor_thicc_no_norm

from neko_2020nocr.dan.dan_modules.neko_SDPE2 import neko_SDPE2;
from neko_2020nocr.dan.dan_modules.neko_SDPE2K import neko_SDPE2K,neko_SDPE2I,neko_SDPE2S;

from neko_2020nocr.dan.dan_modules.neko_os_DTD_mk4_cclcf import neko_os_DTD_mk4_cclcf
from neko_2020nocr.dan.dan_modules.neko_os_DTD_mk4_cclcf_ms import neko_os_DTD_mk4_cclcf_ms

from neko_2020nocr.dan.dan_modules.neko_os_DTD_mk4_cc import neko_os_DTD_mk4_cclcf_semi
# from neko_2020nocr.dan.dan_modules.dan_CAM_ms import CAM_ms;

from neko_2020nocr.dan.configs.pipelines_pamiother import get_dos_cco_rgb_hcam,get_dos_cco_rgb_large,get_dos_cco_rgb_multiscale,get_dos_cco_rgb_xl;
from neko_2020nocr.dan.configs.pipelines_pamixl import get_dos_cco_rgb;
from neko_2020nocr.dan.configs.pipelines_pami import get_dos_cco;

def get_dos_basic_ccos(ptfile,prefix,token,maxT=25):
    return get_dos_cco(neko_cco_Feature_Extractor,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.2,maxT);
def get_dos_basic_cco2(ptfile,prefix,token,maxT=25):
    return get_dos_cco(neko_cco_Feature_Extractor2,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);
def get_dos_basic_cco2rgb(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb(neko_cco_Feature_Extractor2,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);

def get_dos_basic_cco_se(ptfile,prefix,token,maxT=25):
    return get_dos_cco(neko_cco_se_Feature_Extractor,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);
def get_dos_basic_ccorgbsemi(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb(neko_cco_Feature_Extractor,CAM,neko_os_DTD_mk4_cclcf_semi,neko_SDPE2S,ptfile,prefix,token,0.5,maxT);
def get_dos_basic_ccosemi(ptfile,prefix,token,maxT=25):
    return get_dos_cco(neko_cco_Feature_Extractor,CAM,neko_os_DTD_mk4_cclcf_semi,neko_SDPE2S,ptfile,prefix,token,0.5,maxT);

def get_dos_basic_ccorgb_masked(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb(neko_cco_Feature_Extractor_masked,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);
def get_dos_basic_ccorgb_thicc_hcam(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb_hcam(neko_cco_Feature_Extractor_thicc,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);
def get_dos_nsp_cco(ptfile,prefix,token,maxT=25):
    return get_dos_cco(neko_cco_Feature_Extractor,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2K,ptfile,prefix,token,0.5,maxT);
def get_dos_inv_cco(ptfile,prefix,token,maxT=25):
    return get_dos_cco(neko_cco_Feature_Extractor,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2I,ptfile,prefix,token,0.5,maxT);
def get_dos_basic_ccorgb_thicc_large(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb_large(neko_cco_Feature_Extractor_thicc,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);
def get_dos_basic_ccorgb_Thicc(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb(neko_cco_Feature_Extractor_Thicc,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);
def get_dos_basic_ccorgb_Thicc_tcam(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb(neko_cco_Feature_Extractor_Thicc,neko_transformer_CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);
def get_dos_basic_ccorgb_fpnthicc(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb_xl(neko_cco_Feature_Extractor2F_thicc,neko_transformer_CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);

def get_dos_basic_ccorgb_thicc_nonorm(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb_large(neko_cco_Feature_Extractor_thicc_no_norm,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);
def get_dos_basic_ccorgb_thicc_tcam(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb(neko_cco_Feature_Extractor_thicc,neko_transformer_CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);

def get_dos_basic_cco2rgb_thicc_large(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb_large(neko_cco_Feature_Extractor2_thicc,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);

# def get_dos_basic_ccorgb_thicc_multiscale(ptfile,prefix,token,maxT=25):
#     return get_dos_cco_rgb_multiscale(neko_cco_Feature_Extractor_thicc,CAM_ms,neko_os_DTD_mk4_cclcf_ms,neko_SDPE2,ptfile,prefix,token,0.5,maxT);

def get_dos_basic_ccorgb(ptfile,prefix,token,maxT=25):
    return get_dos_cco_rgb(neko_cco_Feature_Extractor,CAM,neko_os_DTD_mk4_cclcf,neko_SDPE2,ptfile,prefix,token,0.5,maxT);
