from neko_2020nocr.dan.configs.pipelines_pami import _get_dos,_get_dos_cco;
from neko_2020nocr.dan.configs.pipelines_pami import armtoken_xos
from neko_2020nocr.dan.configs.pipelines_pami import get_cco_fe_args,get_bl_fe_args,get_cam_args,get_xos_dtd_args,get_pe_args

def _get_dos_cco_rgb(FE,CAM,DTD,PE,ptfile,hardness,maxT):
    return {
        'FE': FE,
        'FE_args': get_cco_fe_args(hardness,ich=3),
        'CAM': CAM,
        'CAM_args': get_cam_args(maxT),
        'DTD': DTD,
        'DTD_args': get_xos_dtd_args(),
        'PE':PE,
        "PE_args":get_pe_args(ptfile)
    }
def _get_dos_rgb(FE,CAM,DTD,PE,ptfile,maxT):
    return {
        'FE': FE,
        'FE_args': get_bl_fe_args(ich=3),
        'CAM': CAM,
        'CAM_args': get_cam_args(maxT),
        'DTD': DTD,
        'DTD_args': get_xos_dtd_args(),
        'PE':PE,
        "PE_args":get_pe_args(ptfile)
    }

def get_dos_rgb(FE,CAM,DTD,PE,ptfile,prefix,token,maxT=25):
    d=_get_dos_rgb(FE,CAM,DTD,PE,ptfile,maxT);
    return armtoken_xos(d, token, prefix);

def get_dos_cco_rgb(FE,CAM,DTD,PE,ptfile,prefix,token,hardness,maxT=25):
    d=_get_dos_cco_rgb(FE,CAM,DTD,PE,ptfile,hardness,maxT);
    return armtoken_xos(d, token, prefix);
