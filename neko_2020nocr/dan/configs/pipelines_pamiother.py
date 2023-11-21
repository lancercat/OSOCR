from neko_2020nocr.dan.configs.pipelines_pami import get_cco_fe_args,get_cam_args,get_xos_dtd_args,get_pe_args,armtoken_xos

def get_hcam_args(maxT):
    return  {
            'maxT': maxT,
            'depth': 8,
            'num_channels': 256,
        }

def _get_dos_cco_rgb_hcam(FE,CAM,DTD,PE,ptfile,hardness,maxT):
    return {
        'FE': FE,
        'FE_args': get_cco_fe_args(hardness,ich=3),
        'CAM': CAM,
        'CAM_args': get_hcam_args(maxT),
        'DTD': DTD,
        'DTD_args': get_xos_dtd_args(),
        'PE':PE,
        "PE_args":get_pe_args(ptfile)
    }

def get_dos_cco_rgb_hcam(FE,CAM,DTD,PE,ptfile,prefix,token,hardness,maxT=25):
    d=_get_dos_cco_rgb_hcam(FE,CAM,DTD,PE,ptfile,hardness,maxT);
    return armtoken_xos(d, token, prefix);
def get_xos_dtd_args_large():
    return {
            'nchannel': 1024,
            'dropout': 0.3,
        };

def get_pe_args_large(ptfile):
    return {
        "meta_path": ptfile,
        "nchannel": 1024,
        'case_sensitive': False,
    }
def _get_dos_cco_rgb_large(FE,CAM,DTD,PE,ptfile,hardness,maxT):
    return {
        'FE': FE,
        'FE_args': get_cco_fe_args(hardness,nchannel=1024,ich=3),
        'CAM': CAM,
        'CAM_args': get_cam_args(maxT),
        'DTD': DTD,
        'DTD_args': get_xos_dtd_args_large(),
        'PE':PE,
        "PE_args":get_pe_args_large(ptfile)
    }

def get_dos_cco_rgb_large(FE,CAM,DTD,PE,ptfile,prefix,token,hardness,maxT=25):
    d=_get_dos_cco_rgb_large(FE,CAM,DTD,PE,ptfile,hardness,maxT);
    return armtoken_xos(d, token, prefix);
def get_cco_fe_argsxl(hardness,nchannel=512,ich=1):
    return {
            'strides': [(1,1), (2,2), (1,1), (2,2), (1,1), (1,1)],
            'compress_layer' : False,
            'input_shape': [ich, 48, 192], # C x H x W
            "hardness": hardness,
            "oupch": nchannel,
        }
def _get_dos_cco_rgb_xl(FE,CAM,DTD,PE,ptfile,hardness,maxT):
    return {
        'FE': FE,
        'FE_args': get_cco_fe_argsxl(hardness,ich=3),
        'CAM': CAM,
        'CAM_args': get_cam_args(maxT),
        'DTD': DTD,
        'DTD_args': get_xos_dtd_args_large(),
        'PE':PE,
        "PE_args":get_pe_args_large(ptfile)
    }
def get_dos_cco_rgb_xl(FE,CAM,DTD,PE,ptfile,prefix,token,hardness,maxT=25):
    d=_get_dos_cco_rgb_xl(FE,CAM,DTD,PE,ptfile,hardness,maxT);
    return armtoken_xos(d, token, prefix);
def _get_dos_cco_rgb_multiscale(FE,CAM,DTD,PE,ptfile,hardness,maxT):
    return {
        'FE': FE,
        'FE_args': get_cco_fe_args(hardness,nchannel=512,ich=3),
        'CAM': CAM,
        'CAM_args': get_cam_args(maxT),
        'DTD': DTD,
        'DTD_args': get_xos_dtd_args(),
        'PE':PE,
        "PE_args":get_pe_args(ptfile)
    }
def get_dos_cco_rgb_multiscale(FE,CAM,DTD,PE,ptfile,prefix,token,hardness,maxT=25):
    d=_get_dos_cco_rgb_multiscale(FE,CAM,DTD,PE,ptfile,hardness,maxT);
    return armtoken_xos(d, token, prefix);