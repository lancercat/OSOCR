from neko_2020nocr.dan.dataloaders.dataset_online_synth import nekoOLSDataset,nekoOLSCDataset
from neko_2020nocr.dan.dataloaders.dataset_scene import *
from neko_sdk.ocr_modules import mtsaug
from torchvision import transforms
from neko_2020nocr.dan.configs.datasets.ds_paths import *



def get_dataset_test(maxT,root,dict_dir):
    return {
        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [root],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dataloader_test': {
            'batch_size': 128,
            'shuffle': False,
            'num_workers': 4,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }

def get_dataset_testC(maxT,root,dict_dir,batch_size=128):
    return {
        'dataset_test': colored_lmdbDataset,
        'dataset_test_args': {
            'roots': [root],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dataloader_test': {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 4,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }
def get_dataset_testCF(maxT,root,dict_dir):
    return {
        'dataset_test': colored_lmdbDatasetT,
        'dataset_test_args': {
            'roots': [root],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
            "force_target_ratio": 1.,
        },
        'dataloader_test': {
            'batch_size': 128,
            'shuffle': False,
            'num_workers': 4,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }

def get_std_uncased_ds(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt'):
    return {
        'dataset_train': lmdbDataset,
        'dataset_train_args': {
            'roots': [get_nips14(root),get_cvpr16(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 3,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_iiit5k(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 3,
        },
        'case_sensitive': False,
        'te_case_sensitive': False,

        'dict_dir' : dict_dir
    }
def get_std_uncased_ds_rgb(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt'):
    return {
        'dataset_train': colored_lmdbDataset,
        'dataset_train_args': {
            'roots': [get_nips14(root),get_cvpr16(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 3,
        },

        'dataset_test': colored_lmdbDataset,
        'dataset_test_args': {
            'roots': [get_iiit5k(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 3,
        },
        'case_sensitive': False,
        'te_case_sensitive': False,

        'dict_dir' : dict_dir
    }

def get_std_uncased_dsCQA(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt'):
    return {
        'dataset_train': colored_lmdbDataset,
        'dataset_train_args': {
            'roots': [get_nips14(root),get_cvpr16(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
            'qhb_aug':True
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 3,
        },

        'dataset_test': colored_lmdbDataset,
        'dataset_test_args': {
            'roots': [get_iiit5k(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 3,
        },
        'case_sensitive': False,
        'te_case_sensitive': False,

        'dict_dir' : dict_dir
    }
def get_std_uncased_dsCQAF(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt'):
    return {
        'dataset_train': colored_lmdbDataset,
        'dataset_train_args': {
            'roots': [get_nips14(root),get_cvpr16(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
            'qhb_aug':True,
            "force_target_ratio":1.,
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 3,
        },

        'dataset_test': colored_lmdbDataset,
        'dataset_test_args': {
            'roots': [get_iiit5k(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
            "force_target_ratio": 1.,
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 3,
        },
        'case_sensitive': False,
        'te_case_sensitive': False,

        'dict_dir' : dict_dir
    }

def get_std_uncased_dsCQA_semi(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt'):
    return {
        'dataset_train': colored_lmdbDataset_semi,
        'dataset_train_args': {
            'roots': [get_nips14(root),get_cvpr16(root)],
            "cased_annoatations" :[True,False],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
            'qhb_aug':True
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 3,
        },

        'dataset_test': colored_lmdbDataset,
        'dataset_test_args': {
            'roots': [get_iiit5k(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 3,
        },
        'case_sensitive': False,
        'te_case_sensitive': False,

        'dict_dir' : dict_dir
    }

def get_std_uncased_ds_semi(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt'):
    return {
        'dataset_train': lmdbDataset_semi,
        'dataset_train_args': {
            'roots': [get_nips14(root),get_cvpr16(root)],
            "cased_annoatations" :[True,False],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 3,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_iiit5k(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 3,
        },
        'case_sensitive': False,
        'te_case_sensitive': False,

        'dict_dir' : dict_dir
    }

def get_test_all_uncased_ds(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt'):
    return {
        "dict_dir":dict_dir,
        "case_sensitive": False,
        "te_case_sensitive": False,

        "datasets":{
            # "ic15": get_dataset_test(maxT, get_IC15_1811(root), dict_dir),

            "CUTE": get_dataset_test(maxT, get_cute(root), dict_dir),

            "IC03": get_dataset_test(maxT, get_IC03_867(root), dict_dir),
            "IIIT5k": get_dataset_test(maxT, get_iiit5k(root), dict_dir),
            "SVTP": get_dataset_test(maxT, get_SVTP(root), dict_dir),
        "SVT": get_dataset_test(maxT, get_SVT(root), dict_dir),
        "IC13": get_dataset_test(maxT, get_IC13_1015(root), dict_dir),
        }
            }
def get_test_all_uncased_dsrgb(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt',batchsize=128):
    return {
        "dict_dir":dict_dir,
        "case_sensitive": False,
        "te_case_sensitive": False,
        "datasets":{
        "CUTE": get_dataset_testC(maxT, get_cute(root), dict_dir,batchsize),
        "IIIT5k": get_dataset_testC(maxT, get_iiit5k(root), dict_dir,batchsize),
        "SVT": get_dataset_testC(maxT, get_SVT(root), dict_dir,batchsize),
        "IC03": get_dataset_testC(maxT, get_IC03_867(root), dict_dir,batchsize),
        "IC13": get_dataset_testC(maxT, get_IC13_1015(root), dict_dir,batchsize),
        }
            }
def get_test_all_uncased_dsrgbF(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt'):
    return {
        "dict_dir":dict_dir,
        "case_sensitive": False,
        "te_case_sensitive": False,
        "datasets":{
        "SVTP": get_dataset_testCF(maxT, get_SVTP(root), dict_dir),
        "CUTE": get_dataset_testCF(maxT, get_cute(root), dict_dir),
        "IIIT5k": get_dataset_testCF(maxT, get_iiit5k(root), dict_dir),
        "SVT": get_dataset_testCF(maxT, get_SVT(root), dict_dir),
        "IC03": get_dataset_testCF(maxT, get_IC03_867(root), dict_dir),
        "IC13": get_dataset_testCF(maxT, get_IC13_1015(root), dict_dir),
        }
            }


def get_debug_uncased_ds(maxT=25,root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset,
        'dataset_train_args': {
            'roots': [get_iiit5k(root),get_iiit5k(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 3,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_iiit5k(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 3,
        },
        'case_sensitive': False,
        'dict_dir' : '../../dict/dic_36.txt'
    }


def get_lmdb_ds(trroot,teroot,maxT=25,dict_dir=None):
    return {
        'dataset_train': lmdbDataset_repeat,
        'dataset_train_args': {
            'roots': trroot,
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': teroot,
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }



def get_chs(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeat,
        'dataset_train_args': {
            "repeat": 20,
            'roots': [get_lsvt_path(root),
                      get_ctw(root),
                      get_mlt_chlat_path(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mlt_chlatval(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }



def get_chs_rgen(trgenroo='/home/lasercat/ssddata/synth_data/synthmeta3755af.pt',teroot='/home/lasercat/ssddata/ctw_fslchr/',dict_dir=None):
    return {
        'dataset_train': nekoOLSDataset,
        'dataset_train_args': {
            'root': trgenroo,
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 32,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [teroot],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
        },
        'dataloader_test': {
            'batch_size': 36,
            'shuffle': False,
            'num_workers': 5,
        },

        'case_sensitive': False,
        'dict_dir': dict_dir
    }

def get_chs_wart(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeat,
        'dataset_train_args': {
            "repeat": 16,
            'roots': [get_lsvt_path(root),
                      get_ctw(root),
                      get_mlt_chlat_path(root),
                      get_art_path(root),
                      ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mlt_chlatval(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }
def get_chs_wartH(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeatH,
        'dataset_train_args': {
            "repeat": 16,
            'roots': [get_lsvt_path(root),
                      get_ctw(root),
                      get_mlt_chlat_path(root),
                      get_art_path(root),
                      ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mlt_chlatval(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }

def get_chs_wartHK(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeatH,
        'dataset_train_args': {
            "repeat": 16,
            'roots': [get_lsvtK_path(root),
                      get_ctwK_path(root),
                      get_mlt_chlatK_path(root),
                      get_artK_path(root),
                      ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mlt_chlatval(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }
def get_chs_wrctwHK(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeatH,
        'dataset_train_args': {
            "repeat": 16,
            'roots': [get_lsvtK_path(root),
                      get_ctwK_path(root),
                      get_mlt_chlatK_path(root),
                      get_artK_path(root),
                      get_rctwK_path(root),
                      ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mlt_chlatval(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }
def get_chs_wrctwHKS(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeatHS,
        'dataset_train_args': {
            "repeat": 16,
            'roots': [get_lsvtK_path(root),
                      get_ctwK_path(root),
                      get_mlt_chlatK_path(root),
                      get_artK_path(root),
                      get_rctwK_path(root),
                      ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mlt_chlatval(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }

def get_chs_ARTHS(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeatHS,
        'dataset_train_args': {
            "repeat": 24,
            'roots': [get_art_path(root),
                      ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_artval(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }

def get_chs_8mHS(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeatHS,
        'dataset_train_args': {
            "repeat": 1,
            'roots': get_8mtr(root)+[],
            'img_height': 32,
            'img_width': 256,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 45,
            'shuffle': True,
            'num_workers': 5,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_qhbcsvtr(root)],
            'img_height': 32,
            'img_width': 256,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }
def get_chs_ctw2kS(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeatS,
        'dataset_train_args': {
            "repeat": 16,
            'roots': [get_ctw2k(root)],
            'img_height': 32,
            'img_width': 64,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 64,
            'shuffle': True,
            'num_workers': 5,
        },
        "temeta": "../../dict/ctw2kchunseen_500.pt",
        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_ctw2kus(root)],
            'img_height': 32,
            'img_width': 64,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 1024,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }

def get_chs_lsvtdbg(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeatH,
        'dataset_train_args': {
            "repeat": 16,
            'roots': [get_lsvt_path(root)
                      ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 2,
            'shuffle': True,
            'num_workers': 2,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_lsvt_path(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }


def get_jap(maxT,dict_dir=None,root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeat,
        'dataset_train_args': {
            "repeat": 16,
            'roots': [get_mlt_chlat_path(root),
                       ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT": maxT
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mlt_jpval(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir': dict_dir
    }

def get_ulsvtA(maxT,dict_dir=None,root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeat,
        'dataset_train_args': {
            "repeat": 16,
            'roots': [get_mlt_chlat_path(root),
                       ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT": maxT
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_ulsvta(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 48,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir': dict_dir
    }

def get_chs_rgen(trgenroo='/home/lasercat/ssddata/synth_data/synthmeta3755af.pt',teroot='/home/lasercat/ssddata/ctw_fslchr/',dict_dir=None):
    return {
        'dataset_train': nekoOLSDataset,
        'dataset_train_args': {
            'root': trgenroo,
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 32,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [teroot],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
        },
        'dataloader_test': {
            'batch_size': 36,
            'shuffle': False,
            'num_workers': 5,
        },

        'case_sensitive': False,
        'dict_dir': dict_dir
    }

def get_chs_cgen(trgenroo='/home/lasercat/ssddata/synth_data/Csynthmeta3817af.pt',teroot='/home/lasercat/ssddata/ctw_fslchr/',dict_dir=None):
    return {
        'dataset_train': nekoOLSCDataset,
        'dataset_train_args': {
            'root': trgenroo,
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 32,
        },

        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [teroot],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
        },
        'dataloader_test': {
            'batch_size': 36,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        'dict_dir': dict_dir
    }


