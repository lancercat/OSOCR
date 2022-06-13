from neko_2020nocr.dan.dataloaders.dataset_scene import *
from torchvision import transforms
from neko_2020nocr.dan.methods_pr.pami_osds_paths import \
    get_lsvtK_path,get_ctwK_path,get_mlt_chlatK_path,get_artK_path,get_rctwK_path,\
    get_mltjp_path,get_mlt_krK_path,get_mltkr_path

def get_chs_wrctwHKS(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeatHS,
        'dataset_train_args': {
            "repeat": 3,
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
        "temeta": os.path.join(root,"dicts","dabjpmlt.pt"),
        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mltjp_path(root)],
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
        "te_case_sensitive" : False,
        "tr_case_sensitive" : False,
        'dict_dir' : dict_dir
    }
def get_jap_test(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset,
        'dataset_train_args': {
            'roots': [get_mltjp_path(root)],
            'img_height': 32,
            'img_width': 32,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 16,
            'shuffle': True,
            'num_workers': 12,
        },
        "temeta": os.path.join(root,"dicts","dabjpmlt.pt"),
        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mltjp_path(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 16,
            'shuffle': False,
            'num_workers': 12,
        },
        'case_sensitive': False,
        "te_case_sensitive" : False,
        "tr_case_sensitive" : False,
        'dict_dir' : dict_dir
    }

def get_kr_test(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset,
        'dataset_train_args': {
            "repeat": 3,
            'roots': [ get_mltkr_path(root) ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
            "qhb_aug":True
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 12,
        },
        "temeta": os.path.join(root,"dicts","dabkrmlt.pt"),
        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mltkr_path(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 16,
            'shuffle': False,
            'num_workers': 12,
        },
        'case_sensitive': False,
        "te_case_sensitive" : False,
        "tr_case_sensitive" : False,
        'dict_dir' : dict_dir
    }

def get_chs_wrctwHKSK(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': lmdbDataset_repeatHS,
        'dataset_train_args': {
            "repeat": 3,
            'roots': [get_lsvtK_path(root),
                      get_ctwK_path(root),
                      get_mlt_chlatK_path(root),
                      get_artK_path(root),
                      get_rctwK_path(root),
                      get_mlt_krK_path(root)
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
        "temeta": os.path.join(root,"dicts","dabjpmlt.pt"),
        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mltjp_path(root)],
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
        "te_case_sensitive" : False,
        "tr_case_sensitive" : False,
        'dict_dir' : dict_dir
    }
def get_chs_wrctwHKSCQA(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': colored_lmdbDataset_repeatHS,
        'dataset_train_args': {
            "repeat": 3,
            'roots': [get_lsvtK_path(root),
                      get_ctwK_path(root),
                      get_mlt_chlatK_path(root),
                      get_artK_path(root),
                      get_rctwK_path(root)
                      ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
            "qhb_aug":True
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },
        "temeta": os.path.join(root,"dicts","dabjpmlt.pt"),
        'dataset_test': colored_lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mltjp_path(root)],
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
        "te_case_sensitive" : False,
        "tr_case_sensitive" : False,
        'dict_dir' : dict_dir
    }
def get_chs_wrctwHKSCQA(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': colored_lmdbDataset_repeatHS,
        'dataset_train_args': {
            "repeat": 3,
            'roots': [get_lsvtK_path(root),
                      get_ctwK_path(root),
                      get_mlt_chlatK_path(root),
                      get_artK_path(root),
                      get_rctwK_path(root)
                      ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
            "qhb_aug":True
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },
        "temeta": os.path.join(root,"dicts","dabjpmlt.pt"),
        'dataset_test': colored_lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mltjp_path(root)],
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
        "te_case_sensitive" : False,
        "tr_case_sensitive" : False,
        'dict_dir' : dict_dir
    }
def get_test_kr_rgb(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': colored_lmdbDataset_repeatHS,
        'dataset_train_args': {
            "repeat": 3,
            'roots': [ get_mltkr_path(root) ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
            "qhb_aug":True
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 12,
        },
        "temeta": os.path.join(root,"dicts","dabkrmlt.pt"),
        'dataset_test': colored_lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mltkr_path(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 16,
            'shuffle': False,
            'num_workers': 12,
        },
        'case_sensitive': False,
        "te_case_sensitive" : False,
        "tr_case_sensitive" : False,
        'dict_dir' : dict_dir
    }

def get_test_jap_rgb(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/",meta=None):
    if(meta is None):
        meta=os.path.join(root,"dicts","dabjpmlt.pt");
    return {
        'dataset_train': colored_lmdbDataset_repeatHS,
        'dataset_train_args': {
            "repeat": 3,
            'roots': [ get_mltjp_path(root) ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
            "qhb_aug":True
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 12,
        },
        "temeta": meta,
        'dataset_test': colored_lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mltjp_path(root)],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 16,
            'shuffle': False,
            'num_workers': 12,
        },
        'case_sensitive': False,
        "te_case_sensitive" : False,
        "tr_case_sensitive" : False,
        'dict_dir' : dict_dir
    }

def get_chs_wrctwHKSCQAF(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': colored_lmdbDataset_repeatHS,
        'dataset_train_args': {
            "repeat": 3,
            'roots': [get_lsvtK_path(root),
                      get_ctwK_path(root),
                      get_mlt_chlatK_path(root),
                      get_artK_path(root),
                      get_rctwK_path(root)
                      ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
            "qhb_aug":True,
            "force_target_ratio":1.,

        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },
        "temeta": os.path.join(root,"dicts","dabjpmlt.pt"),
        'dataset_test': colored_lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mltjp_path(root)],
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
            'num_workers': 5,
        },
        'case_sensitive': False,
        "te_case_sensitive" : False,
        "tr_case_sensitive" : False,
        'dict_dir' : dict_dir
    }

def get_chs_wrctwHKSKCQA(maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    return {
        'dataset_train': colored_lmdbDataset_repeatHS,
        'dataset_train_args': {
            "repeat": 3,
            'roots': [get_lsvtK_path(root),
                      get_ctwK_path(root),
                      get_mlt_chlatK_path(root),
                      get_artK_path(root),
                      get_rctwK_path(root),
                      get_mlt_krK_path(root)
                      ],
            'img_height': 32,
            'img_width': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT,
            "qhb_aug":True
        },
        'dataloader_train': {
            'batch_size': 48,
            'shuffle': True,
            'num_workers': 5,
        },
        "temeta": os.path.join(root,"dicts","dabjpmlt.pt"),
        'dataset_test': colored_lmdbDataset,
        'dataset_test_args': {
            'roots': [get_mltjp_path(root)],
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
        "te_case_sensitive" : False,
        "tr_case_sensitive" : False,
        'dict_dir' : dict_dir
    }
