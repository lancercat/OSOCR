from neko_2020nocr.dan.dataloaders.dataset_scene import *
from torchvision import transforms
from neko_2020nocr.dan.methods_pami.pami_osds_paths import \
    get_stdhwdbtr,get_stdhwdbte,\
    get_stdctwchtr,get_stdctwchte


def get_chs_hwdbS(trcnt=1000,maxT=2, dict_dir=None, root="/home/lasercat/ssddata/"):
    teroot,tedict=get_stdhwdbte(root);
    trroot,trdict=get_stdhwdbtr(trcnt,root);

    return {
        'dataset_train': lmdbDataset_repeatS,
        'dataset_train_args': {
            "repeat": 1,
            'roots': [trroot],
            'img_height': 32,
            'img_width': 64,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT": maxT
        },
        'dataloader_train': {
            'batch_size': 160,
            'shuffle': True,
            'num_workers': 5,
        },
        "temeta":tedict,
        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [teroot],
            'img_height': 32,
            'img_width': 64,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 160,
            'shuffle': False,
            'num_workers': 5,
        },
        'tr_case_sensitive': False,
        'te_case_sensitive': False,
        'case_sensitive': False,
        'dict_dir': trdict
    }

#
#
def get_chs_ctwS(trcnt=1000,maxT=25, dict_dir=None, root="/home/lasercat/ssddata/"):
    rep=1;
    if(trcnt<=700):
        rep=2;
    teroot, tedict = get_stdctwchte(root);
    trroot, trdict = get_stdctwchtr(trcnt, root);
    return {
        'dataset_train': lmdbDataset_repeatS,
        'dataset_train_args': {
            "repeat": rep,
            'roots': [trroot],
            'img_height': 32,
            'img_width': 64,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT":maxT
        },
        'dataloader_train': {
            'batch_size': 160,
            'shuffle': True,
            'num_workers': 5,
        },
        "temeta":tedict,
        'dataset_test': lmdbDataset,
        'dataset_test_args': {
            'roots': [teroot],
            'img_height': 32,
            'img_width': 64,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dataloader_test': {
            'batch_size': 8,
            'shuffle': False,
            'num_workers': 5,
        },
        'case_sensitive': False,
        "te_case_sensitive" : False,
        "tr_case_sensitive": False,

        'dict_dir' : trdict
    }
