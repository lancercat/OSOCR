from neko_2020nocr.dan.dataloaders.dataset_scene import *
from torchvision import transforms
from neko_2020nocr.dan.configs.datasets.ds_paths import *

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


def get_std_cased_ds(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt'):
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
            'roots': [get_IC15_2077(root)],
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
        'case_sensitive': True,
        'te_case_sensitive': False,
        'dict_dir' : dict_dir
    }

def get_std_uncased_dsXL(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt'):
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
            'batch_size': 96,
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
        'te_case_sensitive':False,
        'case_sensitive': False,
        'dict_dir' : dict_dir
    }
