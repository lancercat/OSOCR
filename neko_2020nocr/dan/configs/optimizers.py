from torch import optim
def get_baseline_opt():
    return {
        # optim for FE
        'optimizer_0': optim.Adadelta,
        'optimizer_0_args':{
            'lr': 1.0,
        },

        'optimizer_0_scheduler': optim.lr_scheduler.MultiStepLR,
        'optimizer_0_scheduler_args': {
            'milestones': [3, 5],
            'gamma': 0.1,
        },

        # optim for CAM
        'optimizer_1': optim.Adadelta,
        'optimizer_1_args':{
            'lr': 1.0,
        },
        'optimizer_1_scheduler': optim.lr_scheduler.MultiStepLR,
        'optimizer_1_scheduler_args': {
            'milestones': [3, 5],
            'gamma': 0.1,
        },

        # optim for DTD
        'optimizer_2': optim.Adadelta,
        'optimizer_2_args':{
            'lr': 1.0,
        },
        'optimizer_2_scheduler': optim.lr_scheduler.MultiStepLR,
        'optimizer_2_scheduler_args': {
            'milestones': [3, 5],
            'gamma': 0.1,
        },
    }
def get_dos_optim():
    return {
        # optim for FE
        'optimizer_0': optim.Adadelta,
        'optimizer_0_args': {
            'lr': 1.0,
        },

        'optimizer_0_scheduler': optim.lr_scheduler.MultiStepLR,
        'optimizer_0_scheduler_args': {
            'milestones': [3, 5],
            'gamma': 0.1,
        },

        # optim for CAM
        'optimizer_1': optim.Adadelta,
        'optimizer_1_args': {
            'lr': 1.0,
        },
        'optimizer_1_scheduler': optim.lr_scheduler.MultiStepLR,
        'optimizer_1_scheduler_args': {
            'milestones': [3, 5],
            'gamma': 0.1,
        },

        # optim for DTD
        'optimizer_2': optim.Adadelta,
        'optimizer_2_args': {
            'lr': 1.0,
        },
        'optimizer_2_scheduler': optim.lr_scheduler.MultiStepLR,
        'optimizer_2_scheduler_args': {
            'milestones': [3, 5],
            'gamma': 0.1,
        },
        # optim for PE
        'optimizer_3': optim.Adadelta,
        'optimizer_3_args': {
            'lr': 1.0,
        },

        'optimizer_3_scheduler': optim.lr_scheduler.MultiStepLR,
        'optimizer_3_scheduler_args': {
            'milestones': [3, 5],
            'gamma': 0.1,
        },

    }