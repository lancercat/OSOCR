# MODEL_ROOT='../../trained_models/';
from neko_sdk.root import find_model_root
def get_train_cfg(epochs=10):
    return {
        'state': 'Train',
        'epoch': epochs,
        'show_interval': 50,
        'test_interval': 1000,
        'test_miter':100,
    };

def get_test_cfg():
    return {
        'state': 'Test',
        'epoch': 10,
        'show_interval': 50,
        'test_interval': 1000,
        'test_miter': 100000,
        "print_net":False,
    };
def get_save_cfgs(prefix,root_override=None):
    models=find_model_root(root_override)+prefix+"_";
    return {
        'saving_iter_interval': 20000,
        'saving_epoch_interval': 1,
        'saving_path': models,
    }