import torch;

def display_cfgs(cfgs,models):
    try:
        print('global_cfgs')
        cfgs.showcfgs(cfgs.global_cfgs)
        print('dataset_cfgs')
        cfgs.showcfgs(cfgs.dataset_cfgs)
    except:
        pass;
    print('net_cfgs')
    cfgs.showcfgs(cfgs.net_cfgs)
    print('optimizer_cfgs')
    cfgs.showcfgs(cfgs.optimizer_cfgs)
    print('saving_cfgs')
    cfgs.showcfgs(cfgs.saving_cfgs)
    for model in models:
        print(model)


def flatten_label(target):
    label_flatten = []
    label_length = []
    for i in range(0, target.size()[0]):
        cur_label = target[i].tolist()
        label_flatten += cur_label[:cur_label.index(0)+1]
        label_length.append(cur_label.index(0)+1)
    label_flatten = torch.LongTensor(label_flatten)
    label_length = torch.IntTensor(label_length)
    return (label_flatten, label_length)
def flatten_label_oc(target,cased):
    label_flatten = []
    cased_flatten=[];
    label_length = []
    for i in range(0, target.size()[0]):
        cur_label = target[i].tolist()
        label_flatten += cur_label[:cur_label.index(0)+1]
        cased_flatten+=[cased[i] for j in cur_label[:cur_label.index(0)+1]];
        label_length.append(cur_label.index(0)+1)
    label_flatten = torch.LongTensor(label_flatten)
    label_length = torch.IntTensor(label_length)
    cased_flatten = torch.LongTensor(cased_flatten)
    return (label_flatten, label_length,cased_flatten)
def Train_or_Eval(models, state = 'Train'):
    for model in models:
        if state == 'Train':
            model.train()
        else:
            model.eval()
def Zero_Grad(models):
    for model in models:
        model.zero_grad()
def Updata_Parameters(optimizers, frozen):
    for i in range(0, len(optimizers)):
        if i not in frozen:
            optimizers[i].step()

#---------------------dataset
def load_dataset(cfgs,DataLoader):
    # train_data_set = cfgs.dataset_cfgs['dataset_train'](**cfgs.dataset_cfgs['dataset_train_args'])
    # train_loader = DataLoader(train_data_set, **cfgs.dataset_cfgs['dataloader_train'])

    try:
        train_data_set = cfgs.dataset_cfgs['dataset_train'](**cfgs.dataset_cfgs['dataset_train_args'])
        train_loader = DataLoader(train_data_set, **cfgs.dataset_cfgs['dataloader_train'])
    except:
        train_loader=None
        print("Failed loading training data")
    test_data_set = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_args'])
    test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs['dataloader_test'])
    # pdb.set_trace()
    return (train_loader, test_loader)

def load_dataset_driect(dataset_cfgs,DataLoader):
    train_data_set = dataset_cfgs['dataset_train'](**dataset_cfgs['dataset_train_args'])
    train_loader = DataLoader(train_data_set, **dataset_cfgs['dataloader_train'])

    test_data_set = dataset_cfgs['dataset_test'](**dataset_cfgs['dataset_test_args'])
    test_loader = DataLoader(test_data_set, **dataset_cfgs['dataloader_test'])
    # pdb.set_trace()
    return (train_loader, test_loader)
#---------------------dataset
def load_all_dataset(cfgs,DataLoader):
    retdict={};
    for ds in cfgs.dataset_cfgs["datasets"]:
        try:
            test_data_set = cfgs.dataset_cfgs["datasets"][ds]['dataset_test'](**cfgs.dataset_cfgs["datasets"][ds]['dataset_test_args'])
            test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs["datasets"][ds]['dataloader_test'])
            retdict[ds]=test_loader;
        except:
            print ("missing DS",ds,", skipping");
    # pdb.set_trace()
    return retdict

#----------------------optimizer
def generate_optimizer(cfgs,models):
    out = []
    scheduler = []
    for i in range(0, len(models)):
        out.append(cfgs.optimizer_cfgs['optimizer_{}'.format(i)](
                    models[i].parameters(),
                    **cfgs.optimizer_cfgs['optimizer_{}_args'.format(i)]))
        scheduler.append(cfgs.optimizer_cfgs['optimizer_{}_scheduler'.format(i)](
                    out[i],
                    **cfgs.optimizer_cfgs['optimizer_{}_scheduler_args'.format(i)]))
    return tuple(out), tuple(scheduler)
