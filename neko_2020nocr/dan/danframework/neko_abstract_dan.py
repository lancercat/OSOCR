from neko_2020nocr.dan.common.common import display_cfgs,load_dataset,load_all_dataset,Updata_Parameters,generate_optimizer,flatten_label;
from torch import nn;
import datetime
import torch;
from torch.utils.data import DataLoader
from neko_2020nocr.dan.utils import Loss_counter
import os
class neko_abstract_DAN:

    def get_ar_cntr(this,key,case_sensitive):
        return None;
    def get_rej_ar_cntr(this,key,case_sensitive):
        return None
    def get_loss_cntr(this,show_interval):
        return Loss_counter(show_interval);
    def set_cntrs(this):
        this.train_acc_counter = this.get_ar_cntr('train accuracy: ',
                                                 this.cfgs.dataset_cfgs['te_case_sensitive'])
        this.test_acc_counter = this.get_ar_cntr('\ntest accuracy: ',
                                                this.cfgs.dataset_cfgs['te_case_sensitive'])
        this.test_rej_counter = this.get_rej_ar_cntr('\ntest rej accuracy: ',
                                                this.cfgs.dataset_cfgs['te_case_sensitive'])
        this.loss_counter = this.get_loss_cntr(this.cfgs.global_cfgs['show_interval'])


    def setuploss(this):
        this.criterion_CE = nn.CrossEntropyLoss().cuda()
    def load_network(this):
        pass;
    def set_up_etc(this):
        pass;
    def setup_dataloaders(this):
        try:
            if "dataset_train" in this.cfgs.dataset_cfgs:
                this.train_loader, this.test_loader = load_dataset(this.cfgs, DataLoader)
                this.set_cntrs();
            else:
                this.all_test_loaders = load_all_dataset(this.cfgs, DataLoader);
        except:
            print("no scheduled datasets")
    def setup(this):
        this.model = this.load_network();
        this.setuploss();
        this.optimizers, this.optimizer_schedulers = generate_optimizer(this.cfgs, this.model);
        print('preparing done')
        # --------------------------------
        # prepare tools
        this.set_up_etc();


        pass;


    def __init__(this,cfgs):
        this.cfgs = cfgs;
        this.cfgs.mkdir(this.cfgs.saving_cfgs['saving_path']);
        this.setup_dataloaders();
        this.setup();
        # Let's not show cfgs every time
        # display_cfgs(this.cfgs, this.model)

        # ---------------------------------

    def test(this,test_loader, model, tools ,miter=1000,debug=False,dbgpath=None):
        pass;
    def dump(this,test_loader, model, tools,miter=100,debug=False,dbgpath=None):
        pass;

    def fpbp(this,data,label,cased=None):
        pass;

    def fpbp2(this, sample_batched):
        data = sample_batched['image']
        label = sample_batched['label']
        if("cased" in sample_batched):
            return this.fpbp(data,label,sample_batched["cased"]);
        return this.fpbp(data,label);

    def runtest(this,miter=1000,debug=False,dbgpath=None,measure_rej=False):
        pass
    def rundump(this,miter=1000,debug=False,dbgpath=None):
        pass

    def dump_chk(this,dbgpath):
        with torch.no_grad():
            this.rundump(this.cfgs.global_cfgs['test_miter'], True, dbgpath)
        exit()
    def show(this):
        this.train_acc_counter.show();
        this.train_acc_counter.clear();

    def tr_iter(this,nEpoch,batch_idx,sample_batched,total_iters):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare

        this.fpbp2(sample_batched);

        for i in range(len(this.model)):
            nn.utils.clip_grad_norm_(this.model[i].parameters(), 20, 2)
        Updata_Parameters(this.optimizers, frozen=[])
        # visualization and saving
        if batch_idx % this.cfgs.global_cfgs['show_interval'] == 0 and batch_idx != 0:
            print(datetime.datetime.now().strftime('%H:%M:%S'))
            loss, terms = this.loss_counter.get_loss_and_terms();
            print('Epoch: {}, Iter: {}/{}, Loss dan: {}'.format(
                nEpoch,
                batch_idx,
                total_iters,
                loss))
            if (len(terms)):
                print(terms);
            this.show();
        if batch_idx % this.cfgs.global_cfgs['test_interval'] == 0 and batch_idx != 0:
            with torch.no_grad():
                this.runtest(this.cfgs.global_cfgs['test_miter']);
        if nEpoch % this.cfgs.saving_cfgs['saving_epoch_interval'] == 0 and \
                batch_idx % this.cfgs.saving_cfgs['saving_iter_interval'] == 0 and \
                batch_idx != 0:
            for i in range(0, len(this.model)):
                torch.save(this.model[i].state_dict(),
                           this.cfgs.saving_cfgs['saving_path'] + 'E{}_I{}-{}_M{}.pth'.format(
                               nEpoch, batch_idx, total_iters, i))

    def run(this,dbgpath=None,measure_rej=False):
        # ---------------------------------
        if this.cfgs.global_cfgs['state'] == 'Test':
            with torch.no_grad():
                this.runtest(this.cfgs.global_cfgs['test_miter'],False,dbgpath,measure_rej=measure_rej)
            return
        # --------------------------------
        total_iters = len(this.train_loader)
        for model in this.model:
            model.train();
        for nEpoch in range(0, this.cfgs.global_cfgs['epoch']):
            for batch_idx, sample_batched in enumerate(this.train_loader):
                this.tr_iter(nEpoch,batch_idx,sample_batched,total_iters)
            Updata_Parameters(this.optimizer_schedulers, frozen=[])
            for i in range(0, len(this.model)):
                torch.save(this.model[i].state_dict(),
                           this.cfgs.saving_cfgs['saving_path'] + 'E{}_M{}.pth'.format(
                               nEpoch, i))


    def test_all(this,dbgpath=None):
        debug=False;
        if(dbgpath):
            debug=True;
        retdb = {}
        for ds in this.all_test_loaders:
            if(dbgpath is not None):
                dspath=os.path.join(dbgpath,ds);
            else:
                dspath=None;
            test_acc_counter=this.get_ar_cntr(ds,False);
            with torch.no_grad():
                this.test(this.all_test_loaders[ds], this.model, [test_acc_counter,
                                                       flatten_label,
                                                ], miter=999999999, debug=debug, dbgpath=dspath);
            retdb[ds]=test_acc_counter;
            # test_acc_counter.show();
        exit()
