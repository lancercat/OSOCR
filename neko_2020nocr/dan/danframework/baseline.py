from torch.utils.data import DataLoader
#------------------------
from neko_2020nocr.dan.utils import *
from neko_2020nocr.dan.common.common import display_cfgs,load_dataset,Zero_Grad,Train_or_Eval,generate_optimizer,Updata_Parameters,flatten_label
from neko_2020nocr.dan.common.common_bl import load_network;
from neko_2020nocr.dan.danframework.neko_abstract_dan import neko_abstract_DAN;
import torch;
import os;
import time;
class baseline(neko_abstract_DAN):
    def load_network(this):
        return load_network(this.cfgs);
    def get_ar_cntr(this,key,case_sensitive):
        return  Attention_AR_counter(key,this.cfgs.dataset_cfgs['dict_dir'],case_sensitive);
    def get_loss_cntr(this,show_interval):
        return Loss_counter(show_interval);
    def set_up_etc(this):
        this.encdec = cha_encdec(this.cfgs.dataset_cfgs['dict_dir'], this.cfgs.dataset_cfgs['case_sensitive'])
        pass;

    def rundump(this, miter=1000, debug=False, dbgpath=None):
        if (dbgpath is not None):
            debug = True;
        this.dump((this.test_loader),
                  this.model,
                  [this.encdec,
                   flatten_label,
                   this.test_acc_counter], miter, debug, dbgpath)

    def runtest(this,miter=1000,debug=False,dbgpath=None,measure_rej=False):
        if(dbgpath is not None):
            debug=True;
        this.test((this.test_loader),
             this.model,
             [this.encdec,
              flatten_label,
              this.test_acc_counter],miter,debug)


    def test_all(this,dbgpath=None):
        debug=False;
        if(dbgpath):
            debug=True;
        retdb = {}
        for ds in this.all_test_loaders:
            test_acc_counter=this.get_ar_cntr(ds,False);
            with torch.no_grad():
                this.test(this.all_test_loaders[ds], this.model, [this.encdec,flatten_label,
                                                                  test_acc_counter,
                                                ], miter=999999999, debug=debug);
            retdb[ds]=test_acc_counter;
            # test_acc_counter.show();
        exit()
    def test_prune(this,dbgpath=None):
        debug=False;
        if(dbgpath):
            debug=True;
        retdb = {};
        this.model[0].prune(0.1)
        for ds in this.all_test_loaders:
            test_acc_counter=this.get_ar_cntr(ds,False);

            with torch.no_grad():
                this.test(this.all_test_loaders[ds],this.model , [this.encdec,flatten_label,
                                                                  test_acc_counter,
                                                ], miter=999999999, debug=debug, dbgpath=dbgpath);
            retdb[ds]=test_acc_counter;
            # test_acc_counter.show();
        exit()

    def test(this,test_loader, model, tools,miter=1000,debug=False):
        Train_or_Eval(model, 'Eval');
        i=0;
        tot=0;
        start=time.time();
        for sample_batched in test_loader:
            if i>miter:
                break;
            i+=1;
            data = sample_batched['image']
            label = sample_batched['label']
            tot+=len(label);
            target = tools[0].encode(label)

            data = data.cuda()
            target = target
            label_flatten, length = tools[1](target)
            target, label_flatten = target.cuda(), label_flatten.cuda()

            features = model[0](data)
            A = model[1](features)
            output, out_length = model[2](features[-1], A, target, length, True)
            tools[2].add_iter(output, out_length, length, label,debug)
        end=time.time();
        print((end-start)/tot,tot)
        tools[2].show();
        tools[2].clear();
        Train_or_Eval(model, 'Train')

    def _dumpaug(this, test_loader, model,tools, miter=100, debug=False, dbgpath=None):
        Train_or_Eval(model, 'Eval');
        i = 0;
        dcnt=0;
        for sample_batched in test_loader:
            if i > miter:
                break;
            i += 1;
            data = sample_batched['image']
            label = sample_batched['label']

            data = data.cuda()
            model[0].jitter.train()
            jittered= model[0].jitter.forward_tr(data);
            im=torch.cat([data,jittered],dim=2).cpu().detach().numpy();
            target = tools[0].encode(label)
            label_flatten, length = tools[1](target)
            target, label_flatten = target.cuda(), label_flatten.cuda()

            features = model[0](data)
            A = model[1](features)
            output, out_length = model[2](features[-1], A, target, length, True)
            res=tools[2].add_iter(output, out_length, length, label,debug)


            for j in range(len(im)):
                cv2.imwrite(os.path.join(dbgpath, str(dcnt)+"_jittered.jpg"), ((im[j][0]+0.1)*200).astype(np.uint8));
                dcnt+=1
                print(os.path.join(dbgpath,str(dcnt),"orig.jpg"))
                pass;



        Train_or_Eval(model, 'Train')
    def dumpaug(this,miter=1000,debug=False,dbgpath=None):
        if(dbgpath is not None):
            debug=True;

        with torch.no_grad():
            this._dumpaug(this.train_loader,this.model,[this.encdec,
              flatten_label,
              this.test_acc_counter],miter,True,dbgpath)

    def dump(this,test_loader, model, tools,miter=100,debug=False,dbgpath=None):
        Train_or_Eval(model, 'Eval');
        i=0;
        start=time.time();
        scnt=0;
        for sample_batched in test_loader:
            if i>miter:
                break;
            i+=1;
            data = sample_batched['image']
            label = sample_batched['label']
            scnt+=data.shape[0];
            target = tools[0].encode(label)
            data = data.cuda()
            target = target
            label_flatten, length = tools[1](target)
            target, label_flatten = target.cuda(), label_flatten.cuda()
            features,dump = model[0].forward_dump(data)
            A = model[1]([features[-1]])
            dump["A"]=A.detach().cpu();
            torch.save(dump,os.path.join(dbgpath,"pack_"+str(i)+".pt"))
            output, out_length = model[2](features[-1], A, target, length, True)
            tools[2].add_iter(output, out_length, length, label,debug)
        end=time.time();
        print((end-start)/scnt)
        tools[2].show();
        tools[2].clear();
        Train_or_Eval(model, 'Train')
    def fpbp(this, data, label,cased=None):
        target = this.encdec.encode(label)
        Train_or_Eval(this.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = this.model[0](data)
        A = this.model[1](features)
        output, attention_maps = this.model[2](features[-1], A, target, length)
        # computing accuracy and loss
        this.train_acc_counter.add_iter(output, length.long(), length, label)
        loss = this.criterion_CE(output, label_flatten)
        this.loss_counter.add_iter(loss)
        # update network
        Zero_Grad(this.model)
        loss.backward()
