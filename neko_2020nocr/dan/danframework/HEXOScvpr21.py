#------------------------
import copy

from neko_2020nocr.dan.utils import *
from neko_2020nocr.dan.common.common import Zero_Grad,Train_or_Eval, flatten_label
from neko_2020nocr.dan.visdan import visdan;

from neko_2020nocr.dan.danframework.HXOS import HXOSC, neko_cos_loss2;
from neko_sdk.ocr_modules.prototypers.neko_prototyper_core import neko_prototype_core_basic_shared;

from neko_sdk.ocr_modules.trainable_losses.neko_url import neko_unknown_ranking_loss;
from neko_sdk.AOF.neko_lens import vis_lenses;


# HSOS, HDOS, HDOSCS
# the main feature is that the DPE returns embeddings for characters.
import torch;
import time
class HXXOS2C(HXOSC):
    def label_weight(this,shape,label):
        weight=torch.zeros(shape).to(label.device)+0.1
        weight=torch.scatter_add(weight,0,label,torch.ones_like(label).float());
        weight=1./weight;
        weight[-1]/=200;
        return weight;

    def setuploss(this):
        this.criterion_CE = nn.CrossEntropyLoss();
        # this.aceloss=
        this.url=neko_unknown_ranking_loss();
        this.cosloss=neko_cos_loss2().cuda();
        this.wcls = this.cfgs.loss_weight["wcls"];
        this.wsim = this.cfgs.loss_weight["wsim"];
        this.wemb=  this.cfgs.loss_weight["wemb"];
        this.wmar = this.cfgs.loss_weight["wmar"];
        # this.wace=this.cfgs.loss_weight["wace"];

    def fpbp(this, data, label,cased=None):
        proto, semb, plabel, tdict = this.mk_proto(label);
        target = this.model[3].encode(proto, plabel, tdict, label);

        Train_or_Eval(this.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        Zero_Grad(this.model)
        features = this.model[0](data)
        A = this.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        outcls, outcos = this.model[2](features[-1], proto,semb , plabel, A, target, length)
        choutput, prdt_prob, = this.model[3].decode(outcls, length, proto, plabel, tdict);

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        this.train_acc_counter.add_iter(choutput, length, tarswunk)
        # 0.14 means 50k classes with 512d configuration.
        proto_loss=torch.nn.functional.relu(proto[1:].matmul(proto[1:].T)-0.14).mean();
        w=torch.ones_like(torch.ones(outcls.shape[-1])).to(proto.device).float();
        w[-1]=0.1;
        clsloss = torch.nn.functional.cross_entropy(outcls, label_flatten,w);
        cos_loss= this.cosloss(outcos,label_flatten);
        margin_loss = this.url.forward(outcls, label_flatten, 0.5)
        # ace_loss=this.aceloss(outcls,label_flatten)
        loss=cos_loss*this.wsim+clsloss*this.wcls+margin_loss*this.wmar+this.wemb*proto_loss;

        terms={
            "total": loss.detach().item(),
            "margin": margin_loss.detach().item(),
            "main": clsloss.detach().item(),
            "sim":cos_loss.detach().item(),
            "emb":proto_loss.detach().item(),
        }
        this.loss_counter.add_iter(loss,terms)
        # update network
        loss.backward()
    def test_im(this,data,args):
        model=this.model;
        proto, semb, plabel, tdict=args;
        data = data.cuda()
        features = model[0](data)
        A = model[1](features)
        # A0=A.detach().clone();
        output, out_length, A = model[2](features[-1], proto, semb, plabel, A, None, 30, True)
        # A=A.max(dim=2)[0];
        choutput, prdt_prob = model[3].decode(output, out_length, proto, plabel, tdict);
        return choutput;

        # A.detach().reshape(A.shape[0], A.shape[1], A.shape[2], A.shape[3]).sum(2)

    def testready(this,):
        Train_or_Eval(this.model, 'Eval')
        temeta = None;
        try:
            if ("temeta" in this.cfgs.dataset_cfgs):
                temeta = torch.load(this.cfgs.dataset_cfgs["temeta"]);
        except:
            print("wild running")
        proto, semb, plabel, tdict = this.model[3].dump_all();
        return proto, semb, plabel, tdict;

    def test(this,test_loader, model, tools,miter=1000,debug=False,dbgpath=None):
        Train_or_Eval(model, 'Eval')
        # model[3].train()
        temeta =None;
        if("temeta" in this.cfgs.dataset_cfgs):
            temeta=torch.load(this.cfgs.dataset_cfgs["temeta"]);
        tmetastart=time.time();
        if(temeta is None):
            proto,semb, plabel, tdict = model[3].dump_all();
        else:
            teng=neko_prototype_core_basic_shared(temeta,model[3].dwcore);
            teng.cuda()
            proto, plabel, tdict=teng.dump_all();
            semb=None;
        tmetaend = time.time();

        fwdstart = time.time()
        idi=0;
        all=0
        # visualizer=None;
        if dbgpath is not None:
            visualizer=visdan(dbgpath);
        # cfm=neko_confusion_matrix();
        for sample_batched in test_loader:
            if idi>miter:
                break;
            idi+=1;
            data = sample_batched['image']
            label = sample_batched['label'];
            target = model[3].encode(proto, plabel, tdict, label)

            all+=data.shape[0];
            data = data.cuda()
            target = target
            label_flatten, length = tools[1](target)
            # target, label_flatten = target.cuda(), label_flatten.cuda()

            features = model[0](data)
            A = model[1](features)
            #
            # # A0=A.detach().clone();
            output, out_length,A = model[2](features[-1], proto,semb, plabel, A, None, length, True)
            # # A=A.max(dim=2)[0];
            choutput, prdt_prob= model[3].decode(output, out_length, proto, plabel, tdict);
            rchoutput=[[i] for  i in choutput];
            ulabels=[];
            if(len(tools)==3):
                for l in label:
                    ulabels.append("".join([c if c in tdict else "⑨" for c in l]));
                hpwu=[b[0] for b in rchoutput]
                tools[2].add_iter(hpwu, out_length, ulabels,debug)
            else:
                tools[0].add_iter(choutput, out_length, label,debug)
            # A.detach().reshape(A.shape[0], A.shape[1], A.shape[2], A.shape[3]).sum(2)
            # for i in range(len(choutput)):
            #     cfm.addpairquickandugly(choutput[i],label[i]);
            if(dbgpath):
                A = model[1](features)
                features,grid = model[0](data,True)
                try:
                    if (len(grid) > 0):
                        data_=vis_lenses(data,grid)[1];
                        if (visualizer is not None):
                            # visualizer.addbatch(data_, A, label,choutput)
                            visualizer.add_image([data, data_], label,rchoutput,choutput, ["before", "after"])
                except:
                    pass;
                    # print("meow")
                #
                # fgrid=F.interpolate(grid[0],data.shape[-2:],mode="bilinear");
                # data=F.grid_sample(data,fgrid.permute(0,2,3,1));

            # if(visualizer is not None):
            #     visualizer.add_image([data,data_],None,None,["before","after"])
            #     visualizer.addbatch(data, A, label,choutput)
        # if(dbgpath):
        #     try:
        #         cfm.save_matrix(os.path.join())
        #     except:
        #         pass;
        fwdend=time.time()
        print((fwdend-fwdstart)/all,all)
        if (len(tools) == 3):
            try:
                tools[2].show();
            except:
                pass;
        else:
            tools[0].show();
        print(all);
        Train_or_Eval(model, 'Train')


class HDOS2C(HXXOS2C):
    def mk_proto(this,label):
        return this.model[3].sample_tr(label)

class HSOS2C(HXXOS2C):
    def mk_proto(this,label):
        return this.model[3].dump_all()
class HDOS2C_Debug(HDOS2C):
    def tr_iter(this, nEpoch, batch_idx, sample_batched, total_iters):
        data = sample_batched['image'].permute([0,2,3,1])
        label = sample_batched['label']
        ims=[];
        cv2.namedWindow("image",0);
        for i in range(len(data)):
            im=(data[i].cpu().detach()*255).numpy().astype(np.uint8);
            print(label[i])
            cv2.imshow("image",im);
            cv2.waitKey(0);

