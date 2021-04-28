import torch;
class neko_confusion_matrix:
    def __init__(this,charset=None):
        if(charset is not None):
            this.charset=charset;
        else:
            this.charset=set();

        this.edict={};
        this.iddict={};
        total=0;
        for k in this.charset:
            this.edict[k]={};
            this.iddict[k]=total;
            total+=1;
    def addpairquickandugly(this,pred,gt):
        minlen=min(len(pred),len(gt));
        for i in range(minlen):
            pc=pred[i];
            gc=gt[i];
            if pc not in this.charset:
                this.charset.add(pc);
                this.edict[pc]={};
                this.iddict[pc]=len(this.iddict);
            if gc not in this.charset:
                this.charset.add(gc);
                this.edict[gc]={};
                this.iddict[gc]=len(this.iddict);

            if(pc not in this.edict[gc]):
                this.edict[gc][pc]=0;
            this.edict[gc][pc]+=1;

    def save_matrix(this,dst):
        torch.save(this.edict,dst)