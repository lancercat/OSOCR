cls_only=["C",{
    "wcls":1,
    "wace": 0,
    "wsim": 0,
    "wemb": 0,
    "wmar": 0,
}];
cls_emb=["CE",{
    "wcls":1,
    "wace": 0,
    "wsim": 0,
    "wemb": 0.3,
    "wmar": 0,
}];
#
# this.wcls = this.cfgs["loss_weight"]["lcls"];
# this.wsim = this.cfgs["loss_weight"]["lsim"];
# this.wemb = this.cfgs["loss_weight"]["lemb"];
# this.wmar = this.cfgs["loss_weight"]["lmar"];
sim_emb=["SE",{
    "wcls":0,
    "wace": 0,
    "wsim": 1,
    "wemb": 0.1,
    "wmar": 0,
}];
# naive a: set rejection as don't care and later find a threshold
dcrej_cls_emb=["DCE",{
    "wcls":1,
    "wace": 0,
    "wsim": 0,
    "wemb": 0.3,
    "wmar": 0,
}];
# naive b: set a fix threshold for unk
frej_cls_emb=["FCE",{
    "wcls":1,
    "wace": 0,
    "wsim": 0,
    "wemb": 0.3,
    "wmar": 0,
}];
