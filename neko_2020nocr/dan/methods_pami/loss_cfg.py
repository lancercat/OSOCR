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
cls_emb_ace=["CEA",{
    "wcls":1,
    "wace":0.3,
    "wsim": 0,
    "wemb": 0.3,
    "wmar": 0,
}];
cls_emb_sim=["CES",{
    "wcls":1,
    "wace": 0,
    "wsim": 1,
    "wemb": 0.1,
    "wmar": 0,
}];
cls_emb_sim_alter=["CES_alter",{
    "wcls":1,
    "wace": 0,
    "wsim": 1,
    "wemb": 0.1,
    "wmar": 0,
}];
#
# this.wcls = this.cfgs["loss_weight"]["lcls"];
# this.wsim = this.cfgs["loss_weight"]["lsim"];
# this.wemb = this.cfgs["loss_weight"]["lemb"];
# this.wmar = this.cfgs["loss_weight"]["lmar"];