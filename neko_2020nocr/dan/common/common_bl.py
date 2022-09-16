import torch;
#------------------------
#---------------------network
def load_network(cfgs):
    model_fe = cfgs.net_cfgs['FE'](**cfgs.net_cfgs['FE_args'])
    cfgs.net_cfgs['CAM_args']['scales'] = model_fe.Iwantshapes()
    model_cam = cfgs.net_cfgs['CAM'](**cfgs.net_cfgs['CAM_args'])
    model_dtd = cfgs.net_cfgs['DTD'](**cfgs.net_cfgs['DTD_args'])

    if cfgs.net_cfgs['init_state_dict_fe'] != None:
        model_fe.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_fe']))
    if cfgs.net_cfgs['init_state_dict_cam'] != None:
        model_cam.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_cam']))
    if cfgs.net_cfgs['init_state_dict_dtd'] != None:
        model_dtd.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_dtd']))

    model_fe.cuda()
    model_cam.cuda()
    model_dtd.cuda()
    return (model_fe, model_cam, model_dtd);
