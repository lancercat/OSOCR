from neko_2020nocr.dan.xtract_results import xtra_file;
from neko_sdk.draw_curve import draw;
import numpy as np;
import os;


def compare(fns,names,colors,ltypes,epsize):
    ranges=set();
    teds=[];
    nteds=[];
    for i in range(len(fns)):
        ted=xtra_file(fns[i],epsize).acr_dict["test"];
        if(i==0):
            ranges=set(ted.keys());
        else:
            ranges=ranges.intersection(set(ted.keys()));
        teds.append(ted);
    ranges=sorted(list(ranges));

    for tid in range(len(teds)):
        nt_x=[];
        nt_y=[];
        nt={};
        for i in ranges:
            nt_x.append(i);
            nt_y.append(teds[tid][i])
        nt["x"]=nt_x;
        nt["y"]=nt_y;
        nt["n"]=names[tid];
        nt["c"]=colors[tid];
        nt["l"]=ltypes[tid];
        nteds.append(nt);

    ys=[];
    for nt in nteds:
        ys+=nt["y"];
    xrange=np.min(ranges),np.max(ranges);
    yrange=np.min(ys),np.max(ys);
    return xrange,yrange,nteds;


def diff(root,methods,epsz):
    d=differ();
    # d.register_method("/home/lasercat/cifsl/base.log", "base", "w","dash");
    for method in methods:
        d.register_method(os.path.join(root,method[0])+".log", method[0], method[1],method[2]);
    d.comp(epsz);


class differ:
    def __init__(this):
        this.logs = [];
        this.ids = [];
        this.colors = [];
        this.ltypes=[];
    def register_method(this, log, id, color, ltype):
        this.logs.append(log);
        this.ids.append(id);
        this.colors.append(color);
        this.ltypes.append(ltype);

    def comp(this,eit=239703):
        xrange, yrange, nteds=compare(this.logs,this.ids,this.colors,this.ltypes,eit);
        draw(xrange, yrange, nteds, 00);

def diffchsA():
    d=differ();
    # d.register_method("/home/lasercat/chsres/blchsA.log","base","w","dash");
    # d.register_method("/home/lasercat/chsres/blcfchsA.log", "blcfchsA", "w", "dash");
    # d.register_method("/home/lasercat/chsres/cfv2cco.log", "cfv2cco", "c", "dash");

    # d.register_method("/home/lasercat/chsres/cfv2Acco.log", "cfv2Acco", "b", "dash");
    # d.register_method("/home/lasercat/chsres/baseline_ccoHK.log", "blccoHK", "w","dot");


    # d.register_method("/home/lasercat/chsres/cfv2.log", "cfv2", "g", "solid");
    # d.register_method("/home/lasercat/chsres/cfv2Rcco.log", "cfv2Rcco", "m", "solid");
    # d.register_method("/home/lasercat/chsres/cfv2AccochsC.log","cfv2AccochsC","r","dash");
    # d.register_method("/home/lasercat/chsres/cfv2AccochsCH.log","cfv2AccochsCH","r","dot");
    # d.register_method("/home/lasercat/chsres/baseline_ccoRHK.log", "blccoRHK", "m","solid");
    # d.register_method("/home/lasercat/chsres/basic_chs_C.log","basic_chs_C","b","dash")
    d.register_method("/home/lasercat/chsres/basic_chs_Cv2.log","basic_chs_Cv2","b","dot")

    # d.register_method("/home/lasercat/chsres/conv_chs_CES.log","conv_chs_CES","g","dash")
    d.register_method("/home/lasercat/chsres/basic_chs_CEv2.log","basic_chs_CEv2","c","dot")

    # d.register_method("/home/lasercat/chsres/basic_chsHS_C.log", "basic_chsHS_C", "f93971", "dash")
    #
    #
    # d.register_method("/home/lasercat/chsres/basic_chs_CE.log","basic_chs_CE","c","dash")
    # d.register_method("/home/lasercat/chsres/basic_chs_CES.log", "basic_chs_CES", "g", "dot")
    d.register_method("/home/lasercat/chsres/basic_chs_CESv2.log","basic_chs_CESv2","g","dot")
    d.register_method("/home/lasercat/chsres/basic2_chs_CESv2.log","basic2_chs_CESv2","r","dot")

    # d.register_method("/home/lasercat/chsres/ccoscfchsA.log", "ccoscfchsA", "y", "solid");
    #d.register_method("/home/lasercat/chsres/dosdancfchs.log", "doscf", "f93971");
    # d.register_method("/home/lasercat/chres/dosdancfr_lsvt.log", "doscfr", "c");
    d.comp(161886);

# diffuncased();
def diffweird():
    d=differ()
    d.register_method("/home/lasercat/resultcoll/weird/baseline_lcf_ARTHS.log", "baseline_lcf_ARTHS", "k","dash");
    d.register_method("/home/lasercat/resultcoll/weird/blsc_lcf_ARTHS.log", "blsc_lcf_ARTHS", "k","solid");
    d.register_method("/home/lasercat/resultcoll/weird/mob3_lcf_ARTHS.log", "mob3_lcf_ARTHS", "r", "solid");
    d.register_method("/home/lasercat/resultcoll/weird/eff_lcf_ARTHS.log", "eff_lcf_ARTHS", "f854f1", "solid");

    d.register_method("/home/lasercat/resultcoll/weird/pfsc_lcf_ARTHS.log", "pfsc_lcf_ARTHS", "g", "solid");
    d.register_method("/home/lasercat/resultcoll/weird/htasc_lcf_ARTHS.log", "htasc_lcf_ARTHS", "c", "solid");
    d.register_method("/home/lasercat/resultcoll/weird/gtsc_lcf_ARTHS.log", "gtsc_lcf_ARTHS", "m", "solid");
    d.register_method("/home/lasercat/resultcoll/weird/npgrtsc_lcf_ARTHS.log", "npgrtsc_lcf_ARTHS", "893869", "solid");
    d.register_method("/home/lasercat/resultcoll/weird/npswarmsc_lcf_ARTHS.log", "npswarmsc_lcf_ARTHS", "f83963", "solid");
    # d.register_method("/home/lasercat/resultcoll/weird/npgrtswarmsc_lcf_ARTHS.log", "npgrtswarmsc_lcf_ARTHS", "483963", "solid");
    d.register_method("/home/lasercat/resultcoll/weird/npsc_lcf_ARTHS.log", "npsc_lcf_ARTHS", "66d789", "solid");
    d.register_method("/home/lasercat/resultcoll/weird/npgsc_lcf_ARTHS.log", "npgsc_lcf_ARTHS", "6789f2", "solid");


    d.comp(22236);

def diffmjst_ijcai():

    d=differ()
    d.register_method("/home/lasercat/resultcoll/mjst/bl_lcf.log", "bl_lcf", "k","solid");
    # d.register_method("/home/lasercat/resultcoll/mjst/pf_lcf.log", "pf_lcf", "g","solid");
    # d.register_method("/home/lasercat/resultcoll/mjst/npf_lcf.log", "npf_lcf", "g","dash");
    # d.register_method("/home/lasercat/resultcoll/mjst/mnpf_lcf.log", "mnpf_lcf", "489d23","dash");

    # d.register_method("/home/lasercat/resultcoll/mjst/gta_lcf.log", "gta_lcf", "b","solid");
    # d.register_method("/home/lasercat/resultcoll/mjst/hta_lcf.log", "hta_lcf", "g","dash");
    # d.register_method("/home/lasercat/resultcoll/mjst/bnet_lcf.log", "bnet_lcf", "m","solid");

    d.comp();
#
def diffmjstvlxl_ijcai():

    d=differ()
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vlbaseline.log", "vlbaseline", "k","solid");
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/trio.log", "trio", "b","solid");
    d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vloc-H.log", "vloc-H", "g","solid");
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vloc-H_FAL.log", "vloc-H_FAL", "g","dash");
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/Svloc-H.log", "Svloc-H", "45a3a9", "dash");
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/SDvloc-H.log", "SDvloc-H", "45a3a9", "solid");

    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vloc-O.log", "vloc-O", "45a3a9","solid");
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/pnsv3H_bl.log", "pnsv3H_bl", "a387e9","solid");
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/pnsv3H_npmr.log", "pnsv3H_npmr", "a3f7e9","solid");

    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vloc-H-ES.log", "vloc-H-ES", "a387e9","solid");
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vloc-H-D.log", "vloc-H-D", "8493f6", "solid");

    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vlac.log", "vlac", "m","solid");
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vloc.log", "vloc", "r","solid");
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vlxt.log", "vlxt", "k", "solid");

    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vlhaoc.log", "vlhaoc", "m","solid");
    d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vlhaoL.log", "vlhaoL", "b","solid");
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/vlhaoc2.log", "vlhaoc2", "c", "solid");

    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/Dual-S.log", "Dual-S", "3548a4","dash");
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/dnet.log", "dnet", "a75687", "dash");
    #
    # d.register_method("/home/lasercat/resultcoll/mjst_vlxl/dnet3.log", "dnet3", "75a687", "dash");
    d.comp();

# diffweird();

# diffchr();

# diffchsA();
#
