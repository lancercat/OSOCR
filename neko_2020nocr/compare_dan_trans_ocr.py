import torch.nn.functional

from compare_dans import diff
from neko_sdk.visualization import colors as cp
# diffweird();
colors=list(cp.colors.colors20.values());
# diffchr();

# diffchsA();
#
methods=[
    # ["basic_chsHSsc_CE_alter",colors[1],"solid"],
    # ["basic_chsHS_CE_alter", colors[2], "solid"],
    # ["basic_chsHSwt_CE_alter", colors[12], "solid"],

    # ["basic_chsHSext_CE_alter", colors[2], "solid"],

    # ["basic_chsHS_C_alter", colors[4], "solid"],
    # ["basic_chsHSKCQA_CE_alter", colors[5], "solid"],
    # ["basic_chsHSK_CE_alter", colors[6], "solid"],
    # ["basici_chsHS_CE_alter", colors[19], "solid"],
    # ["basict_chsHS_CE_alter", colors[18], "solid"],
    # ["basict_chsHSCQA_CE_gen1_alter", colors[18], "solid"],
    ["basict_chsHSCQA_CE_alter", colors[18], "dash"],
    # ["basict_chsHSCQA_C_alter", colors[17], "dash"],
    #
    ["rawt_chsHSCQA_C_alter", colors[6], "dash"],
    ["rawt_chsHSCQA_CE_alter", colors[6], "dot"],

    # ["basictt_chsHSCQA_CE_alter", colors[19], "dash"],

    # ["basict_chsHSCQA_CE_alter_gen1", colors[18], "dot"],

    # ["basictn_chsHSCQA_CE_alter", colors[17], "dash"],
    #
    # ["basic_chsHSCQA_CE_alter", colors[1], "dash"],
    # ["basic2_chsHSCQA_CE_alter", colors[3], "dash"],
    # ["basic2tl_chsHSCQA_CE_alter", colors[3], "solid"],

    # ["basict_chsHSKCQA_CE_alter", colors[16], "solid"],
    # ["basicht_chsHSCQA_CE_alter", colors[14], "solid"],
    #
    # ["basicn_chsHS_CE_alter", colors[15], "solid"],
    # ["basicnt_chsHSKCQA_CE_alter", colors[17], "solid"],
    # ["conventional_chsHS_CE_alter", colors[3], "solid"],

    # ["basic_chsHS_C_alter", colors[4], "dash"],
    # ["basic_chsHS_C_alter", colors[5], "solid"],
    # ["basic_chsHS_CEF_alter", colors[6], "solid"],

]

diff("/home/lasercat/resultcollpami/trans_ocr/",methods,23686);
