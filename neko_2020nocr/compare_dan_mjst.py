from compare_dans import diff
from neko_sdk.visualization import colors as cp
colors=list(cp.colors.colors20.values())
methods=[
    # ["basic_mjst_CES_3ca",colors[0],"solid"],
    # ["basic_mjst_CE_3ca", colors[1], "dash"],
    # ["rawS_mjst_C_alter", colors[2], "solid"],
    # ["basic_mjst_CE_alter", colors[3], "solid"],
    # ["basict_mjstcqa_CE_alter", colors[8], "solid"],
    # ["basict_mjst_CE_alter",colors[14],"solid"],

    # xl with gen3 aug. removed clipping,
    ["rawt_mjstcqa_CE_alter", colors[7], "dot"],
    ["rawt_mjstcqa_C_alter", colors[6], "dot"],
    ["basict_mjstcqa_CE_gen3_alter", colors[8], "dot"],
    ["basic_mjstcqaSemi_CE_alter", colors[9], "dot"],

    # ["basict_mjstcqa_C_alter", colors[9], "dot"],

    # xxl candidates
    # ["basicT_mjstcqa_CE_alter", colors[15], "solid"],
    # ["basicTt_mjstcqa_CE_alter_gen2", colors[13], "solid"],
    # ["basictt_mjstcqa_CE_alter", colors[17], "solid"],
    # ["basic2tl_mjstcqa_CE_alter", colors[18], "dash"],
    #
    # ["basics_mjst_CE_alter", colors[10], "solid"],
    # ["basic_mjstcqa_CE_alter", colors[13], "solid"],
    #
    # ["basict_mjstcqaf_CE_alter", colors[8], "dot"],
    # ["basict_mjstcqa_CE_alter_gen2", colors[8], "dash"],
    # ["basicm_mjstcqa_CE_alter", colors[11], "solid"],
    # ["basic2_mjst_CE_alter", colors[12], "solid"],

    # ["basictms_mjstcqa_CE_alter", colors[17], "dash"],

    #
    # ["basicht_mjstcqa_CE_alter", colors[9], "solid"],

    # ["basict_mjstcqa_CE_alter_gen2", colors[9], "dash"],


    # ["basic_mjstcqaSemi_CE_alter", colors[16], "solid"],
    # ["basictn_mjstcqa_CE_alter", colors[10], "dash"],

    # ["raw_mjst_C_alter", colors[4], "dash"],
    # ["raw_mjst_CE_alter", colors[9], "dash"],
    # ["raw_mjstsc_CE_alter", colors[6], "dash"],
    # ["basict_mjstcqaSemi_CE_alter", colors[11], "solid"],
    # ["basic_mjstcqa_CE_alter", colors[7], "solid"],
    # ["basic_mjstcqaXL_CE_alter", colors[12], "solid"],
    # ["basici_mjst_CE_alter", colors[15], "solid"],
]

diff("/home/lasercat/resultcollpami/mjst/",methods,239703);
