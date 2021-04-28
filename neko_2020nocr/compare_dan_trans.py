from compare_dans import diff
from neko_sdk.visualization import colors as cp
# diffweird();
colors=list(cp.colors.colors20.values());
# diffchr();

# diffchsA();
#
methods=[
    # ["basic_chsHSsc_CE_alter",colors[1],"solid"],
    # ["basic_chsHS_CEF_alter", colors[2], "solid"],
    # ["basic_chsHS_C_alter", colors[4], "dash"],
    # ["basic_chsHS_C_alter", colors[5], "solid"],
    # ["basic_chsHS_CEF_alter", colors[6], "solid"],

]

diff("/home/lasercat/resultcollpami/trans/",methods,23686);
