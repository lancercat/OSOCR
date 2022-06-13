from scipy import io;
import cv2;

root="/home/lasercat/Downloads/iiit5k/IIIT5K/"
droot="/run/media/lasercat/ssddata/iiit5kdicted/"
a=io.loadmat(root+"testdata.mat")

def l2t(lexi):
    s=lexi[0][0];
    for i in lexi[1:]:
        s+=",";
        s+=i[0];
    return s;

for i in range(3000):
    m=a['testdata'][0][i]
    sl=m['smallLexi'];
    ml=m["mediumLexi"];
    ss=l2t(sl[0]);
    ms=l2t(ml[0]);
    ispath=root+m["ImgName"][0];
    im=cv2.imread(ispath)

    cv2.imwrite(droot + str(i) + ".jpg", im);
    with open(droot + str(i) + ".lexs", "w+") as ofp:
        ofp.writelines(ss);
    with open(droot + str(i) + ".lexm", "w+") as ofp:
        ofp.writelines(ms);
    with open(droot + str(i) + ".gt", "w+") as ofp:
        ofp.writelines(m["GroundTruth"][0]);

pass;
