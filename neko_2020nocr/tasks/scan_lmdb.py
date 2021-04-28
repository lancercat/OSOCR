from neko_sdk.lmdb_wrappers.ocr_lmdb_reader import neko_ocr_lmdb_mgmt
from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper
import numpy
import cv2

def view_db(root,each=20):
    db = neko_ocr_lmdb_mgmt(root, False, 1000);
    cv2.namedWindow("show",0);
    for i in range(len(db)):
        im, t,_ = db.getitem_kv(i, ["image"], ["label"],[]);
        if(i%each==0):
            open_cv_image = numpy.array(im[0])
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            cv2.imshow("show",open_cv_image);
            print(t[0]);
            ch=cv2.waitKey(300)
            if(ch==112):
                cv2.waitKey(0);
if __name__ == '__main__':
    view_db("/media/lasercat/backup/deployedlmdbs/lsvtdb_seen/",20)