import os;
import glob;
import shutil;
def wrong(src,dst,cnt):
    shutil.rmtree(dst,True);
    os.makedirs(dst);
    for i in range(cnt):
        try:
            res=os.path.join(src,str(i)+"_res.txt");
            imo = os.path.join(src, str(i) + "before_img.jpg");
            imp = os.path.join(src, str(i) + "after_img.jpg");

            resd=os.path.join(dst,str(i)+"_res.txt");
            imdo = os.path.join(dst, str(i) + "before_img.jpg");
            imdp = os.path.join(dst, str(i) + "after_img.jpg");
            with open(res,"r") as fp:
                gt,pred=[i.strip() for i in fp];
            if(gt!=pred):
                print(gt,"->-",pred);
                shutil.copy(res,resd);
                shutil.copy(imo, imdo);
                shutil.copy(imp, imdp);
        except:
            pass;

if __name__ == '__main__':
    wrong("/home/lasercat/ssddata/mjst/SVTP","/home/lasercat/ssddata/mjst/SVTPerr",1500);
