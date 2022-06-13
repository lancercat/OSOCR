import xmltodict;
import cv2;
with open("/home/lasercat/Downloads/svt1/test.xml") as fd:
    doc = xmltodict.parse(fd.read())
tot=0;
dst="/run/media/lasercat/ssddata/svtdicted/"
for im in doc["tagset"]["image"]:
    img=cv2.imread("/home/lasercat/Downloads/svt1/"+im['imageName']);
    lex=im['lex'];
    try:
        for box in im['taggedRectangles']['taggedRectangle']:
            x,y,h,w=int(box["@x"]),int(box["@y"]),int(box["@height"]),int(box["@width"]);
            t = int(y)
            if(t<0):
                t=0;
            b = int(y + h)
            l = int(x)
            r = int(x + w);
            crop=img[t:b,l:r,:];
            try:
                cv2.imwrite(dst+str(tot)+".jpg",crop);
            except:
                print("???")
            with open(dst+str(tot)+".lex","w+") as ofp:
                ofp.writelines(lex);
            with open(dst + str(tot) + ".gt", "w+") as ofp:
                ofp.writelines(box["tag"]);
            tot+=1
    except:
        box = im['taggedRectangles']['taggedRectangle'];
        x, y, h, w = int(box["@x"]), int(box["@y"]), int(box["@height"]), int(box["@width"]);
        t = int(y)
        b = int(y + h)
        l = int(x)
        r = int(x + w);
        crop = img[t:b, l:r, :];
        cv2.imwrite(dst + str(tot) + ".jpg", crop);
        with open(dst + str(tot) + ".lex", "w+") as ofp:
            ofp.writelines(lex);
        with open(dst + str(tot) + ".gt", "w+") as ofp:
            ofp.writelines(box["tag"]);
        tot += 1
pass;

