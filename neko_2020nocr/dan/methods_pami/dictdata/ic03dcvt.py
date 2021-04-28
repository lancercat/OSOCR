import xmltodict;
import cv2;
with open("/home/lasercat/Downloads/ic03/words.xml",encoding="ISO-8859-1") as fd:
    doc = xmltodict.parse(fd.read())
tot=0;
dst="/run/media/lasercat/ssddata/ic03dicted/"
lexs=[]
for im in doc["tagset"]["image"]:
    img=cv2.imread("/home/lasercat/Downloads/ic03/"+im['imageName']);
    # lex=im['lex'];
    try:
        for box in im['taggedRectangles']['taggedRectangle']:
            x,y,h,w=float(box["@x"]),float(box["@y"]),float(box["@height"]),float(box["@width"]);
            t = int(y)
            b = int(y + h)
            l = max(int(x),0)
            r = int(x + w);
            crop=img[t:b,l:r,:];
            cv2.imwrite(dst+str(tot)+".jpg",crop);

            # with open(dst+str(tot)+".lex","w+") as ofp:
            #     ofp.writelines(lex);
            with open(dst + str(tot) + ".gt", "w+") as ofp:
                ofp.writelines(box["tag"]);
            lexs.append(box["tag"]);
            tot+=1
    except:
        box = im['taggedRectangles']['taggedRectangle'];
        x, y, h, w = float(box["@x"]), float(box["@y"]), float(box["@height"]), float(box["@width"]);
        t = int(y)
        b = int(y + h)
        l = int(x)
        r = int(x + w);
        crop = img[t:b, l:r, :];
        cv2.imwrite(dst + str(tot) + ".jpg", crop);
        with open(dst + str(tot) + ".gt", "w+") as ofp:
            ofp.writelines(box["tag"]);
        lexs.append(box["tag"]);
        tot += 1
ls=lexs[0];
for i in lexs[1:]:
    ls+=",";
    ls+=i;

for i in range(len(lexs)):
    with open(dst+str(i)+".lex","w+") as ofp:
        ofp.writelines(ls);

pass;

