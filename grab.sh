mkdir good;
for i in $(cat good.txt  | awk '{print $1}');do cp /run/media/lasercat/a503e9e0-732c-4244-88fb-663e2a3edbee/mini/miniImageNet_ims/images/$i good/; done;

mkdir bad;
for i in $(cat bad.txt  | awk '{print $1}');do cp /run/media/lasercat/a503e9e0-732c-4244-88fb-663e2a3edbee/mini/miniImageNet_ims/images/$i bad/; done;

mkdir middle;
for i in $(cat middle.txt  | awk '{print $1}');do cp /run/media/lasercat/a503e9e0-732c-4244-88fb-663e2a3edbee/mini/miniImageNet_ims/images/$i middle; done;
