import getpass;
import os;
def find_data_root():
    username = getpass.getuser()
    if(username!="prir1005"):
        print("own PC?")
        return os.path.join("/home",username,"ssddata");
    else:
        print("lab pc!")
        return "/home/prir1005/cat/recdata";
