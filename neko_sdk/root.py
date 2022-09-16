import getpass;
import os;
def find_data_root():
    username = getpass.getuser()
    if(username!="prir1005"):
        print("own PC?")
        return os.path.join("/home",username,"ssddata");
    else:
        print("lab pc!")
        return "/home/prir1005/cat/recdatassd/";
def find_model_root(root_override=None):
    if(root_override is not None):
        return root_override;
    username = getpass.getuser()
    if (username != "prir1005"):
        print("own PC?")
        return os.path.join("/run/media/", username, "ssddata/pamidump/trained_models/models/scene/prextra2/");
    else:
        print("lab pc!")
        return "/home/prir1005/cat/recdata/";
