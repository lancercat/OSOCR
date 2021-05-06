# OSOCR
## Trained models
https://drive.google.com/drive/folders/1WqpL1EAg2A5LXV8V7I1wK6XOMSGAR7Z4?usp=sharing
## Evaluation LMDBs
https://drive.google.com/drive/folders/1g6XBx2xbD7T5PgGlnXWgFGu00oPRQSzX?usp=sharing

### Naming rules:
\[model\]\_\[trainingset\]\[inputformat\]\_\[epoch\]\_\[loss\]

#### Models
basic: Regular model

basict: Large model

conventional: Conventional close-set recognition model (With out the lable to prototype learning framework).

#### Trainingsets
ctwch: Characters from the CTW dataset. These models are used in zero-shot character recognition tasks. 

mjst: Synthdataset for English.   These models are used in close set text recognition tasks. 

chsHS: Simplified Chinese. These models are used in open set text recognition tasks. 

#### Input Formats
None: Normal input

cqa: Colored image with augmentation adapted from SRN by Qin. (C for color, Q for Qin, A for augmentation).

#### Losses
C: Cross Entropy only

CE: With L_{emb}


## Manual



### Dependencies (Ubuntu 20.04)
PyTorch>1.4 with CUDA support.  

Other dependencies can be installed via the following commands.

```
pip3 install torchvision lmdb opencv lmdb scikit-learn torch_scatter regex editdistance

sudo apt install python3-pip nvidia-cuda-toolkit python3-opencv
```

### Paths
The default paths:
    
    CODEROOT: /home/yourusername/cat/neko_wcki
    
    DATAROOT: /home/yourusername/ssddata
`$DATAROOT` is defined in `$CODEROOT/neko_sdk/root.py` 

`$CODEROOT` is where code resides, should be okay anywhere, in theroy. 



### Evaluation with the prepared LMDBs
1. Download the trained models and put them in `$CODEROOT/neko_2020nocr/dan/models`

2. Download the evaluation LMDBs and put them in `$DATAROOT`

3. Setup python path:
    ```export PYTHONPATH=${CODEROOT}```

4. Change neko_sdk.py accordingly if you do not use default ${DATAROOT}

5. pick an experiment in `neko_2020nocr/dan/methods_pami/`, for example basict_mjstcqa_CE_alter

6. Double check the `cfgs_scene.py` file in the experiment dir, make sure you have the correct number of epochs corresponding to your checkpoints.

7. Evaluate with `python3 test.py`. 



### Preparing dataset for training and evauation from raw materials.
1. Use https://github.com/lancercat/OSOCR-data to unzip the datasets.

2. ... TBA.

## About
This is the code repo of a currently under reviewing paper. 

And feel free to contact the author should you encounter any problems using the code.

If you want models for other experiments in the paper, please email me the specific experiment(s). 


