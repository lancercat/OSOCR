# OSOCR

[Update] 
1. The data repo is now fully set up at 
https://github.com/lancercat/OSOCR-data

2. The training scripts are added.

3. The manual manul.pdf is updated with training instructions, and a few manul photos.

4. GLHF and Mewrry Xmeows

If you have any concerns, please email us. 

Cat
24-Dec-2022

-----------History

[Update]
The evaluation code and core implementation for revision two is updated. The extra models and training code will be released upon acceptance. 




[Update]
The paper is undergoing a major revision and more experiments are added (synchronizing, not fully done yet).

[Update]
The paper is undergoing a minor revision and more experiments will be added.


We will soon perform a full test of the code on a fresh installation of Manjaro and upload a manual to guide you through the evaluation.


## Trained models
https://drive.google.com/drive/folders/1WqpL1EAg2A5LXV8V7I1wK6XOMSGAR7Z4?usp=sharing

Most models including ablative study models are now uploaded, and 
training dataset and scripts will be uploaded upon acceptance.


## Evaluation LMDBs
https://www.kaggle.com/lancercat/osocr-test

## Training LMDBs
https://www.kaggle.com/vsdf2898kaggle/osocrtraining


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
### Evaluateion
Please refer to manul.pdf (mostly there, we are still working on it.)

As we no more have a Ubuntu testing bed, we are discontinuing support for Ubuntu systems. 

The framework is still likely to work, just we cannot test for sure.

If your are setting up on old enviornments, please refer to AeroX's guide in Issue #6 of the VSDF repo:

https://github.com/lancercat/VSDF/issues/6

If you have any problems, please open an issue.


### Preparing dataset for training and evauation from raw materials.
Please refer to the OSOCR-data repo

https://github.com/lancercat/OSOCR-data/manul.pdf


## About

And feel free to contact me(lasercat@gmx.us) should you encounter any problems using the code.



