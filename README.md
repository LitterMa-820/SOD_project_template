# What is this project about?
&emsp;&emsp;This project is a template project for multi modality Salient Object Detection(SOD).
# Why?
&emsp;&emsp;Many authors of SOD paper uploaded corresponding codes to GitHub, but these codes are diverse and some may be hard to read.
In order to provide a simple, uniform and useful environment for researchers and readers, we built this template. 
# Introduction
- models
  - Model structure storing in this directory. 
- saved_model
  - model after training should be stored in here.
- datasets
  - SOD datasets is here.
- data_loader
  - The method to load image data.
- result_eval
  - This directory has some evaluating metrics of SOD task (MAE,maxF,avgF,wFm,SM,EM,et al), filling path in score_config.py and run sod_test_score.py to use it.
  - first python .\prediction.py --ss='your model snapshot' --modal='rgbd or rgbd'
  - second python .\mae_mF_wF_Sm_Em.py --mn='your method name' --modal='rgbd or rgbd' --p='the prediction document name in results' --sn='result file name'
- utils
  - This directory has many useful tools like model parameters loading and visible model test. The utils chapter will introduce all function of it at large.
- test
  - Some test During development could put in this directory.

# Utils

# Acknowledgement

## Technical supports
- The evaluation metrics code was extracted from https://github.com/Xiaoqi-Zhao-DLUT/SSLSOD. 
## The works that used our template 
