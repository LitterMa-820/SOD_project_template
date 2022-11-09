# coding: utf-8
import os

dutrgbd_root_test = '../Testing_dataset/DUT-RGBD/'
njud_root_test = '../Testing_dataset/NJUD/'
nlpr_root_test = '../Testing_dataset/NLPR/'
stere_root_test = '../Testing_dataset/STERE/'
sip_root_test = '../Testing_dataset/SIP/'
rgbd135_root_test = '../Testing_dataset/RGBD135/'
ssd_root_test = '../Testing_dataset/SSD/'
lfsd_root_test = '../Testing_dataset/LFSD/'

dutrgbd = os.path.join(dutrgbd_root_test)
njud = os.path.join(njud_root_test)
nlpr = os.path.join(nlpr_root_test)
stere = os.path.join(stere_root_test)
sip = os.path.join(sip_root_test)
rgbd135 = os.path.join(rgbd135_root_test)
ssd = os.path.join(ssd_root_test)
lfsd = os.path.join(lfsd_root_test)

results_path = '../SSLSOD_v2Model_100_gen/'
results_save_path = '../results_eval/'
RGBD_SOD_Models = {'methodName': results_path}
