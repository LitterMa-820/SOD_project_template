import datetime

import numpy as np
import sys
import argparse

sys.path.append('..')
from results_eval.load_test_data import test_dataset
from results_eval.saliency_metric import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber, cal_acc
from score_config import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mn', required=True, type=str, help='model name')
parser.add_argument('--modal', required=True, type=str, help='rgbd or rgbt')
parser.add_argument('--p', required=True, type=str, help='path')
parser.add_argument('--sn', required=True, type=str, help='save_name')
opt = parser.parse_args()
if opt.modal == 'rgbd':
    test_datasets = {'DUT-RGBD': dutrgbd, 'NJU2K': njud, 'NJUD': njud, 'NLPR': nlpr, 'STERE': stere, 'SIP': sip,
                     'DES': rgbd135, 'RGBD135': rgbd135, 'SSD': ssd, 'LFSD': lfsd}
elif opt.modal == 'rgbt':
    test_datasets = {'VT-821': vt821, 'VT-1000': vt1000, 'VT-5000': vt5000}
else:
    print('please input rgbd or rgbt')
    exit()

RGBD_SOD_Models = {opt.mn: os.path.join(results_path, opt.p)}

results_save_path = results_save_path + opt.sn + '-' + str(
    datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')) + '.txt'
# table head
open(results_save_path, 'w').write(opt.mn + '\n' + '\\begin{tabular}{cccccc}\n\\toprule' + '\n' + '\midrule\n')
open(results_save_path, 'a').write('datasets/metric & mae & maxF & wFm & Em & Sm \\\\\n')
avg_mae, avg_max_f, avg_sm, avg_em, avg_wfm = [], [], [], [], []
for method_name, method_map_root in RGBD_SOD_Models.items():
    print('test method:', method_name, method_map_root)
    for name, root in test_datasets.items():
        print(name)
        sal_root = method_map_root + name
        print(sal_root)
        gt_root = root + 'GT'
        print(gt_root)
        if os.path.exists(sal_root):
            print('\033[32m file exist! \033[0m')
            test_loader = test_dataset(sal_root, gt_root)
            mae, fm, sm, em, wfm, m_dice, m_iou, ber, acc = cal_mae(), cal_fm(
                test_loader.size), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber(), cal_acc()
            for i in tqdm(range(test_loader.size)):
                # print ('predicting for %d / %d' % ( i + 1, test_loader.size))
                sal, gt = test_loader.load_data()
                if sal.size != gt.size:
                    x, y = gt.size
                    sal = sal.resize((x, y))
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                gt[gt > 0.5] = 1
                gt[gt != 1] = 0
                res = sal
                res = np.array(res)
                if res.max() == res.min():
                    res = res / 255
                else:
                    res = (res - res.min()) / (res.max() - res.min())
                # 二值化会提升mae和meanf,em
                # res[res > 0.5] = 1
                # res[res != 1] = 0
                mae.update(res, gt)
                sm.update(res, gt)
                fm.update(res, gt)
                em.update(res, gt)
                wfm.update(res, gt)
            MAE = mae.show()
            avg_mae.append(MAE)
            # maxf, meanf, _, _ = fm.show()
            maxf, _, _, _ = fm.show()
            avg_max_f.append(maxf)
            # avg_mean_f.append(meanf)
            sm = sm.show()
            avg_sm.append(sm)
            em = em.show()
            avg_em.append(em)
            wfm = wfm.show()
            avg_wfm.append(wfm)
            log = 'method_name: {} dataset: {} MAE: {:.4f} maxF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f} '.format(
                method_name, name, MAE, maxf, wfm, sm, em)
            print('\n' + log)
            table_content = name + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' \\\\'
            table_content = table_content.format(MAE, maxf, wfm, em, sm)
            open(results_save_path, 'a').write(table_content + '\n')
        else:
            print('\033[31m file is not exist! \033[0m')

    avg_mae, avg_max_f, avg_wfm, avg_sm, avg_em = np.mean(avg_mae), np.mean(avg_max_f), np.mean(avg_wfm), np.mean(
        avg_sm), np.mean(avg_em)
    avg_log = 'method_name: {} on all dataset avg_MAE: {:.4f} avg_maxF: {:.4f} avg_wfm: {:.4f} avg_Sm: {:.4f} avg_Em:'.format(
        method_name, np.mean(avg_mae), np.mean(avg_max_f), np.mean(avg_wfm), np.mean(avg_sm), np.mean(avg_em))
    print(avg_log)

    table_avg = 'average' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' \\\\'
    table_avg = table_avg.format(avg_mae, avg_max_f, avg_wfm, avg_sm, avg_em)
    open(results_save_path, 'a').write(table_avg + '\n')
    open(results_save_path, 'a').write('\\bottomrule\n\end{tabular}' + '\n')
