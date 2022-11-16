import numpy as np
import os
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from results_eval.score_config import *
from utils.misc import check_mkdir
from models.VGG_baseline import VGG16Baseline
import ttach as tta

torch.manual_seed(2018)
torch.cuda.set_device(0)
ckpt_path = '../saved_models/baseLineModel'
args = {
    'snapshot': 'Baseline_1_gen',
    'crf_refine': False,
    'save_results': True
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

''''
rgbd prediction 
'''
to_test = {'DUT-RGBD': dutrgbd, 'NJUD': njud, 'NLPR': nlpr, 'STERE': stere, 'SIP': sip, 'RGBD135': rgbd135, 'SSD': ssd,
           'LFSD': lfsd}



transforms = tta.Compose(
    [
        # tta.HorizontalFlip(),
        # tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
        tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)

def main():
    t0 = time.time()
    net = VGG16Baseline().cuda()
    print('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot']+'.pth')))
    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            print(name, root)
            root1 = os.path.join(root, 'depth')
            img_list = [os.path.splitext(f) for f in os.listdir(root1)]
            # print(img_list)
            for idx, img_name in enumerate(img_list):
                print('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                rgb_png_path = os.path.join(root, 'RGB', img_name[0] + '.png')
                rgb_jpg_path = os.path.join(root, 'RGB', img_name[0] + '.jpg')
                depth_jpg_path = os.path.join(root, 'depth', img_name[0] + '.jpg')
                depth_png_path = os.path.join(root, 'depth', img_name[0] + '.png')
                if os.path.exists(rgb_png_path):
                    img = Image.open(rgb_png_path).convert('RGB')
                else:
                    img = Image.open(rgb_jpg_path).convert('RGB')
                if os.path.exists(depth_jpg_path):
                    depth = Image.open(depth_jpg_path).convert('L')
                else:
                    depth = Image.open(depth_png_path).convert('L')

                w_, h_ = img.size
                img_resize = img.resize([256, 256], Image.BILINEAR)  # Foldconv cat是320
                depth_resize = depth.resize([256, 256], Image.BILINEAR)  # Foldconv cat是320
                img_var = Variable(img_transform(img_resize).unsqueeze(0)).cuda()
                depth_var = Variable(depth_transform(depth_resize).unsqueeze(0)).cuda()
                n, c, h, w = img_var.size()
                depth_3 = torch.cat((depth_var, depth_var, depth_var), 1)
                mask = []
                for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var)
                    d_trans = transformer.augment_image(depth_3)
                    side_out5, side_out4, side_out3, side_out2, side_out1 = net(rgb_trans, d_trans)
                    deaug_mask = transformer.deaugment_mask(side_out1)
                    mask.append(deaug_mask)

                prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction = prediction.sigmoid()
                prediction = to_pil(prediction.data.squeeze(0).cpu())
                prediction = prediction.resize((w_, h_), Image.BILINEAR)
                # if args['crf_refine']:
                #     prediction = crf_refine(np.array(img), np.array(prediction))
                if args['save_results']:
                    check_mkdir(os.path.join(results_save_path, args['snapshot'], name))
                    prediction.save(os.path.join(results_save_path, args['snapshot'], name, img_name[0] + '.png'))


if __name__ == '__main__':
    main()