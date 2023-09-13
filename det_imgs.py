import os
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from clrnet.utils.config import Config
from clrnet.models.registry import build_net
from mmcv.parallel import MMDataParallel

from tqdm import tqdm

def to_cuda(batch):
    for k in batch:
        if not isinstance(batch[k], torch.Tensor):
            continue
        batch[k] = batch[k].cuda()
    return batch

def draw_lanes(img, output, lab_lines, ori_w, ori_h):
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255], [255,255,255], [0,0,0]]

    h,w,_ = img.shape

    # import pdb
    # print('draw lanes')
    # pdb.set_trace()
    for lane_id in range(len(output[0])):
        lane = output[0][lane_id]
        cr = colors[lane_id%len(colors)]

        for pt in lane:
            cx = int(pt[0]*w)
            cy = int(pt[1]*h)
            cv2.circle(img, (cx,cy), 3, cr, -1)

    if len(lab_lines) > 0:
        for line in lab_lines:
            line_pts = line.strip().split()
            pt_num = len(line_pts) // 2
            for i in range(pt_num):
                x = float(line_pts[2*i])
                y = float(line_pts[2*i+1])
                cx = int(x / float(ori_w) * w)
                cy = int(y / float(ori_h) * h)
                cv2.circle(img, (cx,cy), 4, [128,128,255], -1)


    

def main():
    args = parse_args()

    # import pdb 
    # pdb.set_trace()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.view = args.view
    cfg.seed = args.seed

    cfg.img_list = args.img_list
    cfg.save_dir = args.save_dir

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    fp = open(cfg.img_list)
    img_lines = fp.readlines()
    fp.close()


    #####################
    net = build_net(cfg)
    net = MMDataParallel(net, device_ids=range(cfg.gpus)).cuda()
    pretrained_model = torch.load(cfg.load_from)
    net.load_state_dict(pretrained_model['net'], strict=False)
    # import pdb
    # print('load weights')
    # pdb.set_trace()

    net.eval()
    predictions = []
    for i, img_line in enumerate(tqdm(img_lines, desc=f'Testing')):
        img_path = img_line.strip()
        img = cv2.imread(img_path)



        ori_h,ori_w,_ = img.shape
        if ori_h != cfg.ori_img_h or ori_w != cfg.ori_img_w:
            img = cv2.resize(img, (cfg.ori_img_w,cfg.ori_img_h), cv2.INTER_LINEAR)
        
        
        lab_lines = []
        if False:
            lab_path = img_path
            lab_path = lab_path.replace('.jpg', '.lines.txt')
            lab_path = lab_path.replace('.png', '.lines.txt')
            if os.path.exists(lab_path):
                # import pdb
                # pdb.set_trace()
                fp_lab = open(lab_path)
                lab_lines = fp_lab.readlines()
                fp_lab.close()


        cut_resize_img = cv2.resize(img[cfg.cut_height:, :, :], (cfg.img_w,cfg.img_h), cv2.INTER_LINEAR)

        data = torch.from_numpy(cut_resize_img/255.0).permute(2,0,1).float().unsqueeze(0)
        # data = torch.from_numpy(cut_resize_img).permute(2,0,1).float().unsqueeze(0)

        # import pdb
        # pdb.set_trace()

        data = data.cuda()
        with torch.no_grad():
            output = net(data)
            output = net.module.heads.get_lanes(output)

            img_name = os.path.basename(img_path)
            save_path = os.path.join(cfg.save_dir, img_name)

            # import pdb
            # print('save to file: '+save_path)
            # pdb.set_trace()
            draw_lanes(img, output, lab_lines, ori_w, ori_h)
            cv2.imwrite(save_path, img)




def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--img_list',
                        type=str,
                        default=None,
                        help='work dirs')
    parser.add_argument('--save_dir',
                        type=str,
                        default=None,
                        help='save dirs')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--view', action='store_true', help='whether to view')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args

# python det_imgs.py configs/clrnet/clr_resnet18_culane.py --load_from weights/culane_r18.pth --img_list banqiao_list.txt --save_dir results/banqiao_rs18 --gpus 0
# python det_imgs.py configs/clrnet/clr_dla34_culane.py --load_from weights/culane_dla34.pth --img_list banqiao_list.txt --save_dir results/banqiao_dla34 --gpus 0
# python det_imgs.py configs/clrnet/clr_resnet101_culane.py --load_from weights/culane_r101.pth --img_list banqiao_list.txt --save_dir results/banqiao_rs101 --gpus 0

# python det_imgs.py configs/clrnet/clr_resnet101_culane.py --load_from weights/culane_r101.pth --img_list banqiao_list.txt --save_dir results/banqiao_rs101 --gpus 0


if __name__ == '__main__':
    main()
