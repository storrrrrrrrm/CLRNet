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


class ConfigReader():
    def __init__(self) -> None:
        self.rx = 40
        self.ry = 7
        self.img_h = 736
        self.img_w = 1280
    
config_linefit = ConfigReader()

def to_cuda(batch):
    for k in batch:
        if not isinstance(batch[k], torch.Tensor):
            continue
        batch[k] = batch[k].cuda()
    return batch

def draw_lanes(img, output, lab_lines, ori_w, ori_h):
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255], [255,255,255], [0,0,0]]

    h,w,_ = img.shape
    print('draw_lanes,h:{},w:{}'.format(h,w))

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

        x = [int(pt[0]*w) for pt in lane]
        y = [int(pt[1]*h) for pt in lane]
        lanename = '{}'.format(lane_id)
        fit_line((x,y),img,ori_w,ori_h,lanename)

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


def from_front_to_bev(lane):
    
    return lane_bev

def convert_to_dst_img(lane,img,ori_w, ori_h):
    """
    将模型输出的x,y变换为config_linefit.img_w x config_linefit.img_h的图片上的x,y(H矩阵是在此分辨率标出的)
    """
    print('convert_to_dst_img begin----------------------')
    x,y = lane
    h,w,_ = img.shape

    # print('w:{},ori_w:{}'.format(w,ori_w))

    h_ratio,w_ratio = h/ori_h,w/ori_w

    x_on_ori = [int(e/w_ratio) for e in x]
    y_on_ori = [int(e/h_ratio) for e in y]

    # print(x)
    # print(x_on_dst)

    h_ratio,w_ratio = ori_w/config_linefit.img_w, ori_h/config_linefit.img_h

    x_on_dst = [min(config_linefit.img_w,int(e/w_ratio)) for e in x_on_ori]
    y_on_dst = [min(config_linefit.img_h,int(e/h_ratio)) for e in y_on_ori ]

    # print('x_on_dst:{}'.format(x_on_dst))
    # print('y_on_dst:{}'.format(y_on_dst))
    #draw on config_linefit.img_w x config_linefit.img_h img
    dst_img = np.zeros((config_linefit.img_h,config_linefit.img_w,3))
    dst_img[y_on_dst,x_on_dst,0] = 255
    for x,y in zip(x_on_dst, y_on_dst):
        dst_img[y,x,0] = 255
        cv2.circle(dst_img, (x,y), 3, (255,0,0), -1)
    # cv2.imwrite('./lane_H.png',dst_img) 

    print('convert_to_dst_img end----------------------')
    return (x_on_dst,y_on_dst),dst_img

def cord_from_bev_to_car(x,y):
    """
    car cord: forward:y right:x
    """
    rx,ry = config_linefit.rx,config_linefit.ry  #40 pixel every meter on x axis,7 pixel every meter on y axis
    pos_on_car_cord_x = (x - config_linefit.img_w / 2)/rx
    pos_on_car_cord_y = -(y - config_linefit.img_h)/ry

    # print('pos_on_car_cord_x:{}\n,pos_on_car_cord_y:{}'.format(pos_on_car_cord_x,pos_on_car_cord_y))

    return pos_on_car_cord_x,pos_on_car_cord_y

def fit_line(lane,img,ori_w,ori_h,lanename):
    """
    lane:(x:list,y:list)
    """
    print('fit line begin---------------------')

    x,y = lane
    (x_on_dst,y_on_dst),dst_img = convert_to_dst_img(lane,img,ori_w,ori_h)

    cv2.imwrite('./lane_H_{}.png'.format(lanename),dst_img) 

    ones = [1] * len(x_on_dst)
    lane_vec = (x_on_dst,y_on_dst,ones)
    lane_vec = np.array(lane_vec)  # 3 x N

    H = np.array([[-0.190334708341416, -1.894825663288161, 761.9104204838158],
    [1.419364949735544e-06, -2.179288342761436, 781.8075505551056],
    [2.236333712141676e-09, -0.002960664791094558, 1]])

    bev_pos = H.dot(lane_vec) # 3 x n
    bev_pos = bev_pos.T # n x 3
    last_col = bev_pos[:,-1].reshape(-1,1)  # (n,1)
    bev_pos = bev_pos/last_col #normalize
    
    x_bev,y_bev = bev_pos[:,0],bev_pos[:,1]
    x_bev = [int(e) for e in x_bev]
    y_bev = [int(e) for e in y_bev]
    
    #filter bev 由于H的限制(H的最远范围为100m),转换出的y可能为负值.
    idx = -1
    for i,y in enumerate(y_bev):
        if y > 0:
            idx = i
            break
    x_bev = x_bev[idx:]
    y_bev = y_bev[idx:]
    # print('x_bev:{}'.format(x_bev))
    # print('y_bev:{}'.format(y_bev))

    #draw on bev
    bev_img = np.zeros((config_linefit.img_h,config_linefit.img_w,3))
    bev_img[y_bev,x_bev,0] = 255
    for x,y in zip(x_bev, y_bev):
        bev_img[y,x,0] = 255
        cv2.circle(dst_img, (x,y), 3, (255,0,0), -1)
    cv2.imwrite('./bev_H_{}.png'.format(lanename),bev_img) 

    pos_on_car_cord_x,pos_on_car_cord_y = cord_from_bev_to_car(np.array(x_bev),np.array(y_bev))

    line_params = np.polyfit(pos_on_car_cord_y,pos_on_car_cord_x, 3)

    print('line_params on car_cord:{}'.format(line_params))

    h,w,_ = img.shape
    ploty = np.linspace(int(h/2), h-1, int(h/2)) 
    fitx = line_params[0]*ploty**3 + line_params[1]*ploty**2 + line_params[2]*ploty + line_params[3]

    # draw lane points calculated by left_fit/right_fit
    ploty = [int(e) for e in ploty ]
    fitx = [min(max(int(e), 0), w-1) for e in fitx] #limit x to [0,w]
    
    for i in range(-3,3): #只画一条线的话　太模糊了　看不出来
        fitx = [min(w-1,max(0,e+i)) for e in fitx]
        
        # img[ploty,fitx] = (255,0,0)
    

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

        print('cut_resize_img shape:',cut_resize_img.shape)

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

# python det_imgs.py  configs/clrnet/clr_resnet18_culane.py --load_from weights/llamas_r18.pth --img_list banqiao_keypoint_label.txt --save_dir results/banqiao_llamas_rs18 --gpus 0
# python det_imgs.py  configs/clrnet/clr_resnet18_culane.py --load_from weights/llamas_r18.pth --img_list banqiao_test.txt --save_dir results/banqiao_llamas_finetune_rs18 --gpus 0
 
if __name__ == '__main__':
    main()
