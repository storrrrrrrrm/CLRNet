import cv2
def culane_draw_label(label_txt,img_path):
    img = cv2.imread(img_path)
    
    vis_img = img.copy()

    with open(label_txt) as f:
        lines = f.readlines()
        for line in lines:
            line_points = line.split(' ')
            len_points = (len(line_points) - 1)/2 #remove last '\n'
            len_points = int(len_points)

            print('line has {} points'.format(len_points))

            for i in range(len_points):
                point_x = int(float(line_points[i * 2]))
                point_y = int(float(line_points[i * 2 + 1]))

                cv2.circle(vis_img, (point_x, point_y), 10, (255, 0, 0), -1)
    
    img_name = img_path.split('/')[-1]
    save_name = 'culane/{}'.format(img_name)
    cv2.imwrite(save_name,vis_img)

# import os
# def get_file(dirname,sortKey=None):
#     filelist=[]
#     for dirpath, dirname, filenames in os.walk(dirname):
#         img_filenames = [i for i in filenames if i.endswith('.png') or i.endswith('.jpg')]

#         # if sort:
#         #     filenames = sorted(filenames) #要排序　否则会乱序
#         for filename in sorted(img_filenames,key=sortKey): 
#             fullpath = os.path.join(dirpath, filename)
#             filelist.append(fullpath)

#     return filelist

# def sort_by_time(file_name):
#     """
#     文件名是数字｀
#     """
#     file_name_num = file_name[:-4]
#     nums = file_name_num.split('_')
   
#     return (int(nums[0]),int(nums[1]))


def culane_draw_curve_path():
    rootdir = '/mnt/data/public_datasets/CULane'
    with open('/mnt/data/public_datasets/CULane/list/test_split/test6_curve.txt') as f:
        imgs = f.readlines()
        #driver_100_30frame/05251624_0451.MP4/03570.jpg
        for i,img in enumerate(imgs):
            # if i > 20:
            #     break

            while img.endswith('\n') or img.endswith(' '): #remove ' ' and '\n' in the end
                img = img[:-1]

            print('img:{}'.format(img))
            full_imgpath = '{}/{}'.format(rootdir,img)
            img_name = img[:-4]
            print('img:{},img_name:{}'.format(img,img_name))
            label_name = '{}.lines.txt'.format(img_name)
            full_labelpath = '{}/{}'.format(rootdir,label_name)
            # print(full_labelpath)

            culane_draw_label(full_labelpath,full_imgpath)
                

import json
def curvelane_draw_label(label_txt,img_path):
    img = cv2.imread(img_path)
    
    vis_img = img.copy()

    with open(label_txt) as f:
        label = json.load(f)
        for line in label['Lines']:
            print(line)

            for point in line:
                print(point)
                point_y = int(float(point['y']))
                point_x = int(float(point['x']))
                cv2.circle(vis_img, (point_x, point_y), 10, (255, 0, 0), -1)
    
    img_name = img_path.split('/')[-1]
    save_name = 'curvelane/{}'.format(img_name)
    cv2.imwrite(save_name,vis_img)

if __name__ == '__main__':
    # culane_draw_label('03570.lines.txt',
    #            '03570.jpg')

    culane_draw_curve_path()

    # curvelane_draw_label('/mnt/data/public_datasets/curvelanes/Curvelanes/train/labels/0a5edd29c45fe55eb34407cef5df6902.lines.json',
    #     '/mnt/data/public_datasets/curvelanes/Curvelanes/train/images/0a5edd29c45fe55eb34407cef5df6902.jpg')

    # curvelane_draw_label('/mnt/data/public_datasets/curvelanes/Curvelanes/train/labels/1ac6bf9c8e44b54c61268e509d9758d1.lines.json',
    #     '/mnt/data/public_datasets/curvelanes/Curvelanes/train/images/1ac6bf9c8e44b54c61268e509d9758d1.jpg')
    