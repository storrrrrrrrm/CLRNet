#encoding=utf-8
import os
import random
import shutil

def get_file(dirname,sortKey=None):
    filelist=[]
    for dirpath, dirname, filenames in os.walk(dirname):
        img_filenames = [i for i in filenames if i.endswith('.png') or i.endswith('.jpg')]

        # if sort:
        #     filenames = sorted(filenames) #要排序　否则会乱序
        for filename in sorted(img_filenames,key=sortKey): 
            fullpath = os.path.join(dirpath, filename)
            filelist.append(fullpath)

    return filelist

def sort_by_time(file_name):
    """
    文件名是数字｀
    """
    file_name_num = file_name[:-4]
    nums = file_name_num.split('_')
   
    return (int(nums[0]),int(nums[1]))


def random_choose(imgdir='/mnt/data/public_datasets/banqiao/test/cam0'):
    filelist = get_file(imgdir,sort_by_time)
    # print(filelist)
    picked_list = random.sample(filelist, 200)
    print(picked_list)

    return picked_list


def gen_tobelabeled(imgdir,dstdir='/mnt/data/public_datasets/banqiao/keypoint_label'):
    picked_list = random_choose(imgdir)

    for imgpath in picked_list:
        shutil.copy(imgpath, dstdir)


def gen_txt(imgdir,txtname):
    filelist=[]
    filelist = get_file(imgdir,sort_by_time)
    with open(txtname,'w+') as f:
        for i,img_path in enumerate(filelist):
            f.writelines('{}{}'.format(img_path,'\n'))


if __name__ == '__main__':
    # gen_tobelabeled('/mnt/data/public_datasets/banqiao/test/cam0')
    gen_txt('/mnt/data/public_datasets/banqiao/keypoint_label','../banqiao_keypoint_label.txt')