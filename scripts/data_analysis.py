import cv2
def draw_label(label_txt,img_path):
    img = cv2.imread(img_path)
    
    vis_img = img.copy()

    with open(label_txt) as f:
        lines = f.readlines()
        for line in lines:
            line_points = line.split(' ')
            len_points = (len(line_points) - 1)/2 #remove lase '\n'
            len_points = int(len_points)

            print('line has {} points'.format(len_points))

            for i in range(len_points):
                point_x = int(float(line_points[i * 2]))
                point_y = int(float(line_points[i * 2 + 1]))

                cv2.circle(vis_img, (point_x, point_y), 10, (255, 0, 0), -1)
    
    cv2.imwrite('./vis_img.png',vis_img)

if __name__ == '__main__':
    draw_label('03570.lines.txt',
               '03570.jpg')