#coding=utf-8
import sys
#sys.path.append("/users/lihuan/pyworkspace/autoDriver/auto/lib/python3.6/site-packages")
import cv2
import numpy as np
image_file = "/users/lihuan/pyworkspace/autoDriver/road1.jpg"


def region_of_interest(img, vertices):
    #定义一个和输入图像同样大小的全黑图像mask，这个mask也称掩膜
    #掩膜的介绍，可参考：https://www.cnblogs.com/skyfsm/p/6894685.html
    mask = np.zeros_like(img)   
 
    #根据输入图像的通道数，忽略的像素点是多通道的白色，还是单通道的白色
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255


    #[vertices]中的点组成了多边形，将在多边形内的mask像素点保留，
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
 
    #与mask做"与"操作，即仅留下多边形部分的图像
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    left_lines_x = []
    left_lines_y = []
    right_lines_x = []
    right_lines_y = []
    center_lines_x = []
    center_lines_y = []
    line_y_max = 0
    line_y_min = 999
    print(len(lines))
    for line in lines:
        for x1,y1,x2,y2 in line:
            if y1 > line_y_max:
                line_y_max = y1
            if y2 > line_y_max:
                line_y_max = y2
            if y1 < line_y_min:
                line_y_min = y1
            if y2 < line_y_min:
                line_y_min = y2
            k = (y2 - y1)/(x2 - x1)
            if k < -0.3 and k > -0.4:
                left_lines_x.append(x1)
                left_lines_y.append(y1)
                left_lines_x.append(x2)
                left_lines_y.append(y2)
            elif k > 0.3:
                right_lines_x.append(x1)
                right_lines_y.append(y1)
                right_lines_x.append(x2)
                right_lines_y.append(y2)
            if (x2-x1 == 0):
                center_lines_x.append(x1)
                center_lines_y.append(y1)
                center_lines_x.append(x2)
                center_lines_y.append(y2)
    #最小二乘直线拟合
    print(left_lines_x)
    print(left_lines_y)
    left_line_k, left_line_b = np.polyfit(left_lines_x, left_lines_y, 1)
    right_line_k, right_line_b = np.polyfit(right_lines_x, right_lines_y, 1)
    # print(left_line_k)
    # print(right_line_k)
    center_line_k, center_line_b = np.polyfit(center_lines_x, center_lines_y, 1)
    #根据直线方程和最大、最小的y值反算对应的x
    cv2.line(img,
             (int((line_y_max - left_line_b)/left_line_k), line_y_max),
             (int((line_y_min - left_line_b)/left_line_k), line_y_min),
             [0,0,255], thickness)
    cv2.line(img,
             (int((line_y_max - right_line_b)/right_line_k), line_y_max),
             (int((line_y_min - right_line_b)/right_line_k), line_y_min),
             color, thickness)
    cv2.line(img, 
             (int(center_lines_x[0]), line_y_max),
             (int(center_lines_x[1]), line_y_min),
             [0, 255, 0], 2)
def main():
    img = cv2.imread(image_file)

    #gray(x,y) = 0.299*Red(x,y) + 0.587*Green(x,y) + 0.114*Blue(x,y)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #cv2.imshow("grayimage", gray)
    low_threshold = 50
    high_threshold = 150
    canny_image = cv2.Canny(gray, low_threshold, high_threshold)
    #cv2.imshow("cannyimage", canny_image)

    left_bottom = [0, canny_image.shape[0]]
    right_bottom = [canny_image.shape[1] - 20, canny_image.shape[0]]
    apex = [canny_image.shape[1]/2 - 10, 246]
   
    vertices = np.array([left_bottom, right_bottom, apex], np.int32)
    roi_image = region_of_interest(canny_image, vertices)
    cv2.imshow("roiimage", roi_image)
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 50     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 120 #minimum number of pixels making up a line
    max_line_gap = 70    # maximum gap in pixels between connectable line segments
    # Hough Transform 检测线段，线段两个端点的坐标存在lines中
    lines = cv2.HoughLinesP(roi_image, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    line_image = np.copy(img) # 复制一份原图，将线段绘制在这幅图上
    draw_lines(line_image, lines, [255, 0, 0], 6)
    cv2.imshow("lineImage", line_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    exit(main())





