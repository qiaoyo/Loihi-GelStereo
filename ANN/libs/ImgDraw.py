import cv2
import numpy as np


def compute_cov(init_blobs, cur_blobs):
    a = np.array(init_blobs)
    b = np.array(cur_blobs)

def draw_arrow(img, init_blobs, current_blobs,width,height):
    assert len(init_blobs) == len(current_blobs)
    init_blobs = blobs_sorting_temp1(init_blobs, width=width, height=height)
    current_blobs = blobs_sorting_temp1(current_blobs, width=width, height=height)
    # init_blobs = blobs_sorting(init_blobs,width=width,height=height)
    # current_blobs = blobs_sorting(current_blobs, width=width,height=height)

    for i in range(len(init_blobs)):
        # print(init_blobs[i], current_blobs[i])
        cv2.arrowedLine(img, tuple(current_blobs[i]), tuple(init_blobs[i]), (0, 0, 255), 2, 0, 0, 0.1)
    return img

def draw_arrow_2(img, current_blobs, init_blobs, color=(0, 255, 255)):
    assert len(init_blobs) == len(current_blobs)
    # init_blobs = blobs_sorting_temp1(init_blobs, width=width, height=height)
    # current_blobs = blobs_sorting_temp1(current_blobs, width=width, height=height)
    # init_blobs = blobs_sorting(init_blobs,width=width,height=height)
    # current_blobs = blobs_sorting(current_blobs, width=width,height=height)

    for i in range(len(init_blobs)):
        # print(init_blobs[i], current_blobs[i])
        cv2.arrowedLine(img, tuple(current_blobs[i]), tuple(init_blobs[i]), color, 2, 0, 0, 0.1)
    return img

def draw_points(img, blobs, point_color=(0, 0, 255)):
    point_size = 2
    # point_color = (0, 0, 255)  # BGR
    thickness = 2  # 0 、4、8
    for ce in blobs:
        cv2.circle(img, (int(ce[0]), int(ce[1])), point_size, point_color, thickness)
    return img

def draw_lines(img, blobs, line_color=(0, 255, 0)):
    thickness = 1
    # line_type = LIN
    for i in range(len(blobs)-1):
        first_blobs = blobs[i]
        second_blobs = blobs[i+1]
        cv2.line(img, (int(first_blobs[0]), int(first_blobs[1])), (int(second_blobs[0]), int(second_blobs[1])), color=line_color, thickness=thickness)
    return img