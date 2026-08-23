import argparse
import logging
import time
import cv2
import numpy as np
from PIL import ImageTk
from PIL.Image import Image

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh





def comp(a, b):
    if (a > b):
        return a > 8 * b
    else:
        return b > 8 * a


def comp1(a, b):
    if (a < b):
        return b < 4 * a
    else:
        return a < 4 * b


def charge0(humans):
    shang = (14, 15, 16, 17, 1, 0, 8, 11)
    xia = (8, 9, 10, 11, 12, 13)
    mo = 0
    if (len(humans) > 0):
        for ss in range(0, len(humans)):
            J89 = J910 = J1112 = J1213 = J010 = J013 = J18 = J111 = J98 = J1211 = 10000
            if ((8 in humans[ss].body_parts.keys()) == True & (9 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[9].y - humans[ss].body_parts[8].y == 0):
                    J89 = 0.001
                else:
                    J89 = abs(humans[ss].body_parts[9].x - humans[ss].body_parts[8].x) / abs(
                        humans[ss].body_parts[9].y - humans[ss].body_parts[8].y)
            if ((8 in humans[ss].body_parts.keys()) == True & (9 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[9].x - humans[ss].body_parts[8].x == 0):
                    J98 = 10000
                else:
                    J98 = abs(humans[ss].body_parts[9].y - humans[ss].body_parts[8].y) / abs(
                        humans[ss].body_parts[9].x - humans[ss].body_parts[8].x)
            if ((9 in humans[ss].body_parts.keys()) == True & (10 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[10].y - humans[ss].body_parts[9].y == 0):
                    J910 = 0.0001
                else:
                    J910 = abs(humans[ss].body_parts[10].x - humans[ss].body_parts[9].x) / abs(
                        humans[ss].body_parts[10].y - humans[ss].body_parts[9].y)
            if ((11 in humans[ss].body_parts.keys()) == True & (12 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[11].y - humans[ss].body_parts[12].y == 0):
                    J1112 = 0.001
                else:
                    J1112 = abs(humans[ss].body_parts[11].x - humans[ss].body_parts[12].x) / abs(
                        humans[ss].body_parts[11].y - humans[ss].body_parts[12].y)
            if ((11 in humans[ss].body_parts.keys()) == True & (12 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[11].x - humans[ss].body_parts[12].x == 0):
                    J1211 = 10000
                else:
                    J1211 = abs(humans[ss].body_parts[11].y - humans[ss].body_parts[12].y) / abs(
                        humans[ss].body_parts[11].x - humans[ss].body_parts[12].x)
            if ((12 in humans[ss].body_parts.keys()) == True & (13 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[12].y - humans[ss].body_parts[13].y == 0):
                    J1213 = 0.001
                else:
                    J1213 = abs(humans[ss].body_parts[13].x - humans[ss].body_parts[12].x) / abs(
                        humans[ss].body_parts[13].y - humans[ss].body_parts[12].y)
            if ((0 in humans[ss].body_parts.keys()) == True & (10 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[0].x - humans[ss].body_parts[10].x == 0):
                    J010 = 0.001
                else:
                    J010 = abs(humans[ss].body_parts[0].y - humans[ss].body_parts[10].y) / abs(
                        humans[ss].body_parts[0].x - humans[ss].body_parts[10].x)
            if ((0 in humans[ss].body_parts.keys()) == True & (13 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[0].x - humans[ss].body_parts[13].x == 0):
                    J013 = 0.001
                else:
                    J013 = abs(humans[ss].body_parts[0].y - humans[ss].body_parts[13].y) / abs(
                        humans[ss].body_parts[0].x - humans[ss].body_parts[13].x)
            if ((1 in humans[ss].body_parts.keys()) == True & (8 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[1].x - humans[ss].body_parts[8].x == 0):
                    J18 = 0.001
                else:
                    J18 = abs(humans[ss].body_parts[1].y - humans[ss].body_parts[8].y) / abs(
                        humans[ss].body_parts[1].x - humans[ss].body_parts[8].x)
            if ((1 in humans[ss].body_parts.keys()) == True & (11 in humans[ss].body_parts.keys()) == True):
                if (humans[ss].body_parts[1].x - humans[ss].body_parts[11].x == 0):
                    J111 = 0.001
                else:
                    J111 = abs(humans[ss].body_parts[1].y - humans[ss].body_parts[11].y) / abs(
                        humans[ss].body_parts[1].x - humans[ss].body_parts[11].x)

            x = np.empty(shape=[0, 2], dtype=int)
            y = np.empty(shape=[0, 2], dtype=int)
            for ca in range(0, 18):
                if ((ca in humans[ss].body_parts.keys()) == True & (ca in shang) == True):
                    x = np.append(x, [[int(100 * round(humans[ss].body_parts[ca].x, 2)),
                                       int(100 * round(humans[ss].body_parts[ca].y, 2))]], axis=0)
                elif ((ca in humans[ss].body_parts.keys()) == True & (ca in xia) == True):
                    y = np.append(x, [[int(100 * round(humans[ss].body_parts[ca].x, 2)),
                                       int(100 * round(humans[ss].body_parts[ca].y, 2))]], axis=0)
            xx = cv2.minAreaRect(x)
            yy = cv2.minAreaRect(y)
            # print("xx=",abs(int(xx[2])))
            # print("yy=",abs(int(yy[2])))
            # print("chang =",yy[1][0])
            # print("yuan = ",yy[1][1])
            # if((abs(int(yy[2])>70)) == True &)
            mkb = 0
            if (((8 in humans[ss].body_parts.keys()) == True & (
                    9 in humans[ss].body_parts.keys()) == True) == True | (
                    (11 in humans[ss].body_parts.keys()) == True & (
                    12 in humans[ss].body_parts.keys()) == True) == True):
                if ((8 in humans[ss].body_parts.keys()) == True & (9 in humans[ss].body_parts.keys()) == True):
                    print(J98)
                    print(J1211)
                    if ((J98 < 0.5) == True):
                        if ((16 in humans[ss].body_parts.keys()) == True):
                            cv2.putText(image, "sit", (
                                int(600 * round(humans[ss].body_parts[16].x, 2)),
                                int(450 * round(humans[ss].body_parts[16].y, 2))),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            mkb = 1
                        elif ((17 in humans[ss].body_parts.keys()) == True):
                            cv2.putText(image, "sit", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                                                       int(450 * round(humans[ss].body_parts[17].y, 2))),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            mkb = 1
                elif ((11 in humans[ss].body_parts.keys()) == True & (12 in humans[ss].body_parts.keys()) == True):
                    if ((J1211 < 0.5) == True):
                        if ((16 in humans[ss].body_parts.keys()) == True):
                            cv2.putText(image, "sit", (
                                int(600 * round(humans[ss].body_parts[16].x, 2)),
                                int(450 * round(humans[ss].body_parts[16].y, 2))),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            mkb = 1
                        elif ((17 in humans[ss].body_parts.keys()) == True):
                            cv2.putText(image, "sit", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                                                       int(450 * round(humans[ss].body_parts[17].y, 2))),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            mkb = 1

            if ((abs(int(xx[2])) > 57) == True & (abs(int(yy[2])) > 70) == True & comp(yy[1][0],
                                                                                       yy[1][1]) == True & (
                    mkb == 0) == True):
                if ((16 in humans[ss].body_parts.keys()) == True):
                    cv2.putText(image, "stand", (int(600 * round(humans[ss].body_parts[16].x, 2)),
                                                 int(450 * round(humans[ss].body_parts[16].y, 2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                elif ((17 in humans[ss].body_parts.keys()) == True):
                    cv2.putText(image, "stand", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                                                 int(450 * round(humans[ss].body_parts[17].y, 2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif ((abs(int(xx[2])) < 8) == True & (abs(int(yy[2])) > 70) == True & comp(yy[1][0],
                                                                                        yy[1][1]) == True & (
                          mkb == 0) == True):
                if ((16 in humans[ss].body_parts.keys()) == True):
                    cv2.putText(image, "stand", (
                        int(600 * round(humans[ss].body_parts[16].x, 2)),
                        int(450 * round(humans[ss].body_parts[16].y, 2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                elif ((17 in humans[ss].body_parts.keys()) == True):
                    cv2.putText(image, "stand", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                                                 int(450 * round(humans[ss].body_parts[17].y, 2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif ((abs(int(xx[2])) > 57) == True & (abs(int(yy[2])) < 8) == True & comp(yy[1][0],
                                                                                        yy[1][1]) == True & (
                          mkb == 0) == True):
                if ((16 in humans[ss].body_parts.keys()) == True):
                    cv2.putText(image, "stand", (
                        int(600 * round(humans[ss].body_parts[16].x, 2)),
                        int(450 * round(humans[ss].body_parts[16].y, 2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                elif ((17 in humans[ss].body_parts.keys()) == True):
                    cv2.putText(image, "stand", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                                                 int(450 * round(humans[ss].body_parts[17].y, 2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif ((abs(int(xx[2])) < 8) == True & (abs(int(yy[2])) < 8) == True & comp(yy[1][0],
                                                                                       yy[1][1]) == True & (
                          mkb == 0) == True):
                if ((16 in humans[ss].body_parts.keys()) == True):
                    cv2.putText(image, "stand", (
                        int(600 * round(humans[ss].body_parts[16].x, 2)),
                        int(450 * round(humans[ss].body_parts[16].y, 2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                elif ((17 in humans[ss].body_parts.keys()) == True):
                    cv2.putText(image, "stand", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                                                 int(450 * round(humans[ss].body_parts[17].y, 2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif (comp1(yy[1][0], yy[1][1]) == True):
                if (mkb == 0):
                    if ((16 in humans[ss].body_parts.keys()) == True):
                        cv2.putText(image, "bend", (int(600 * round(humans[ss].body_parts[16].x, 2)),
                                                    int(450 * round(humans[ss].body_parts[16].y, 2))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    elif ((17 in humans[ss].body_parts.keys()) == True):
                        cv2.putText(image, "bend", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                                                    int(450 * round(humans[ss].body_parts[17].y, 2))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if (((0 in humans[ss].body_parts.keys()) == True & (13 in humans[ss].body_parts.keys()) == True) | (
                    (0 in humans[ss].body_parts.keys()) == True & (10 in humans[ss].body_parts.keys()) == True)):
                liex = 0
                if ((0 in humans[ss].body_parts.keys()) == True & (13 in humans[ss].body_parts.keys()) == True):
                    if ((J013 < 0.15) == True):
                        if ((16 in humans[ss].body_parts.keys()) == True):
                            cv2.putText(image, "lie", (
                                int(600 * round(humans[ss].body_parts[16].x, 2)),
                                int(450 * round(humans[ss].body_parts[16].y, 2))),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            liex = 1

                        elif ((17 in humans[ss].body_parts.keys()) == True):
                            cv2.putText(image, "lie", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                                                       int(450 * round(humans[ss].body_parts[17].y, 2))),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            liex = 1

                if ((0 in humans[ss].body_parts.keys()) == True & (10 in humans[ss].body_parts.keys()) == True & (
                        liex == 0) == True):
                    if ((J010 < 0.15) == True):
                        if ((16 in humans[ss].body_parts.keys()) == True):
                            cv2.putText(image, "lie", (
                                int(600 * round(humans[ss].body_parts[16].x, 2)),
                                int(450 * round(humans[ss].body_parts[16].y, 2))),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        elif ((17 in humans[ss].body_parts.keys()) == True):
                            cv2.putText(image, "lie", (int(600 * round(humans[ss].body_parts[17].x, 2)),
                                                       int(450 * round(humans[ss].body_parts[17].y, 2))),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if (((4 in humans[ss].body_parts.keys()) == True & (
                    3 in humans[ss].body_parts.keys()) == True) == True | (
                    (7 in humans[ss].body_parts.keys()) == True & (
                    6 in humans[ss].body_parts.keys()) == True) == True):
                armx = 0
                if ((4 in humans[ss].body_parts.keys()) == True & (3 in humans[ss].body_parts.keys()) == True):
                    if (humans[ss].body_parts[4].y <= humans[ss].body_parts[3].y):
                        cv2.putText(image, "arm", (int(600 * round(humans[ss].body_parts[4].x, 2)),
                                                   int(450 * round(humans[ss].body_parts[4].y, 7))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        armx = 1
                if ((7 in humans[ss].body_parts.keys()) == True & (6 in humans[ss].body_parts.keys()) == True & (
                        armx == 0) == True):
                    if (humans[ss].body_parts[7].y <= humans[ss].body_parts[6].y):
                        cv2.putText(image, "arm", (int(600 * round(humans[ss].body_parts[7].x, 2)),
                                                   int(450 * round(humans[ss].body_parts[7].y, 2))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
parser.add_argument('--camera', type=int, default=0)

parser.add_argument('--resize', type=str, default='0x0',
                    help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                    help='if provided, resize heatmaps before they are post-processed. default=1.0')

parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
parser.add_argument('--show-process', type=bool, default=False,
                    help='for debug purpose, if enabled, speed for inference is dropped.')
args = parser.parse_args()

logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
w, h = model_wh(args.resize)
if w > 0 and h > 0:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
else:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
logger.debug('cam read+')
image = self.im_rd

logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

while True:

    image = self.im_rd

    logger.debug('')
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    #判断站立的几种情况：
    charge0(humans)
    #charge(humans)

      #      print(humans[ss].body_parts)
         #   x = np.empty(shape=[0, 2], dtype=int)
          #  for ca in range(0,17):
           #     if((ca in humans[ss].body_parts.keys()) == True):
            #        x = np.append(x,[[int(100*round(humans[ss].body_parts[ca].x,2)),int(100*round(humans[ss].body_parts[ca].y,2))]],axis=0)
            #print(x)
            #axis_list = np.array(x)
            #hull = cv2.convexHull(axis_list, clockwise=True, returnPoints=True)
            #print(hull)
            #hull = np.squeeze(hull)
            #plt.scatter(axis_list[:, 0], axis_list[:, 1])
            #plt.plot(hull[:, 0], hull[:, 1], "r")
            #plt.plot([hull[-1, 0], hull[0, 0]], [hull[-1, 1], hull[0, 1]], "r")
           ## plt.show()
    logger.debug('')
    image0 = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    logger.debug('')
    image1 = Image.fromarray(cv2.cvtColor(image0, cv2.COLOR_BGR2RGB))
    image2 = ImageTk.PhotoImage(image=image1)
    #cv2.putText(image,
     #           "FPS: %f" % (1.0 / (time.time() - fps_time)),
      #          (100, 200),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
       #         (0, 255, 0), 2)
    #if (len(humans) > 0):
     #   for ss in range(0, len(humans)):
      #      print(humans[ss].body_parts)
         #   if((14 in humans[ss].body_parts.keys()) == True):
          #      print(int(400 * round(humans[ss].body_parts[14].x, 2)))
           #     print(int(400 * round(humans[ss].body_parts[14].y, 2)))
            #    cv2.putText(image, "zhanli", (int(600*round(humans[ss].body_parts[14].x,2)), int(290*round(humans[ss].body_parts[14].x,2))
#), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    self.lab[1].imgtk = image2
    self.lab[1].config(image=image2)

    fps_time = time.time()

    logger.debug('')


