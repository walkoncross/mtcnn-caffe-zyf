#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp

import cv2
# import numpy as np

import time
import json

import _init_paths
from mtcnn_detector import MtcnnDetector, draw_faces

DO_RESIZE = False
RESIZED_LONG_SIDE = 640

MODEL_PATH = '../../model'

FDDB_IMG_ROOT_DIR = r'D:\FDDB_UMass\originalPics'
FDDB_FOLDS_DIR = r'D:\FDDB_UMass\FDDB-folds'
FOLDS_CNT = 10
# FDDB_IMG_ROOT_DIR = '/workspace/data/zjh'
# FDDB_FOLDS_DIR = "/workspace/data/zjh/FDDB-folds/"

OUTPUT_THRESHOLD = 0.7

SAVE_DIR = './fd_rlt_original'


def main(save_dir=None,
         save_img=False,
         show_img=False):

    minsize = 20
    caffe_model_path = MODEL_PATH
    threshold = [0.6, 0.7, OUTPUT_THRESHOLD]
    scale_factor = 0.709

    if not save_dir:
        save_dir = './fd_rlt'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_time = open(osp.join(save_dir, 'fd_time.txt'), 'w')

    t1 = time.clock()
    detector = MtcnnDetector(caffe_model_path)
    t2 = time.clock()

    msg = "initFaceDetector() costs %f seconds" % (t2 - t1)
    print(msg)
    fp_time.write(msg + '\n')

    ttl_time = 0.0
    img_cnt = 0

    for k in range(1, FOLDS_CNT + 1):
        k_str = str(k)
        if k != 10:
            k_str = "0" + k_str

        fn_list = osp.join(FDDB_FOLDS_DIR, "FDDB-fold-" + k_str + ".txt")
        fn_fd_rlt = osp.join(save_dir, "fold-" + k_str + "-out.txt")

        print('===========================')
        print('Process image list: ' + fn_list)
        print('Save results into: ' + fn_fd_rlt)

        fp_list = open(fn_list, 'r')
        fp_fd_rlt = open(fn_fd_rlt, 'w')

        for line in fp_list:
            imgname = line.strip()
            imgpath = osp.join(FDDB_IMG_ROOT_DIR, imgname + ".jpg")

            msg = "---> " + imgpath
            print(msg)
            fp_time.write(msg + 'n')

            img = cv2.imread(imgpath)

            if img is None:
                raise Exception('failed to load image: ' + imgpath)

            resize_factor = 1.0

            if DO_RESIZE:
                print('original image shape: {}'.format(img.shape))
                ht, wd, chs = img.shape

                if ht > wd:
                    resize_factor = float(RESIZED_LONG_SIDE) / ht
                else:
                    resize_factor = float(RESIZED_LONG_SIDE) / wd

                wd_new = int(resize_factor * wd)
                ht_new = int(resize_factor * ht)

                resized_img = cv2.resize(img, (wd_new, ht_new))
                print('resized image shape: {}'.format(resized_img.shape))

                # if show_img:
                #     cv2.imshow('resied_img', resized_img)

                #     ch = cv2.waitKey(0) & 0xFF
                #     if ch == 27:
                #         break
            else:
                resized_img = img

            resize_factor_inv = 1.0 / resize_factor

            img_cnt += 1
            t1 = time.clock()

            bboxes, points = detector.detect_face(resized_img, minsize,
                                                  threshold, scale_factor)

            t2 = time.clock()
            ttl_time += t2 - t1

            msg = "detect_face() costs %f seconds" % (t2 - t1)
            print(msg)
            fp_time.write(msg + '\n')

            fp_fd_rlt.write(imgname + "\n")
            fp_fd_rlt.write(str(len(bboxes)) + "\n")

            print points
            if DO_RESIZE:
                for i in range(len(bboxes)):
                    for j in range(4):
                        bboxes[i][j] *= resize_factor_inv

                    for j in range(10):
                        points[i][j] *= resize_factor_inv

            for i in range(len(bboxes)):
                fp_fd_rlt.write(str(bboxes[i][0]) + " ")
                fp_fd_rlt.write(str(bboxes[i][1]) + " ")
                fp_fd_rlt.write(str(bboxes[i][2] - bboxes[i][0]) + " ")
                fp_fd_rlt.write(str(bboxes[i][3] - bboxes[i][1]) + " ")
                fp_fd_rlt.write(str(bboxes[i][4]) + "\n")

            fp_fd_rlt.flush()

            msg = "===> Processed %d images, costs %f seconds, avg time: %f seconds" % (
                img_cnt, ttl_time, ttl_time / img_cnt)

            print(msg)
            fp_time.write(msg + '\n')
            fp_time.flush()

            if save_img or show_img:
                draw_faces(img, bboxes, points)

            if save_img:
                save_name = osp.join(save_dir, osp.basename(imgpath))
                cv2.imwrite(save_name, img)

            if show_img:
                cv2.imshow('img', img)

                ch = cv2.waitKey(0) & 0xFF
                if ch == 27:
                    break

        fp_list.close()
        fp_fd_rlt.close()

    fp_time.close()

    if show_img:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(SAVE_DIR, save_img=False, show_img=False)
