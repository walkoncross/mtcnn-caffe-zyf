#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp

import cv2
import numpy as np

import time
import json

from mtcnn_detector import MtcnnDetector, draw_faces


def print_usage():
    usage = 'python %s <img-list-file> <save-dir>' % osp.basename(__file__)
    print('USAGE: ' + usage)


def main(img_list_fn,
         save_dir,
         save_img=True,
         show_img=False):

    minsize = 20
    caffe_model_path = "./model"
    threshold = [0.6, 0.7, 0.7]
    scale_factor = 0.709

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_rlt = open(osp.join(save_dir, 'mtcnn_fd_rlt.json'), 'w')

    result_list = []

    t1 = time.clock()
    detector = MtcnnDetector(caffe_model_path)
    t2 = time.clock()
    print("initFaceDetector() costs %f seconds" % (t2 - t1))

    fp = open(img_list_fn, 'r')

    ttl_time = 0.0
    img_cnt = 0

    for line in fp:
        imgpath = line.strip()
        print("\n===>" + imgpath)
        if imgpath == '':
            print 'empty line, not a file name, skip to next'
            continue
        if imgpath[0] == '#':
            print 'skip line starts with #, skip to next'
            continue

        rlt = {}
        rlt["filename"] = imgpath
        rlt["faces"] = []
        rlt['face_count'] = 0

        try:
            img = cv2.imread(imgpath)
        except:
            print('failed to load image: ' + imgpath)
            rlt["message"] = "failed to load"
            result_list.append(rlt)
            continue

        if img is None:
            print('failed to load image: ' + imgpath)

            rlt["message"] = "failed to load"
            result_list.append(rlt)
            continue

        img_cnt += 1
        t1 = time.clock()

        bboxes, points = detector.detect_face(img, minsize,
                                              threshold, scale_factor)

        t2 = time.clock()
        ttl_time += t2 - t1
        print("detect_face() costs %f seconds" % (t2 - t1))

        if bboxes is not None and len(bboxes)>0:
            for (box, pts) in zip(bboxes, points):
#                box = box.tolist()
#                pts = pts.tolist()
                tmp = {'rect': box[0:4],
                       'score': box[4],
                       'pts': pts
                       }
                rlt['faces'].append(tmp)

            rlt['face_count'] = len(bboxes)

        rlt['message'] = 'success'
        result_list.append(rlt)

#        print('output bboxes: ' + str(bboxes))
#        print('output points: ' + str(points))
        # toc()

        print("\n===> Processed %d images, costs %f seconds, avg time: %f seconds" % (
            img_cnt, ttl_time, ttl_time / img_cnt))

        if bboxes is None:
            continue


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

    json.dump(result_list, fp_rlt, indent=4)
    fp_rlt.close()
    fp.close()

    if show_img:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print_usage()

    img_list_fn = "./list_img.txt"
    save_dir = './fd_rlt4'

    print(sys.argv)

    if len(sys.argv) > 1:
        img_list_fn = sys.argv[1]

    if len(sys.argv) > 2:
        save_dir = sys.argv[2]

    if len(sys.argv) > 3:
        show_img = not(not(sys.argv[3]))

    main(img_list_fn, save_dir, show_img=False)
