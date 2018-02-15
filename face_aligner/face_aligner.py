# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:48:19 2017

@author: zhaoy
"""
import cv2
import numpy as np

from mtcnn_aligner import MtcnnAligner
from fx_warp_and_crop_face import warp_and_crop_face, get_reference_facial_points


class FaceAligner:
    def __init__(self, caffe_model_path=None, gpu_id=0):
        self.aligner = None
        if caffe_model_path:
            self.aligner = MtcnnAligner(caffe_model_path, gpu_id)

    def align_face(self, img, face_rects):
        if isinstance(img, str):
            img = cv2.imread(img)

        regressed_rects, facial_points = self.aligner.align_face(
            img, face_rects)

        return (regressed_rects, facial_points)

    def get_face_chips(self, img, face_rects, facial_points=None, output_square=False):
        if facial_points is None:
            if self.aligner is None:
                raise Exception('FaceAligner.aligner is not initialized')

            rects, facial_points = self.aligner.align_face(
                img, face_rects)

        reference_5pts = None
        output_size = (96, 112)  # (w, h) not (h,w)
        if output_square:
            output_size = (112, 112)
            reference_5pts = get_reference_facial_points(
                output_size)

        face_chips = []
        for facial_5pts in facial_points:
            facial_5pts = np.reshape(facial_5pts, (2, -1))
            dst_img = warp_and_crop_face(
                img, facial_5pts, reference_5pts, output_size)
            face_chips.append(dst_img)

        return face_chips


if __name__ == "__main__":
    import os
    import os.path as osp
#    import json
    import time
    from mtcnn_aligner import draw_faces

    show_img = True

    caffe_model_path = '../model'
    gpu_id = 0
    save_dir = './face_chips'
#    save_json = 'mtcnn_align_test_rlt.json'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    img_path = '../test_imgs/Marilyn_Monroe_0002.jpg'
    face_rect1 = [
        [
            91,
            57
        ],
        [
            173,
            57
        ],
        [
            173,
            180
        ],
        [
            91,
            180
        ]
    ]

    face_rects = [face_rect1]

    base_name = osp.basename(img_path)
    name, ext = osp.splitext(base_name)
#    ext = '.png'

#    fp_rlt = open(osp.join(save_dir, save_json), 'w')
#    results = []

    img = cv2.imread(img_path)

    aligner = FaceAligner(caffe_model_path, gpu_id)

    t1 = time.clock()

    # You can align the faces in two steps like this:
#    bboxes, points = aligner.align_face(img, face_rects)
#    face_chips = aligner.get_face_chips(img, bboxes, points)

    # OR just align them in one step by calling the following function,
    # which combine last two functions
    face_chips = aligner.get_face_chips(img, face_rects)
    t2 = time.clock()

    for i, chip in enumerate(face_chips):

        save_name = osp.join(save_dir, name + '_chip' + str(i) + ext)
        cv2.imwrite(save_name, chip)

        if show_img:
            cv2.imshow('face_chip', chip)

            cv2.waitKey(0)

    n_boxes = len(face_rects)
    print("-->Alignment cost %f seconds, processed %d face rects, avg time: %f seconds" %
          ((t2 - t1), n_boxes, (t2 - t1) / n_boxes))

#    rlt = {}
#    rlt["filename"] = img_path
#    rlt["faces"] = []
#    rlt['face_count'] = 0
#
#    if bboxes is not None and len(bboxes) > 0:
#        for (box, pts) in zip(bboxes, points):
#            #                box = box.tolist()
#            #                pts = pts.tolist()
#            tmp = {'rect': box[0:4],
#                   'score': box[4],
#                   'pts': pts
#                   }
#            rlt['faces'].append(tmp)
#
#    rlt['face_count'] = len(bboxes)
#
#    rlt['message'] = 'success'
#    results.append(rlt)
#
#    json.dump(results, fp_rlt, indent=4)
#    fp_rlt.close()

#    draw_faces(img, bboxes, points)
#
#    save_name = osp.join(save_dir, name + ext)
#    cv2.imwrite(save_name, img)
#
#    if show_img:
#        cv2.imshow('img', img)
#
#        cv2.waitKey(0)
#
    if show_img:
        cv2.destroyAllWindows()
