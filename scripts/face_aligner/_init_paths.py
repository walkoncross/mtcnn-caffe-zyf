import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.abspath('../../face_aligner'))

#add caffe path
# add_path('/opt/caffe')
# add_path('/disk2/zhaoyafei/caffe-bvlc-for-mxnet-container')