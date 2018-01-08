import os.path as osp
import sys
import ipdb

# Add module path
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

package_path = osp.join(this_dir, '.')
add_path(package_path)


import tensorflow as tf
import numpy as np
from pprint import pprint as pp
import convert_coordinate

# define variable
pred_bboxes_input = tf.placeholder(shape=[None,4],dtype=tf.float32)
targ_bboxes_input = tf.placeholder(shape=[None,None,4], dtype=tf.float32)

# build grap

def get_iou(pred_bboxes, targ_bboxes):
    '''
    targ_bboxes (N,T,4)
    pred_bboxes (N,T,4) or
                (N,4)-> expand_dim and broadcast, used to find max iou in target
    return iou (N,T,4)
    '''
    pred_bboxes = tf.cond(tf.rank(pred_bboxes) < 3,
                lambda: tf.expand_dims(pred_bboxes,1),
                lambda: tf.identity(pred_bboxes))

    pred_w = tf.slice(pred_bboxes,[0,0,2],[-1,-1,1])
    pred_h = tf.slice(pred_bboxes,[0,0,3],[-1,-1,1])
    targ_w = tf.slice(targ_bboxes,[0,0,2],[-1,-1,1])
    targ_h = tf.slice(targ_bboxes,[0,0,3],[-1,-1,1])
    pred_bboxes = convert_coordinate.convert_coordinate(pred_bboxes, "frcnn","bmp",dim=3)
    targ_bboxes = convert_coordinate.convert_coordinate(targ_bboxes, "frcnn","bmp",dim=3)

    pred_xmin = tf.slice(pred_bboxes,[0,0,0],[-1,-1,1])
    pred_ymin = tf.slice(pred_bboxes,[0,0,1],[-1,-1,1])
    pred_xmax = tf.slice(pred_bboxes,[0,0,2],[-1,-1,1])
    pred_ymax = tf.slice(pred_bboxes,[0,0,3],[-1,-1,1])

    targ_xmin = tf.slice(targ_bboxes,[0,0,0],[-1,-1,1])
    targ_ymin = tf.slice(targ_bboxes,[0,0,1],[-1,-1,1])
    targ_xmax = tf.slice(targ_bboxes,[0,0,2],[-1,-1,1])
    targ_ymax = tf.slice(targ_bboxes,[0,0,3],[-1,-1,1])

    yA = tf.maximum(pred_ymin, targ_ymin)
    xA = tf.maximum(pred_xmin, targ_xmin)
    yB = tf.minimum(pred_ymax, targ_ymax)
    xB = tf.minimum(pred_xmax, targ_xmax)

    inter_Area = tf.maximum(0.0,(xB-xA))*tf.maximum(0.0,(yB-yA))
    pred_Area = pred_w*pred_h
    targ_Area = targ_w*targ_h
    iou = inter_Area / tf.clip_by_value((pred_Area + targ_Area - inter_Area),1e-10, 1)

    return iou

iou = get_iou(pred_bboxes_input, targ_bboxes_input)

# assign variable (design test case)
# input_ = np.array([[1,2,3,4],[3,4,5,6],[5,6,7,8]], dtype=np.float32)
pred = np.random.rand(3,3,4)
targ = np.random.rand(3,3,4)

pp(pred)
pp(targ)

# run graph
with tf.Session() as sess:
    while(True):
        pred = np.random.rand(3,4)
        targ = np.random.rand(3,3,4)

        pp(pred)
        pp(targ)

        sess.run(tf.initialize_all_variables())
        print(sess.run(iou,feed_dict={pred_bboxes_input:pred,targ_bboxes_input:targ}))

        ipdb.set_trace()
        print("a")
