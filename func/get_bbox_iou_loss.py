import os.path as osp
import sys
import ipdb
import colored_traceback.always

# Add module path
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

package_path = osp.join(this_dir, '..')
add_path(package_path)

import tensorflow as tf
import numpy as np
from pprint import pprint as pp
from graph.forward.Iforward import IForward
from utils.config import global_config

global_config.assign_config()
iforward = IForward('train',None)

# define variable
predict_bbox_input = tf.placeholder(shape=[None,4],dtype=tf.float32)
target_bbox_input = tf.placeholder(shape=[None,4],dtype=tf.float32)

# build graph
def get_bbox_iou_loss(predict_bbox_input, target_bbox_input):
    '''
    compute iou then -log(iou) to get log losses scale
    as iou within 0~1 the -log loss within infinte ~ 0
    ps: deal with infinte value by add small value to iou

    @ param
    predict_bbox_input (N,4) with faster-rcnn def (xmid, ymid, w, h)
    target_bbox_input (N,4) with mscoco def (xmin, ymin, w, h)

    @ return
    iou_losses (N), batches of iou loss between predict and target bbox
    ps: after add 1e-10, losses are within 23~0
    however, as filter out iou < 0.5 to let it as background(bbox = (0,0,0,0) ), it will cause
    uncontinuous losses, might have potential problem,
    on the other hand, if we don't count background bbox losses,
    model will prefer get background bbox rather than object bbox, which is not we want.
    '''
    ious = iforward.get_iou(tf.expand_dims(predict_bbox_input,1),tf.expand_dims(target_bbox_input,1))
    ious = tf.squeeze(ious,[1,2])
    iou_losses = -1.0 * tf.log(tf.clip_by_value(ious,1e-10,1))

    return iou_losses

iou_losses = get_bbox_iou_loss(predict_bbox_input, target_bbox_input)

# run graph
with tf.Session() as sess:
    while(True):
        # assign variable (design test case)
        predict_bbox = np.random.rand(10,4)
        target_bbox = np.random.rand(10,4)
        pp(predict_bbox)
        pp(target_bbox)

        sess.run(tf.initialize_all_variables())
        print(sess.run(iou_losses,feed_dict={predict_bbox_input:predict_bbox,target_bbox_input:target_bbox}))

        ipdb.set_trace()
        print("a")
