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
iou_input = tf.placeholder(shape=[None,None],dtype=tf.float32)
class_input = tf.placeholder(shape=[None,None],dtype=tf.int64)
bbox_input = tf.placeholder(shape=[None,None,4],dtype=tf.float32)

# build graph

def get_argmax_ious_class_bbox(iou_input, class_input, bbox_input):
    '''
    Prepare target(index in T, class, bbox) in each episode
    ps: prepare target should stop gradient
    
    @ param
    iou_input (N,T) all iou between predict and target bbox pair
    class_input (N,T) all classes in images
    bbox_input (N,T,4) all bboxes in images
    
    @ return
    argmax_ious (N) max ious index (within T) (-1 if ious < 0.5, represent background)
    class (N) max ious class (insert background class)
    bbox (N,4) max ious bbox (0,0,0,0 for background bbox)
    '''
    
    # filter iou < 0.5
    mask = tf.greater(iou_input,0.5)
    ious = tf.multiply(iou_input,tf.cast(mask, tf.float32))
    
    # select max and its index
    max_ious = tf.reduce_max(ious,1)
    argmax_ious = tf.argmax(ious,1)
    
    # filter max_ious = 0, let index = -1 as Background
    obj_mask = tf.cast(tf.not_equal(max_ious,0.0), tf.int64)
    back_mask = tf.cast(tf.equal(max_ious,0.0), tf.int64)
    # argmax_ious = argmax_ious - mask
    
    # get max_iou index
    N = 4
    num_class = 100
    n_idx = tf.expand_dims(tf.cast(tf.range(N),tf.int64),1)
    argmax_ious = tf.expand_dims(argmax_ious,1)
    n_idx = tf.concat([n_idx,argmax_ious],1)
    
    # get max_iou class, bbox
    classes = tf.gather_nd(class_input, n_idx)*obj_mask + back_mask*num_class
    bboxes = tf.gather_nd(bbox_input, n_idx)*tf.cast(tf.expand_dims(obj_mask,1),tf.float32)
    argmax_ious = tf.squeeze(argmax_ious)
    
    return (argmax_ious, classes, bboxes)

argmax_ious, classes, bboxes = get_argmax_ious_class_bbox(iou_input, class_input, bbox_input)

# run graph
with tf.Session() as sess:
    while(True):
        # assign variable (design test case)
        iou = np.random.rand(4,3)
        class_ = np.random.randint(10,size=(4,3))
        bbox = np.random.randint(10,size=(4,3,4))
        pp(iou)
        pp(class_)
        pp(bbox)

        sess.run(tf.initialize_all_variables())
        print(sess.run(argmax_ious,feed_dict={iou_input:iou,class_input:class_,bbox_input:bbox}))
        print(sess.run(classes,feed_dict={iou_input:iou,class_input:class_,bbox_input:bbox}))
        print(sess.run(bboxes,feed_dict={iou_input:iou,class_input:class_,bbox_input:bbox}))

        ipdb.set_trace()
        print("a")
