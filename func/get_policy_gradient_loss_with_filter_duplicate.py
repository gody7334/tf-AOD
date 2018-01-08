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
argmax_iou_input = tf.placeholder(shape=[None,None],dtype=tf.int64)
target_class_input = tf.placeholder(shape=[None,None],dtype=tf.int64)
target_bbox_input = tf.placeholder(shape=[None,None,4],dtype=tf.int64)
target_bbox_input = tf.placeholder(shape=[None,4],dtype=tf.float32)

batch_size = 6

# to use for maximum likelihood with input location
def gaussian_pdf(mean, sample):
    Z = 1.0 / (loc_sd * tf.sqrt(2.0 * np.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
    return Z * tf.exp(a)

# build graph
# def get_policy_gradient_loss(batch_size,
                                # argmax_iou_input,
                                # target_class_input,
                                # target_bbox_input,
                                # predict_class_input,
                                # predict_bbox_input,
                                # mean_location_input,
                                # sample_location_input,
                                # baseline_input):
    '''
    compute policy reward and function approximator (loss)
    filter duplicate argmax_iou index along each batch 
    to avoid predict same proposal bbox to collect same reward

    @ param
    batch_size (int)
    argmax_iou_input (N,T) the index of max iou within bboxes
    target_class_input (N,T) the target class, which selected using argmax_iou in each episode
    target_bbox_input (N,T,4) the target bbox, which selected using argmax_iou in each episode
    predict_class_input (N,T) the predict class, unnormalized logit, in each episode
    predict_bbox_input (N,T,4) the predict bbox, in each episode
    
    mean_location_input (N,T,4)
    sample_location_input (N,T,4)
    baseline_input (N,T,4)

    @ return
    policy_gradient_loss (N), batches of iou loss between predict and target bbox
    '''
# unstack along each batch
argmax_iou_list = tf.unstack(argmax_iou_input, num=batch_size, axis = 0)
target_class_list = tf.unstack(target_class_input, num=batch_size, axis = 0)
target_bbox_list = tf.unstack(target_bbox_input, num=batch_size, axis = 0)
predict_class_list = tf.unstack(predict_class_input, num=batch_size, axis = 0)
predict_bbox_list = tf.unstack(predict_bbox_input, num=batch_size, axis = 0)

unique_iou_index_array = []
for i in range(batch_size):
    iou_index = argmax_iou_list[i]
    target_class = target_class_list[i]
    target_bbox = target_bbox_list[i]
    predict_class = predict_class_list[i]
    predict_bbox = predict_bbox_list[i]
    
    unique_iou_index, _ = tf.unique(iou_index[i])
    unique_target_class = tf.gather_nd(target_class_input, unique_iou_index)
    unique_target_bbox = tf.gather_nd(target_bbox_input, unique_iou_index)
    unique_iou_index_array.append(unique_iou_inde)

# run graph
with tf.Session() as sess:
    while(True):
        # assign variable (design test case)
        argmax_iou_ = np.random.randint(10,size=(batch_size,10))
        target_class_ = np.random.randint(10,size=(batch_size,10))
        target_bbox_ = np.random.rand

        pp(argmax_iou_)

        sess.run(tf.initialize_all_variables())
        print(sess.run(argmax_iou_list,feed_dict={argmax_iou_input:argmax_iou_}))
        print(sess.run(target_class_list,feed_dict={argmax_iou_input:argmax_iou_}))
        print(sess.run(target_bbox_list,feed_dict={argmax_iou_input:argmax_iou_}))
        print(sess.run(unique_iou_index_array,feed_dict={argmax_iou_input:argmax_iou_}))

        ipdb.set_trace()
        print("a")
