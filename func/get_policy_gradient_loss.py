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
target_class_input = tf.placeholder(shape=[None],dtype=tf.int64)
target_bbox_input = tf.placeholder(shape=[None,4],dtype=tf.float32)
predict_class_input = tf.placeholder(shape=[None,None],dtype=tf.float32)
predict_bbox_input = tf.placeholder(shape=[None,4],dtype=tf.float32)
mean_location_input = tf.placeholder(shape=[None,4],dtype=tf.float32)
sample_location_input = tf.placeholder(shape=[None,4],dtype=tf.float32)
baseline_input = tf.placeholder(shape=[None,4],dtype=tf.float32)

num_class_input = 10

# build graph
def get_policy_gradient_loss(num_class_input,
                                target_class_input,
                                target_bbox_input,
                                predict_class_input,
                                predict_bbox_input,
                                mean_location_input,
                                sample_location_input,
                                baseline_input):
    '''
    compute policy reward and function approximator (loss) in each time step

    @ param
    num_class (int) number of class (including background) for one hot encoding
    target_class_input (N) the target class label, which selected using argmax_iou in each time step
    target_bbox_input (N,4) the target bbox, which selected using argmax_iou in each time step, mscoco fomat
    predict_class_input (N,num_class) the predict class, unnormalized logit, in each time step
    predict_bbox_input (N,4) the predict bbox, in each time step, frcnn format
    mean_location_input (N,4)
    sample_location_input (N,4)
    baseline_input (N,4)

    @ return
    policy_gradient_loss (N), batches of iou loss between predict and target bbox
    '''

    # to use for maximum likelihood with input location
    def gaussian_pdf(mean, sample):
        loc_sd = 0.11
        Z = 1.0 / (loc_sd * tf.sqrt(2.0 * np.pi))
        a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
        return Z * tf.exp(a)

    target_class_one_hot = tf.one_hot(target_class_input, num_class_input)
    predict_class_prob = tf.nn.softmax(predict_class_input)
    predict_target_prob = tf.reduce_sum(predict_class_prob*target_class_one_hot, axis=1)
    iou = iforward.get_iou(tf.expand_dims(target_bbox_input,axis=1),tf.expand_dims(predict_bbox_input,axis=1))

    # rewards
    rewards = tf.squeeze(iou,[1,2]) * predict_target_prob

    # baseline
    no_grad_b = tf.stop_gradient(baseline_input)

    # construct schocastic policy using mean and sample location
    p_loc = gaussian_pdf(mean_location_input, sample_location_input)
    p_loc = tf.tanh(p_loc)

    # likelihood estimator
    rewards = tf.tile(tf.expand_dims(rewards,[1]), [1,4])
    J = tf.log(p_loc + 1e-10) * (rewards - no_grad_b)

    J = tf.reduce_sum(J, 1)
    J = J - tf.reduce_sum(tf.square(rewards - baseline_input), 1)
    J = tf.reduce_mean(J, 0)
    return -J

pg_loss = get_policy_gradient_loss(num_class_input,
                                    target_class_input,
                                    target_bbox_input,
                                    predict_class_input,
                                    predict_bbox_input,
                                    mean_location_input,
                                    sample_location_input,
                                    baseline_input)

# run graph
with tf.Session() as sess:
    while(True):
        # assign variable (design test case)
        target_class_ = np.random.randint(num_class_input,size=(6))
        predict_class_ = np.random.rand(6,num_class_input)
        target_bbox_ = np.random.rand(6,4)
        predict_bbox_ = np.random.rand(6,4)
        baseline_ = np.random.rand(6,4)
        mean_location_ = np.random.rand(6,4)
        sample_location_ = np.random.rand(6,4)

        # pp(target_class_)
        # pp(predict_class_)
        # pp(target_bbox_)
        # pp(predict_bbox_)
        # pp(baseline_)
        # pp(mean_location_)
        # pp(sample_location_)

        sess.run(tf.initialize_all_variables())

        feed_dict_ = {target_class_input:target_class_,
                        predict_class_input:predict_class_,
                        target_bbox_input:target_bbox_,
                        predict_bbox_input:predict_bbox_,
                        baseline_input:baseline_,
                        mean_location_input:mean_location_,
                        sample_location_input:sample_location_
        }

        # print(sess.run(target_class_one_hot,feed_dict=feed_dict_))
        # print(sess.run(predict_class_prob,feed_dict=feed_dict_))
        # print(sess.run(predict_target_prob,feed_dict=feed_dict_))
        # print(sess.run(iou,feed_dict=feed_dict_))
        # print(sess.run(rewards,feed_dict=feed_dict_))
        # print(sess.run(no_grad_b,feed_dict=feed_dict_))
        # print(sess.run(p_loc,feed_dict=feed_dict_))
        # print(sess.run(rb,feed_dict=feed_dict_))
        # print(sess.run(log_loc,feed_dict=feed_dict_))
        # print(sess.run(J,feed_dict=feed_dict_))
        # print(sess.run(cost,feed_dict=feed_dict_))
        print(sess.run(pg_loss,feed_dict=feed_dict_))


        ipdb.set_trace()
        print("a")
