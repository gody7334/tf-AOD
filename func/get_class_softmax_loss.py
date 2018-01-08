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
logit_input = tf.placeholder(shape=[None,None],dtype=tf.float32)
label_input = tf.placeholder(shape=[None],dtype=tf.int64)

# build graph

def get_class_softmax_loss(logit_input, label_input):
    '''
    tf.nn.sparse_softmax_cross_entropy_with_logits
    will apply: softmax to logit_input, one_hot encoded to label_input
    then compute cross-entropy between (above) two value to get losses

    @ param
    logit_input (N,num_class) contains computed scores (unnormalized log probabilities) for each class
    label_input (N) True class, values are within [0,num_class-1]

    @ return
    losses (N), batches of softmax loss between logit and label
    '''
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_input, logits=logit_input)
    return losses


losses = get_class_softmax_loss(logit_input, label_input)

# run graph
with tf.Session() as sess:
    while(True):
        # assign variable (design test case)
        logit = np.random.rand(4,10)
        label = np.random.randint(10,size=(4))
        pp(logit)
        pp(label)

        sess.run(tf.initialize_all_variables())
        print(sess.run(losses,feed_dict={logit_input:logit,label_input:label}))

        ipdb.set_trace()
        print("a")
