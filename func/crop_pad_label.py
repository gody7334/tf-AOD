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
label_input = tf.placeholder(shape=[None],dtype=tf.float32)

# build graph
def crop_pad_label(label, target_length, pad_value=0):
    '''
    crop or pad label into fix length for rnn batch training

    @ param
    label (T,?), should provide T
    target_lengt
    pad_value

    @ return
    label (target_length, ?)
    '''
    label = label_input

    from tensorflow.python.framework import ops
    from tensorflow.python.ops import variables
    from tensorflow.python.ops import math_ops

    def _is_tensor(x):
        """Returns `True` if `x` is a symbolic tensor-like object.
        Args:     x: A python object to check.
        Returns:     `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
        """
        return isinstance(x, (ops.Tensor, variables.Variable))

    def max_(x, y):
        if _is_tensor(x) or _is_tensor(y):
            return math_ops.maximum(x, y)
        else:
            return max(x, y)

    def min_(x, y):
        if _is_tensor(x) or _is_tensor(y):
            return math_ops.minimum(x, y)
        else:
            return min(x, y)

    def equal_(x, y):
        if _is_tensor(x) or _is_tensor(y):
            return math_ops.equal(x, y)
        else:
            return x == y

    label = tf.cond(tf.rank(label) < 2,
            lambda: tf.expand_dims(label,1),
            lambda: tf.identity(label))

    # maybe crop
    label_length = tf.shape(label)[0]
    label = tf.slice(label, [0,0], [min_(label_length, target_length),-1])

    #maybe pad
    diff = tf.subtract(target_length,label_length)
    num_pad = max_(diff,0)
    padding = tf.stack([[0,num_pad],[0,0]])
    label = tf.pad(label,padding)

    label = tf.squeeze(label)
    return label

label = crop_pad_label(label_input, 10, pad_value=0)

# run graph
with tf.Session() as sess:
    while(True):
        # assign variable (design test case)
        label_ = np.random.rand(7)
        pp(label_)

        sess.run(tf.initialize_all_variables())
        print(sess.run(label,feed_dict={label_input:label_}))

        ipdb.set_trace()
        print("a")
