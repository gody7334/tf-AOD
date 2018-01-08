import tensorflow as tf
import numpy as np
from pprint import pprint as pp

ph = tf.placeholder(shape=[None,None,4], dtype=tf.float32)

def convert_coordinate(x, type_from, type_to, dim=3):
    '''
    type:
    bmp:(xmin, ymin, xmax, ymax)
    tf:(ymin, xmin, ymax, xmax)
    mscoco:(xmin, ymin, w, h)
    frcnn:(xmid, ymid, w, h)
    '''
    if dim == 2:
        x = tf.expand_dims(x,1)

    x1 = tf.slice(x, [0,0,0],[-1,-1,1])
    x2 = tf.slice(x, [0,0,1],[-1,-1,1])
    x3 = tf.slice(x, [0,0,2],[-1,-1,1])
    x4 = tf.slice(x, [0,0,3],[-1,-1,1])

    if type_from == "frcnn" and type_to == "bmp":
        x1 = x1-x3/2.0
        x2 = x2-x4/2.0
        x3 = x1+x3
        x4 = x2+x4
        x = tf.concat([x1,x2,x3,x4],-1)

    if type_from == "bmp" and type_to == "frcnn":
        x3 = x3-x1
        x4 = x4-x2
        x1 = x1 + x3/2.0
        x2 = x2 + x4/2.0
        x = tf.concat([x1,x2,x3,x4],-1)

    if type_from == "mscoco" and type_to == "bmp":
        x3 = x1+x3
        x4 = x2+x4
        x = tf.concat([x1,x2,x3,x4],-1)

    if type_from == "bmp" and type_to == "mscoco":
        x3 = x3-x1
        x4 = x4-x2
        x = tf.concat([x1,x2,x3,x4],-1)

    if type_from == "frcnn" and type_to == "mscoco":
        x1 = x1-x3/2.0
        x2 = x2-x4/2.0
        x = tf.concat([x1,x2,x3,x4],-1)

    if type_from == "mscoco" and type_to == "frcnn":
        x1 = x1+x3/2.0
        x2 = x2+x4/2.0
        x = tf.concat([x1,x2,x3,x4],-1)

    if dim == 2:
        x = tf.squeeze(x,[1])

    return x

# look the -1 in the first position

from_ = "bmp"
to_ = "frcnn"
x = convert_coordinate(ph, from_, to_, dim=3)

input_ = np.array([[[1,2,3,4],[3,4,5,6],[5,6,7,8]]], dtype=np.float32)
# input_ = np.random.rand(3,2,4)
pp(input_)
print(from_ +' => '+ to_)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(x,feed_dict={ph:input_}))
