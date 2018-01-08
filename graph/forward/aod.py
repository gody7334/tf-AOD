from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pprint import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.roi_pooling_layer import roi_pooling_op
from utils.roi_pooling_layer import roi_pooling_op_grad
import ipdb

from utils.config import global_config
from graph.ops import image_embedding
from graph.ops import image_processing
from graph.ops import inputs as input_ops
from graph.forward.Iforward import IForward
from graph.ops.debug import _debug_func

class AOD(IForward):
    def __init__(self, mode, data):
        IForward.__init__(self, mode, data)
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
                as its dimension is [N 14 14 512]
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        # TODO: move to global config
        # self.word_to_idx = word_to_idx
        # self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        # self._start = word_to_idx['<START>']
        # self._null = word_to_idx['<NULL>']
        # self.V = len(word_to_idx)

        self.prev2out = False
        self.ctx2out = True
        self.alpha_c = 0.0
        self.selector = True
        self.dropout = True
        # dimension of vggnet19 conv5_3 is [N 14 14 512]
        # self.L = 196
        # self.D = 512
        self.L = None # depend on desired inception layer
        self.D = None # depend on desired inception layer
        self.M = 512 # word embedded
        self.H = self.config.num_lstm_units # hidden stage
        self.T = 10 # Time step size of LSTM (how many predicting bboxes in a image)
        self.B = 4 # bbox dimension

        self.NN = None # batch size
        self.WW = None # feature width
        self.HH = None # feature high
        self.DD = None # feature depth,(dimension)

        self.glimpse_w = 7 # glimpse width
        self.glimpse_h = 7 # glimpse high

        # Place holder for features and captions
        # self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        # self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.features = None
        self.captions = None
        self.logits = None
        self.targets = None
        self.weights = None

        self.loc_sd = 0.1
        self.roises_list = [] # RoI location, used to reconstuct the bbox coordinate in a image
        self.baselines_list = [] # policy baseline
        self.mean_locs_list = [] # expected location
        self.sampled_locs_list = [] # (agent) random sample from gaussion distribution
        self.bboxes_list = [] # predict bbox
        self.class_logits_list = [] # predict class
        self.rewards_list = []

        self.region_proposals = None
        self.glimpses = None
        self.glimpses_project = None
        self.rois = None #
        self.roi_pooling_h = 7
        self.roi_pooling_w = 7

        # for debug
        self.predict_bbox_full = None
        self.iouses = None
        self.argmax_ious = None
        self.target_class = None
        self.target_bbox = None
        self.predict_bbox = None
        self.predict_class_logit = None
        self.rois = None
        self.class_losses = None
        self.bbox_losses = None
        self.policy_losses = None
        self.batch_loss = None

        # self._gen_first_region_proposal()
        self.build_image_embeddings()
        self.get_features_shape()
        self.build_model()
        self.get_loss()

    def build_seq_embeddings(self):
        return

    def build_image_embeddings(self):
        """ Load inception V3 graph (IForward) and post process the image feature map
        Inputs:
          self.images
        Outputs:
          self.image_embeddings
        """
        # call parent function to get original image features map
        super(AOD, self).build_image_embeddings()

        # TOOD experiement different layer
        inception_layer = self.inception_end_points['Mixed_7c']

        # get depth of image embedded
        layer_shape = inception_layer.get_shape().as_list()
        self.D = layer_shape[3]
        self.L = layer_shape[1]*layer_shape[2]

        # RoI pooling need to retain W,H dim, no reshape needed
        self.features = inception_layer

    def get_features_shape(self):
        feature_shape = self.features.get_shape().as_list()
        self.NN = feature_shape[0]
        self.WW = feature_shape[1]
        self.HH = feature_shape[2]
        self.DD = feature_shape[3]

    def _project_glimpses(self, glimpses, reuse=False):
        with tf.variable_scope('project_glimpses', reuse=reuse):
            w = tf.get_variable('w', [self.glimpse_w*self.glimpse_h*self.DD, self.H],
                    initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.H], initializer=self.const_initializer)
            glimpses_flat = tf.reshape(glimpses, [self.NN, -1])
            glimpses_proj = tf.matmul(glimpses_flat, w) + b
            return glimpses_proj

    def _attension_region_proposal_layer(self, h=None, reuse=False):
        # Here bbox def: (xmid,ymid,w,h) as easy to limit the boundry(0~1) and avoid revert coor
        with tf.variable_scope('attension_region_proposal_layer',reuse=reuse):
            baseline_w = tf.get_variable('baseline_w', [self.H, self.B],initializer=self.weight_initializer)
            baseline_b = tf.get_variable('baseline_b', [self.B], initializer=self.const_initializer)
            mean_w = tf.get_variable('mean_w', [self.H, self.B],initializer=self.weight_initializer)
            mean_b = tf.get_variable('mean_b', [self.B],initializer=self.const_initializer)
            if(reuse==False):
                # first set don't have baseline, mean_loc, sample_loc
                # therefore there is no reward in the first step
                rp = np.array([0.5,0.5,1.0,1.0])
                rp = tf.expand_dims(tf.convert_to_tensor(rp),0)
                rps = tf.tile(rp,[self.config.batch_size,1])
                return rps
            else:
                # train a baseline_beline function
                # baseline might out of boundry.
                baseline = tf.sigmoid(tf.matmul(h,baseline_w)+baseline_b)
                self.baselines_list.append(baseline)

                # compute next location
                # mean_loc might out of boundry
                mean_loc = tf.matmul(h,mean_w)+mean_b
                # mean_loc = tf.stop_gradient(mean_loc)
                self.mean_locs_list.append(mean_loc)

                # add noise - sample from guassion distribution
                # as agent to decide the mean_loc vs sample_los is good or bad
                sample_loc = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, self.loc_sd)

                # Need to clip sample_loc to get the right features
                # first clip for invalid (xmin, ymid, w, h)
                sample_loc = tf.maximum(1e-10, tf.minimum(1.0, sample_loc))

                # second clip for invalid (xmin, ymin, xmax, ymax)
                sample_loc = self._convert_coordinate(sample_loc, "frcnn","bmp",dim=2)
                sample_loc = tf.maximum(1e-10, tf.minimum(1.0, sample_loc))
                sample_loc = self._convert_coordinate(sample_loc, "bmp", "frcnn",dim=2)
                # after clipping xmid, ymid will shift too.

                # sample_loc = tf.stop_gradient(sample_loc)
                self.sampled_locs_list.append(sample_loc)

                return sample_loc

    def _ROI_pooling_layer(self, features, region_proposal):
        region_proposal = self._convert_coordinate(region_proposal, "frcnn", "bmp",dim=2)

        # convert from (0-1) to int coordinate
        xmin = tf.slice(region_proposal, [0,0],[-1,1])
        ymin = tf.slice(region_proposal, [0,1],[-1,1])
        xmax = tf.slice(region_proposal, [0,2],[-1,1])
        ymax = tf.slice(region_proposal, [0,3],[-1,1])

        n_idx = tf.expand_dims(tf.cast(tf.range(self.NN),tf.float32),1)
        xmin = tf.cast(tf.floor(xmin*self.WW),tf.float32) #(n,1)
        ymin = tf.cast(tf.floor(ymin*self.HH),tf.float32) #(n,1)
        xmax = tf.cast(tf.ceil(xmax*self.WW),tf.float32)  #(n,1)
        ymax = tf.cast(tf.ceil(ymax*self.HH),tf.float32)  #(n,1)
        rois = tf.concat([n_idx,xmin,ymin,xmax,ymax],1) #(n,5)

        # Q: pooling from 1x1 to 7x7? that all value is the same in 7x7 as region proposal is small?

        [y, argmax] = roi_pooling_op.roi_pool(features, rois, self.roi_pooling_w, self.roi_pooling_h, 1.0/3)

        # store rois for convert the coordinate between region <=> full image
        rois = tf.concat([xmin,ymin,xmax,ymax],1)
        rois = self._convert_coordinate(rois, "bmp", "frcnn",dim=2)
        self.roises_list.append(rois)

        return y

    def _decode_lstm_bbox_class(self, h, reuse=False):
        '''
        @return
        bboxes (N,4) with faster-rcnn def (xmid, ymid, w, h)
        '''
        with tf.variable_scope('decode_bbox_class', reuse=reuse):
            w_bbox = tf.get_variable('w_bbox', [self.H, 4], initializer=self.weight_initializer)
            b_bbox = tf.get_variable('b_bbox', [4], initializer=self.const_initializer)

            # number of class should need to add background
            w_class = tf.get_variable('w_class', [self.H, self.config.num_classes+1], initializer=self.weight_initializer)
            b_class = tf.get_variable('b_class', [self.config.num_classes+1], initializer=self.const_initializer)

            # bboxes (xmid, ymid, w, h), to prevent inverse bbox
            bboxes = tf.maximum(0.0, tf.minimum(1.0,tf.matmul(h, w_bbox) + b_bbox))
            # here only unnormalize log probabilities (logits, score) for each class, need softmax & argmax to find the class
            class_logits = tf.matmul(h, w_class) + b_class

            self.bboxes_list.append(bboxes)
            self.class_logits_list.append(class_logits)

            return (bboxes, class_logits)

    def _convert_bbox_to_full_image_coordinate(self,roises,bboxes):
        # Todo convert logit(bbox) result from region to full image coordinate
        rp_xmid = tf.slice(roises, [0,0],[-1,1])
        rp_ymid = tf.slice(roises, [0,1],[-1,1])
        rp_w = tf.slice(roises, [0,2],[-1,1])
        rp_h = tf.slice(roises, [0,3],[-1,1])

        rp_xmid = rp_xmid/self.WW
        rp_ymid = rp_ymid/self.HH
        rp_w = rp_w/self.WW
        rp_h = rp_h/self.HH

        bbox_xmid = tf.slice(bboxes, [0,0],[-1,1])
        bbox_ymid = tf.slice(bboxes, [0,1],[-1,1])
        bbox_w = tf.slice(bboxes, [0,2],[-1,1])
        bbox_h = tf.slice(bboxes, [0,3],[-1,1])

        bbox_xmid_full = rp_xmid + rp_w*bbox_xmid
        bbox_ymid_full = rp_ymid + rp_h*bbox_ymid
        bbox_w_full = rp_w*bbox_w
        bbox_h_full = rp_h*bbox_h

        bbox_coor_full = tf.concat([bbox_xmid_full,bbox_ymid_full,bbox_w_full,bbox_h_full],1)

        return bbox_coor_full

    def get_argmax_ious_class_bbox(self, iou_input, class_input, bbox_input):
        '''
        Prepare target in each episode
        ps: prepare target should stop gradint
        will compute iou loss later

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
        mask = tf.stop_gradient(tf.greater(iou_input,0.2))
        ious = tf.multiply(iou_input,tf.cast(mask, tf.float32))

        # select max and its index
        max_ious = tf.reduce_max(ious,1)
        argmax_ious = tf.argmax(ious,1)

        # filter max_ious = 0, let index = num_class+1 as Background
        obj_mask = tf.cast(tf.not_equal(max_ious,0.0), tf.int64)
        back_mask = tf.cast(tf.equal(max_ious,0.0), tf.int64)

        # get max_iou index
        N = self.NN
        num_class = self.config.num_classes
        n_idx = tf.expand_dims(tf.cast(tf.range(N),tf.int64),1)
        # argmax_ious = tf.expand_dims(argmax_ious,1)
        n_idx = tf.concat([n_idx,argmax_ious],1)

        # get max_iou class, bbox
        classes = tf.gather_nd(class_input, n_idx)*tf.squeeze(obj_mask) + tf.squeeze(back_mask)*num_class
        bboxes = tf.gather_nd(bbox_input, n_idx)*tf.cast(tf.expand_dims(tf.squeeze(obj_mask),1),tf.float32)
        argmax_ious = tf.squeeze(argmax_ious)

        return (argmax_ious, classes, bboxes)

    def get_class_softmax_loss(self, logit_input, label_input):
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

    def get_bbox_iou_loss(self, predict_bbox_input, target_bbox_input):
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
        ious = self.get_iou(tf.expand_dims(predict_bbox_input,1),tf.expand_dims(target_bbox_input,1))
        ious = tf.squeeze(ious,[1,2])
        iou_losses = -1.0 * tf.log(tf.clip_by_value(ious,1e-10,1))

        return iou_losses

    def get_policy_gradient_loss(self, num_class_input,
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
            Z = 1.0 / (self.loc_sd * tf.sqrt(2.0 * np.pi))
            a = -tf.square(sample - mean) / (2.0 * tf.square(self.loc_sd))
            return Z * tf.exp(a)

        target_class_one_hot = tf.one_hot(target_class_input, num_class_input)
        predict_class_prob = tf.nn.softmax(predict_class_input)
        predict_target_prob = tf.reduce_sum(predict_class_prob*target_class_one_hot, axis=1)
        iou = self.get_iou(tf.expand_dims(target_bbox_input,axis=1),tf.expand_dims(predict_bbox_input,axis=1))

        baseline_input = tf.identity(0.0)

        # rewards
        rewards = (tf.squeeze(iou,[1,2]) * predict_target_prob)
        # rewards = tf.squeeze(iou,[1,2]) * 1e2
        # baseline_input = baseline_input * 1e2

        self.rewards_list = []
        self.rewards_list.append(rewards)

        # baseline
        no_grad_b = tf.stop_gradient(baseline_input)

        # construct schocastic policy using mean and sample location
        p_loc = gaussian_pdf(mean_location_input, sample_location_input)
        p_loc = tf.tanh(p_loc)

        # likelihood estimator
        rewards = tf.tile(tf.expand_dims(rewards,[1]), [1,4])
        J = tf.log(p_loc + 1e-10) * (rewards - baseline_input)

        J = tf.reduce_sum(J, 1)
        # J = J - tf.reduce_sum(tf.square(rewards - baseline_input), 1)
        # J = tf.reduce_mean(J, 0)
        return -J

    def build_model(self):
        features = self.features    #(N,W,H,D)
        batch_size = tf.shape(self.features)[0]
        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')

        h = tf.zeros([self.NN,self.H])
        c = tf.zeros([self.NN,self.H])

        # with tf.variable_scope('initial_lstm'):
            # h = tf.get_variable('h', [self.NN, self.H], initializer=self.emb_initializer)
            # c = tf.get_variable('c', [self.NN, self.H], initializer=self.emb_initializer)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
        region_proposals_list = []

        for t in range(self.T):
            region_proposals = self._attension_region_proposal_layer(h,reuse=(t!=0))
            region_proposals_list.append(region_proposals)

            self.glimpses = self._ROI_pooling_layer(features, region_proposals)
            # self.glimpses = self._batch_norm(self.glimpses, mode='train', name='glimpses_features'+str(t))
            self.glimpses_project = self._project_glimpses(self.glimpses,reuse=(t!=0))

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=self.glimpses_project, state=[c, h])

            self._decode_lstm_bbox_class(h,reuse=(t!=0))


        self.region_proposals = tf.transpose(tf.stack(region_proposals_list),(1,0))



    def get_loss(self):
        # self.roises_list = [] # RoI location, used to reconstuct the bbox coordinate in a image
        # self.baselines_list = [] # policy baseline
        # self.mean_locs_list = [] # expected location
        # self.sampled_locs_list = [] # (agent) random sample from gaussion distribution
        # self.bboxes_list = [] # predict bbox
        # self.class_logits_list = [] # predict class

        class_losses_list = []
        bbox_losses_list = []
        policy_losses_list = []

        for t in range(self.T):
            predict_bbox = self.bboxes_list[t]
            predict_class_logit = self.class_logits_list[t]
            rois = self.roises_list[t]

            predict_bbox_full = self._convert_bbox_to_full_image_coordinate(rois, predict_bbox)
            iouses = self.get_iou(tf.expand_dims(predict_bbox_full,1), self.bbox_seqs)
            argmax_ious, target_class, target_bbox = self.get_argmax_ious_class_bbox(iouses, self.class_seqs, self.bbox_seqs)

            class_losses = self.get_class_softmax_loss(predict_class_logit, target_class)
            bbox_losses = self.get_bbox_iou_loss(predict_bbox,target_bbox)

            class_losses_list.append(class_losses)
            bbox_losses_list.append(bbox_losses)

            if(t > 0):
                mean_location = self.mean_locs_list[t-1]
                sample_location = self.sampled_locs_list[t-1]
                baseline = self.baselines_list[t-1]
                policy_losses = self.get_policy_gradient_loss(self.config.num_classes+1,
                                    target_class,
                                    target_bbox,
                                    predict_class_logit,
                                    predict_bbox,
                                    mean_location,
                                    sample_location,
                                    baseline)
                policy_losses_list.append(policy_losses)

        class_losses = tf.transpose(tf.stack(class_losses_list),(1,0))
        bbox_losses = tf.transpose(tf.stack(bbox_losses_list),(1,0))
        policy_losses = tf.transpose(tf.stack(policy_losses_list),(1,0))
        rewards = tf.transpose(tf.stack(self.rewards_list),(1,0))

        class_loss = tf.reduce_mean(class_losses)
        bbox_loss = tf.reduce_mean(bbox_losses)
        policy_loss = tf.reduce_mean(policy_losses)
        reward = tf.reduce_mean(rewards)

        batch_loss = class_loss + bbox_loss/10 + policy_loss

        batch_loss = self._l2_regularization(batch_loss)

        self.predict_bbox = predict_bbox
        self.predict_class_logit = predict_class_logit
        self.rois = rois
        self.predict_bbox_full = predict_bbox_full
        self.iouses = iouses
        self.argmax_ious = argmax_ious
        self.target_class = target_class
        self.target_bbox = target_bbox
        self.class_losses = class_loss
        self.bbox_losses = bbox_loss
        self.policy_losses = policy_loss
        self.batch_loss = batch_loss

        logits = tf.transpose(tf.stack(self.bboxes_list),(1,0,2))
        self.logits = logits

        image_processing.draw_tf_bounding_boxes(self.images, logits, name="logits")
        image_processing.draw_tf_bounding_boxes(self.images, self.bbox_seqs, name="targets")

        tf.losses.add_loss(batch_loss)
        total_loss = tf.losses.get_total_loss()

        # Add summaries.
        tf.summary.scalar("losses/reward", tf.reduce_mean(reward))
        tf.summary.scalar("losses/class_loss", class_loss)
        tf.summary.scalar("losses/bbox_loss", bbox_loss)
        tf.summary.scalar("losses/policy_loss", policy_loss)
        tf.summary.scalar("losses/batch_loss", batch_loss)
        tf.summary.scalar("losses/total_loss", total_loss)
        for var in tf.trainable_variables():
            tf.summary.histogram("parameters/" + var.op.name, var)

        self.total_loss = total_loss
        self.target_smooth_l1_losses = batch_loss  # Used in evaluation.
        # self.target_smooth_l1_losses_weights = weights  # Used in evaluation.

        return total_loss


