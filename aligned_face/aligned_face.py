#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 ccmicky <chengchun@imdada.cn>
# Licensed under the Dada tech.co.ltd - http://www.imdada.cn

import dlib
import cv2
import pickle
from utils.timer import Timer
import numpy as np
import tensorflow as tf
from fast_rcnn.config import cfg
import os

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]


class FaceDetect(object):
    detector = None
    imgDim = 96
    net = None
    align = None
    predictor = None

    def __init__(self, net, saver, predictor, output_dir,
                 img_dim=96):
        self.imgDim = img_dim
        self.net = net
        self.predictor = predictor
        self.saver = saver
        self.output_dir = output_dir
        self.pretrained_model = './face_model_1/face_vec_model_iter_4000.ckpt'

    def _get_face_rectangle(self, box):
        dets = dlib.rectangle(*box)
        return dets

    def _find_land_marks(self, img, det):
        points = self.predictor(img, det)
        return list(map(lambda p: (p.x, p.y), points.parts()))

    def create_aligned_face(self, box, img,
                            landmark_indices=OUTER_EYES_AND_NOSE):
        det = self._get_face_rectangle(box)
        landmarks = self._find_land_marks(img, det)
        print "landmarks", landmarks

        np_landmarks = np.float32(landmarks)
        np_landmark_indices = np.array(landmark_indices)
        h = cv2.getAffineTransform(np_landmarks[np_landmark_indices],
                                   self.imgDim * MINMAX_TEMPLATE[
                                       np_landmark_indices])
        thumbnail = cv2.warpAffine(img, h, (self.imgDim, self.imgDim))
        return thumbnail

    def rectangle_2_vec(self, sess, aligned_face):
        run_options = None
        run_metadata = None
        feed_dict = {self.net.data: aligned_face}
        print "self.net", self.net, dir(self.net)
        rep, fc_face_vec, triplet_fcs = sess.run(
            [self.net.get_output('avg_pool_736'),
             self.net.get_output('fc_face_vec'),
             self.net.get_output('triplet_fc')],
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata)
        print triplet_fcs.shape
        a = sess.run(tf.argmax(triplet_fcs, 1))
        print a
        return fc_face_vec

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if net.layers.has_key('fc_face_vec'):
            # save original values
            with tf.variable_scope('fc_face_vec', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

        if net.layers.has_key('nn4_conv1_1'):
            # save original values
            with tf.variable_scope('nn4_conv1_1', reuse=True):
                nn4_weights = tf.get_variable("weights")
                nn4_biases = tf.get_variable("biases")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename = ('face_vec_model' +
                    '_iter_{:d}'.format(iter + 1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print "weights", sess.run(weights), sess.run(biases)
        print "nn4_weights", sess.run(nn4_weights), sess.run(nn4_biases)

        print 'Wrote snapshot to: {:s}'.format(filename)

    def train_model(self, sess, aligned_faces, labels, max_iters, margin=0.5,
                    path=None):
        """
        rows = aligned_faces.shape[0]/3
        batch = 100

        mat = self.net.get_output('fc_face_vec')
        mat_a = mat[:batch, :]
        mat_p = mat[batch:2*batch, :]
        mat_n = mat[2*batch:, :]
        d_a_p = tf.reduce_sum(tf.square(mat_a - mat_p), 1)
        d_a_n = tf.reduce_sum(tf.square(mat_a - mat_n), 1)

        # print dd
        pos_dist = tf.maximum((d_a_p - 0.5), 0)
        neg_dist = tf.maximum((1 - d_a_n), 0)
        #basic_loss = tf.add(tf.subtract(d_a_p, d_a_n), margin)
        #score = tf.maximum(basic_loss, 0)
        #triplet_loss = tf.reduce_mean(score, 0)
        triplet_loss = tf.reduce_mean(neg_dist + pos_dist, 0)
        """

        batch = 100
        mat = self.net.get_output('triplet_fc')
        mat_p = tf.nn.softmax(mat[:batch, :])[:, -1]
        mat_n = tf.nn.softmax(mat[batch:, :])[:, -1]

        basic_loss = tf.add(tf.subtract(mat_p, mat_n), margin)
        score = tf.maximum(basic_loss, 0)
        triplet_loss = tf.reduce_mean(score, 0)

        # optimizer and learning rate
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.Variable(0, trainable=False)

        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                        cfg.TRAIN.STEPSIZE, 0.01,
                                        staircase=True)

        # Calculate the total losses
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses,
                              name='total_loss')

        momentum = cfg.TRAIN.MOMENTUM
        with tf.control_dependencies(update_ops):
            train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(
                total_loss, global_step=global_step)
        # iintialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()

        for iter in range(max_iters):
            for i in range(batch):
                filepath = './matrix/face_matrix_train_%s_%s.pkl'
                b = np.random.randint(1, 17)
                idx = np.random.randint(0, 1000)
                r = np.random.randint(0, 10)
                filepath = filepath % (b, idx)
                aligned_faces = pickle.load(open(filepath, 'rb'))
                select_faces = aligned_faces[[r, r+10, r+20], :, :, :]
                if i == 0:
                    input_faces = select_faces
                else:
                    input_faces = np.append(input_faces, select_faces, axis=0)
            #rand_index = np.random.choice(rows, size=batch)
            #indexs = np.append(rand_index, rand_index + rows)
            #indexs = np.append(indexs, rand_index + 2 * rows)
            #input_faces = aligned_faces[indexs, :, :, :]
            feed_dict = {self.net.data: input_faces}
            timer.tic()
            run_options = None
            run_metadata = None
            triplet_fcs, losses, mat0, _ = sess.run(
                [self.net.get_output('triplet_fc'), total_loss,
                 mat, train_op],
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)
            timer.toc()
            if (iter + 1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, lr: %f' % \
                      (iter + 1, max_iters,
                       losses, lr.eval(session=sess))
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


def train_net(sess,  net, predictor, aligned_faces, labels, output_dir,
              max_iters=40000):
    saver = tf.train.Saver(max_to_keep=100)
    fd = FaceDetect(net, saver, predictor, output_dir)
    print 'Solving...'
    fd.train_model(sess, aligned_faces, labels, max_iters)
    print 'done solving'
