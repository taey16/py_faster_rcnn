# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
import cv2
import cPickle
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
import datasets

class imdb(object):
    """Image database."""

    def __init__(self, name):
        print('===> Start imdb.__init__')
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._obj_proposer = 'selective_search'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}
        print('===> Start imdb.__init__. done')

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        proposal_method = 'self.' + method + '_roidb'
        method = eval(proposal_method)
        print('===> imdb.set_proposal_method; %s' % proposal_method )
        self.roidb_handler = method
        print('===> imdb.set_proposal_method. done')

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(datasets.ROOT_DIR, 'data', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
      return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError


    def append_flipped_images(self):
        print('===> Start imdb.append_flipped_images')
        num_images = self.num_images

        cache_file = os.path.join(self.cache_path, self.name + '_image_widths.pkl')
        if os.path.exists(cache_file):
            with open(cache_file,'rb') as fid:
                widths = cPickle.load(fid)
            print '{} image widths loaded form {}'.format(self.name, cache_file)
        else:
            #widths = [cv2.imread(self.image_path_at(i)).shape[1] for i in xrange(num_images)]
            widths = []
            for i in xrange(num_images):
              try:
                width = cv2.imread(self.image_path_at(i)).shape[1]
                widths.append(width)
                if i % 1000 == 0: print('imdb.append_flipped_images %08d completed' % i)
              except Exception as err:
                print('%s' % self.image_path_at(i))
                print(err)
            with open(cache_file,'wb') as fid:
                cPickle.dump(widths, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote image widths to {}'.format(self.name, cache_file)

        roidb_disk = True if type(self.roidb[0]) == str else False
        for i in xrange(num_images):
            if roidb_disk:
                with open(self.roidb[i], 'rb') as fid:
                    orig_roidb = cPickle.load(fid)

                entry = self._roidb_entry_flipped_copy(orig_roidb, widths[i])
                entry = imdb.enrich_roidb_element(self.image_path_at(i), entry)

                flip_cache_file = os.path.join(os.path.dirname(self.roidb[i]), 'flipped',
                                                    self.image_index[i]+'.pkl')

                if not os.path.exists(os.path.dirname(flip_cache_file)):
                    os.makedirs(os.path.dirname(flip_cache_file))

                self.roidb.append(flip_cache_file)
                with open(flip_cache_file, 'wb') as fid:
                    cPickle.dump(entry, fid, cPickle.HIGHEST_PROTOCOL)
            else:
                entry = self._roidb_entry_flipped_copy(self.roidb[i], widths[i])
                self.roidb.append(entry)
        self._image_index = self._image_index * 2
        print('===> Start imdb.append_flipped_images. done')

    def _roidb_entry_flipped_copy(self, orig_roidb, width):
        boxes = orig_roidb['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()

        ex_idx = np.where(oldx2>=width)
        oldx2[ex_idx] = width-1

        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        entry = {'boxes' : boxes,
                 'gt_overlaps' : orig_roidb['gt_overlaps'],
                 'gt_classes' : orig_roidb['gt_classes'],
                 'flipped' : True}

        return entry


    def evaluate_recall(self, candidate_boxes=None, ar_thresh=0.5):
        # Record max overlap value for each gt box
        # Return vector of overlap values
        gt_overlaps = np.zeros(0)
        for i in xrange(self.num_images):
            gt_inds = np.where(self.roidb[i]['gt_classes'] > 0)[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]

            if candidate_boxes is None:
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            overlaps = bbox_overlaps(boxes.astype(np.float),
                                     gt_boxes.astype(np.float))

            # gt_overlaps = np.hstack((gt_overlaps, overlaps.max(axis=0)))
            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in xrange(gt_boxes.shape[0]):
                argmax_overlaps = overlaps.argmax(axis=0)
                max_overlaps = overlaps.max(axis=0)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert(gt_ovr >= 0)
                box_ind = argmax_overlaps[gt_ind]
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert(_gt_overlaps[j] == gt_ovr)
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1

            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        num_pos = gt_overlaps.size
        gt_overlaps = np.sort(gt_overlaps)
        step = 0.001
        thresholds = np.minimum(np.arange(0.5, 1.0 + step, step), 1.0)
        recalls = np.zeros_like(thresholds)
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        ar = 2 * np.trapz(recalls, thresholds)

        return ar, gt_overlaps, recalls, thresholds

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
                'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in xrange(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                            gt_boxes.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({'boxes' : boxes,
                          'gt_classes' : np.zeros((num_boxes,),
                                                  dtype=np.int32),
                          'gt_overlaps' : overlaps,
                          'flipped' : False})
        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in xrange(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass
