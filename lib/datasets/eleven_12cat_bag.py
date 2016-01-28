# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.eleven_12cat_bag
import os
import datasets.imdb
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
from fast_rcnn.config import cfg
import cv2

class eleven_12cat_bag(datasets.imdb):
    def __init__(self, image_set, devkit_path):
        datasets.imdb.__init__(self, 'eleven_12cat_bag_' + image_set)        
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = devkit_path
        self._classes = ('__background__', # always index 0
                         'bag')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        
        self.config = {'rpn_file' : None}

        assert os.path.exists(self._devkit_path), \
                'devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """        
        image_path = os.path.join(self._data_path, 'Images', 
                                  index + self._image_ext)        
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index    

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self._load_11st_skp_annotation()

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_11st_skp_annotation(self):
        """
        Load image and bounding boxes info from annotation txt in the 11st DB
        """
        gt_roidb = []

        filename = os.path.join(self._data_path, 'Annotations', 'annotations_' + self._image_set + '.txt')
        
        f = open(filename)

        index = iter(self.image_index)
        while True:
            line = f.readline()
            if not line:break

            elements = line.split()
            if elements[0] != index.next():
                print 'No match between image index and annotation'
                exit(1)
            
            num_objs = int(elements[1])

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

            # Load object bounding boxes into a data frame.
            for i in range(0, num_objs):
                x1 = float(elements[2+i*5+1])
                y1 = float(elements[2+i*5+2])
                x2 = float(elements[2+i*5+3])
                y2 = float(elements[2+i*5+4])
                cls = int(elements[2+i*5])

                boxes[i, :] = [x1, y1, x2, y2]
                gt_classes[i] = cls
                overlaps[i, cls] = 1.0

            overlaps = scipy.sparse.csr_matrix(overlaps)            

            gt_roidb.append({'boxes' : boxes,
                          'gt_classes': gt_classes,
                          'gt_overlaps' : overlaps,
                          'flipped' : False} )
            
        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _makeDirectoryIfNotExists(self, path):
        if not os.path.exists(path):        
            os.makedirs(path)        

    def _calc_roi_size(self, img):
        height = img.shape[0]
        width = img.shape[1]

        return int(width*height*0.001)

    def rpn_roidb(self):
        if self._image_set != 'test':
            if cfg.TRAIN.RPN_DISK:
                roidb = self._rpn_roidb_disk()
            else:
                gt_roidb = self.gt_roidb()
                rpn_roidb = self._load_rpn_roidb(gt_roidb)
                roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _rpn_roidb_disk(self):
        filename = self.config['rpn_file']
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            cache_files = cPickle.load(f)
                           
        gt_roidb = self.gt_roidb()

        for i, cache_file in enumerate(cache_files):            
            self._update_rpn_roidb_cache_file(i, gt_roidb[i].copy(), cache_file)

        return cache_files

    def _update_rpn_roidb_cache_file(self, i, gt, cache_file):
        with open(cache_file, 'rb') as f:
            rpn = cPickle.load(f)

        rpn_roidb_element = self.create_roidb_element_from_obj_proposals(rpn, gt)

        roidb_element = datasets.imdb.merge_roidb_elements(gt, rpn_roidb_element)
        roidb_element = datasets.imdb.enrich_roidb_element(self.image_path_at(i), roidb_element)

        with open(cache_file, 'wb') as f:
            cPickle.dump(roidb_element, f, cPickle.HIGHEST_PROTOCOL)

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _write_voc_results_file(self, all_boxes):       
        comp_id = 'faster-rcnn-{}'.format(os.getpid())

        path = os.path.join(self._devkit_path, 'results', comp_id)
        
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue

            print 'Writing {} results file'.format(cls)
            filename = path + '/det_' + self._image_set + '_' + cls + '.txt'
            self._makeDirectoryIfNotExists(path)

            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(index, 
                            dets[k, -1], dets[k, 0], dets[k, 1],
                            dets[k, 2], dets[k, 3]))

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)

if __name__ == '__main__':
    d = eleven_12cat_bag('train', '/home/joonyoung/e/data/11st/2015_Nov_12cat/11st_Bag')
    res = d.roidb
    from IPython import embed; embed()
