# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.pascal_voc
import datasets.eleven_all
import datasets.eleven_12cat_bag
import numpy as np

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

"""
# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k))
"""


#import pdb; pdb.set_trace()
eleven_all_devkit_path = '/storage/product/detection/11st_All/'
for split in ['train']:
    name = 'eleven_all_{}'.format(split)
    __sets[name] = (lambda split=split :
        datasets.eleven_all.eleven_all(split, eleven_all_devkit_path))

#import pdb; pdb.set_trace()
eleven_12cat_bag_devkit_path = '/storage/product/detection/11st_Bag/'
for split in ['train', 'val']:
    name = 'eleven_12cat_bag_{}'.format(split)
    __sets[name] = (lambda split=split :
        datasets.eleven_12cat_bag.eleven_12cat_bag(split, eleven_12cat_bag_devkit_path))


def get_imdb(name):
    print('===> Start get_imdb in factory.py')
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))

    print('===> Start get_imdb in factory.py. done')
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()


print('===> Factory __sets.keys()')
for k in __sets.keys():
  print('%s' % k)


