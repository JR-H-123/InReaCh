"""检验多层描述子顺序、原图坐标及补边有效性。"""
import unittest
import numpy as np
import torch
from config import ExperimentConfig
from contracts import TileMeta
from descriptors import build_descriptors


class DescriptorTests(unittest.TestCase):
    def test_multilevel_shapes_and_coordinates_at_shifted_edge(self):
        cfg=ExperimentConfig(tile_size=8,stride=4)
        meta=(TileMeta('a',0,(0,0),(12,13),(8,8),(0,0,0,0)),
              TileMeta('a',1,(5,4),(12,13),(8,8),(0,0,0,0)))
        maps={'Level_1':torch.ones(2,2,2,2), 'Level_2':torch.full((2,3,1,1),2.)}
        out=build_descriptors(maps,meta,cfg)
        self.assertEqual(len(out),1)
        d=out[0]
        self.assertEqual(tuple(d.features.shape),(8,1024))
        self.assertEqual(d.grid_hw,{0:(2,2),1:(2,2)})
        np.testing.assert_array_equal(d.tile_ids,[0,0,0,0,1,1,1,1])
        np.testing.assert_allclose(d.xy,[(2,2),(6,2),(2,6),(6,6),(7,6),(11,6),(7,10),(11,10)])
        self.assertTrue(d.valid.all())
        self.assertTrue(torch.isfinite(d.features).all())

    def test_padding_marks_centers_outside_original_invalid(self):
        cfg=ExperimentConfig(tile_size=8,stride=4)
        meta=(TileMeta('tiny',0,(0,0),(3,5),(3,5),(0,0,3,5)),)
        maps={'Level_1':torch.ones(1,1,2,2)}
        d=build_descriptors(maps,meta,cfg)[0]
        np.testing.assert_array_equal(d.valid,[True,False,False,False])
        np.testing.assert_allclose(d.xy,[(2,2),(6,2),(2,6),(6,6)])

    def test_separate_images_and_reject_bad_feature_batch(self):
        cfg=ExperimentConfig(tile_size=8,stride=4)
        metas=(TileMeta('a',0,(0,0),(8,8),(8,8),(0,0,0,0)),
               TileMeta('b',0,(0,0),(8,8),(8,8),(0,0,0,0)))
        maps={'Level_1':torch.ones(2,1,2,2)}
        self.assertEqual([d.image_id for d in build_descriptors(maps,metas,cfg)],['a','b'])
        with self.assertRaises(ValueError):
            build_descriptors({'Level_1':torch.ones(1,1,2,2)},metas,cfg)

    def test_local_pooling_reacts_to_neighboring_feature(self):
        cfg=ExperimentConfig(tile_size=8,stride=4)
        meta=(TileMeta('a',0,(0,0),(8,8),(8,8),(0,0,0,0)),)
        a=torch.zeros(1,1,2,2)
        b=a.clone();b[0,0,0,1]=10
        x=build_descriptors({'Level_1':a},meta,cfg)[0].features
        y=build_descriptors({'Level_1':b},meta,cfg)[0].features
        self.assertGreater(torch.linalg.vector_norm(y[0]-x[0]).item(),0)

if __name__=='__main__': unittest.main()
