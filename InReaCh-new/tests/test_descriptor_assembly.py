"""跨批汇总与建库位置归属的行为测试。"""
import unittest
import numpy as np
import torch
from config import ExperimentConfig
from contracts import DescriptorSet, TileMeta
from descriptors import build_descriptors, merge_descriptor_sets, select_reference_descriptors


class DescriptorAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.config=ExperimentConfig(tile_size=8,stride=4)
        self.metas=(TileMeta('a',0,(0,0),(8,13),(8,8),(0,0,0,0)),
                    TileMeta('a',1,(5,0),(8,13),(8,8),(0,0,0,0)))
        self.maps={'Level_1':torch.arange(8,dtype=torch.float32).reshape(2,1,2,2)}

    def test_merge_split_batches_matches_one_batch_and_keeps_order(self):
        together=build_descriptors(self.maps,self.metas,self.config)[0]
        pieces=[build_descriptors({'Level_1':v[i:i+1]},(self.metas[i],),self.config)[0]
                for i in range(2) for v in [self.maps['Level_1']]]
        merged=merge_descriptor_sets(pieces)[0]
        torch.testing.assert_close(merged.features,together.features)
        np.testing.assert_array_equal(merged.xy,together.xy)
        np.testing.assert_array_equal(merged.tile_ids,together.tile_ids)
        np.testing.assert_array_equal(merged.valid,together.valid)
        self.assertEqual(merged.grid_hw,together.grid_hw)

    def test_merge_rejects_duplicate_tile_and_malformed_rows(self):
        piece=build_descriptors({'Level_1':self.maps['Level_1'][:1]},self.metas[:1],self.config)[0]
        with self.assertRaisesRegex(ValueError,'重复'):
            merge_descriptor_sets([piece,piece])
        bad=DescriptorSet(piece.features,piece.xy[:-1],piece.image_id,piece.tile_ids,piece.grid_hw,piece.valid)
        with self.assertRaises(ValueError):
            merge_descriptor_sets([bad])

    def test_shifted_edge_tile_wins_overlap_by_margin(self):
        maps={'Level_1':torch.arange(16,dtype=torch.float32).reshape(2,1,2,4)}
        d=build_descriptors(maps,self.metas,self.config)[0]
        selected=select_reference_descriptors([d],self.metas)[0]
        # 非规则末块从x=5起。左块的x=7由末块拥有;
        # 末块的x=6由左块拥有。不能简单按网格下标合并。
        self.assertEqual(selected.features.shape[0],12)
        np.testing.assert_array_equal(selected.tile_ids,[0]*6+[1]*6)
        np.testing.assert_allclose(selected.xy[:,0],[1,3,5,1,3,5,8,10,12,8,10,12])
        self.assertEqual(d.features.shape[0],16)  # 原始描述子仍保留重叠观测

    def test_tie_uses_smallest_tile_id_and_padding_is_removed(self):
        cfg=ExperimentConfig(tile_size=8,stride=4)
        metas=(TileMeta('a',3,(0,0),(8,12),(8,8),(0,0,0,0)),
               TileMeta('a',1,(4,0),(8,12),(8,8),(0,0,0,0)))
        # x=6 对两个块最近边缘距离均为2; 虽后出现, tile_id=1获胜。
        feature=torch.arange(2,dtype=torch.float32).reshape(2,1)
        d=DescriptorSet(feature,np.array([[6.,4.],[6.,4.]]),'a',
                        np.array([3,1]),{3:(1,1),1:(1,1)},np.array([True,True]))
        out=select_reference_descriptors([d],metas)[0]
        np.testing.assert_array_equal(out.tile_ids,[1])
        tiny=TileMeta('tiny',0,(0,0),(3,5),(3,5),(0,0,3,5))
        piece=build_descriptors({'Level_1':torch.ones(1,1,2,2)},(tiny,),cfg)[0]
        self.assertEqual(select_reference_descriptors([piece],(tiny,))[0].features.shape[0],1)

    def test_selection_requires_complete_matching_metadata(self):
        d=build_descriptors(self.maps,self.metas,self.config)[0]
        with self.assertRaises(ValueError):
            select_reference_descriptors([d],self.metas[:1])
        with self.assertRaises(ValueError):
            select_reference_descriptors([d],(self.metas[0],self.metas[0]))

if __name__=='__main__': unittest.main()
