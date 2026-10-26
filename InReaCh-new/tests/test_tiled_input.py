"""检查坐标、有效区域和标注语义, 防止小缺陷在输入阶段丢失."""
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from config import ExperimentConfig
from dataset import discover_images, load_yolo_boxes
from tiling import iter_tiles
from pipeline import prepare_batches


class TiledInputTests(unittest.TestCase):
    def test_invalid_config(self):
        for kwargs in ({'stride':225}, {'batch_size':0}, {'tile_size':0}, {'radius_px':-1}, {'stride':1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExperimentConfig(**kwargs)

    def test_yolo_coordinates_and_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.txt'
            self.assertIsNone(load_yolo_boxes(p,(100,200)))
            p.write_text('')
            self.assertEqual(load_yolo_boxes(p,(100,200)),())
            p.write_text('2 0.5 0.5 0.1 0.2\n')
            b=load_yolo_boxes(p,(100,200))[0]
            np.testing.assert_allclose((b.x1,b.y1,b.x2,b.y2),(90,40,110,60))
            self.assertEqual(b.class_id,2)

    def test_yolo_rounding_at_border_warns_and_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'border.txt'
            p.write_text('1 0.611814 0.949233 0.106892 0.101535')
            with self.assertWarnsRegex(UserWarning,'border.txt:1'):
                box=load_yolo_boxes(p,(847,711))[0]
            self.assertEqual(box.y2,847)
            p.write_text('0 .5 .999 .1 .01')
            with self.assertRaises(ValueError):
                load_yolo_boxes(p,(847,711))

    def test_rejects_invalid_labels_with_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.txt'
            for text in ('0 nan .5 .1 .1','0 .01 .5 .2 .1','0 .5 .5 0 .1','-1 .5 .5 .1 .1','0 .5 .5 .1 .1 .2'):
                p.write_text(text)
                with self.subTest(text=text), self.assertRaisesRegex(ValueError,'a.txt:1'):
                    load_yolo_boxes(p,(100,200))

    def test_discovery_selects_only_pic1_and_keeps_missing_labels_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'images').mkdir(); (root/'labels').mkdir()
            for name in ('pic1_001.jpg','pic10_001.jpg','pic1_002.png'):
                (root/'images'/name).touch()
            records=discover_images(root)
            self.assertEqual([r.image_id for r in records],['pic1_001','pic1_002'])
            self.assertIsNone(records[0].label_path)

    def test_non_divisible_geometry_keeps_pixels_and_covers_image(self):
        image=np.arange(847*711*3,dtype=np.uint32).reshape(847,711,3).astype(np.uint8)
        tiles=list(iter_tiles(image,'sample',ExperimentConfig()))
        self.assertEqual(len(tiles),42)
        self.assertEqual(tiles[-1].meta.origin_xy,(487,623))
        covered=np.zeros((847,711),dtype=bool)
        for tile in tiles:
            x,y=tile.meta.origin_xy
            np.testing.assert_array_equal(tile.image,image[y:y+224,x:x+224])
            covered[y:y+224,x:x+224]=True
        self.assertTrue(covered.all())

    def test_small_image_padding_preserves_valid_area(self):
        image=np.zeros((3,5,3),dtype=np.uint8); image[-1,-1]=[11,22,33]
        tile=list(iter_tiles(image,'small',ExperimentConfig()))[0]
        self.assertEqual(tile.image.shape,(224,224,3))
        self.assertEqual(tile.meta.valid_hw,(3,5))
        self.assertEqual(tile.meta.padding,(0,0,219,221))
        np.testing.assert_array_equal(tile.image[:3,:5],image)
        np.testing.assert_array_equal(tile.image[-1,-1],[11,22,33])

    def test_batching_does_not_resize_or_drop_last_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'images').mkdir()
            rgb=np.zeros((224,449,3),dtype=np.uint8); rgb[:]=[255,0,0]
            cv2.imwrite(str(root/'images'/'pic1_001.png'),rgb[:,:,::-1])
            batches=list(prepare_batches(discover_images(root)[0],ExperimentConfig(batch_size=3)))
            self.assertEqual([tuple(b.inputs.shape) for b in batches],[(3,3,224,224),(1,3,224,224)])
            self.assertEqual(batches[-1].metadata[0].origin_xy,(225,0))
            np.testing.assert_allclose(batches[0].inputs[0,:,0,0].numpy(),[(1-.485)/.229,-.456/.224,-.406/.225],rtol=1e-6)

if __name__=='__main__':
    unittest.main()
