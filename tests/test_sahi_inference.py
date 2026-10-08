import unittest
from unittest.mock import Mock, patch
import numpy as np
from sahi.prediction import ObjectPrediction
from sahi.postprocess.combine import GreedyNMMPostprocess
from vision.sahi_inference import sliced_detect

class SahiTests(unittest.TestCase):
    def test_overlapping_tiles_merge_in_global_coordinates_without_merging_classes(self):
        a = ObjectPrediction(bbox=[90, 20, 130, 60], category_id=0, category_name='crazing', score=.9)
        b = ObjectPrediction(bbox=[10, 20, 50, 60], shift_amount=[80, 0], category_id=0, category_name='crazing', score=.8).get_shifted_object_prediction()
        c = ObjectPrediction(bbox=[90, 20, 130, 60], category_id=1, category_name='inclusion', score=.85)
        merged = GreedyNMMPostprocess(match_threshold=.5, match_metric='IOS', class_agnostic=False)([a, b, c])
        self.assertEqual(len(merged), 2)
        self.assertEqual({p.category.name for p in merged}, {'crazing', 'inclusion'})
        self.assertEqual(merged[0].bbox.to_xyxy(), [90, 20, 130, 60])

    def test_color_conversion_settings_and_clipped_coordinates(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8); frame[:] = [10, 20, 30]
        result = Mock(object_prediction_list=[ObjectPrediction(bbox=[190, 190, 220, 220], category_id=0, category_name='crazing', score=.9)])
        with patch('sahi.models.ultralytics.UltralyticsDetectionModel') as adapter, patch('sahi.predict.get_sliced_prediction', return_value=result) as predict:
            detections, settings = sliced_detect(Mock(), frame, '0', 128, .2)
        np.testing.assert_array_equal(predict.call_args.args[0][0, 0], [30, 20, 10])
        self.assertEqual(adapter.call_args.kwargs['device'], 'cuda:0')
        self.assertEqual(settings['tile_count'], 4)
        self.assertEqual(detections[0]['bbox_xyxy_px'], [190, 190, 200, 200])
        self.assertFalse(predict.call_args.kwargs['postprocess_class_agnostic'])

    def test_workload_and_invalid_settings_fail_before_inference(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        for tile, overlap in [(127, .2), (128, .6), (128, float('nan'))]:
            with self.assertRaises(ValueError): sliced_detect(Mock(), frame, 'cpu', tile, overlap)
        with self.assertRaises(ValueError): sliced_detect(Mock(), np.zeros((2048, 2048, 3), dtype=np.uint8), 'cpu', 128, .5)

if __name__ == '__main__': unittest.main()
