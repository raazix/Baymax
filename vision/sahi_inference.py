"""Optional SAHI slicing for the trained NEU detector; full-image coordinates and class-aware merging."""
import math
import time

MAX_TILES = 64

def sliced_detect(model, frame, device, tile_size=512, overlap=.2):
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or not 128 <= tile_size <= 2048:
        raise ValueError('tile_size must be an integer from 128 to 2048')
    if not math.isfinite(overlap) or not 0 <= overlap <= .5:
        raise ValueError('overlap must be from 0 to 0.5')
    try:
        import sahi
        from sahi.slicing import get_slice_bboxes
        from sahi.models.ultralytics import UltralyticsDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError as error:
        from vision.yolo_detector import ModelUnavailable
        raise ModelUnavailable('Install the pinned SAHI dependency to use sliced inference.') from error
    height, width = frame.shape[:2]
    tiles = get_slice_bboxes(height, width, slice_height=tile_size, slice_width=tile_size,
                             overlap_height_ratio=overlap, overlap_width_ratio=overlap, auto_slice_resolution=False)
    if len(tiles) > MAX_TILES:
        raise ValueError(f'Sliced inference exceeds {MAX_TILES} tiles. Increase tile_size or reduce overlap.')
    adapter = UltralyticsDetectionModel(model=model, device='cuda:' + str(device) if str(device).isdigit() else str(device),
                                        confidence_threshold=.25, image_size=640)
    started = time.perf_counter()
    # API frames are OpenCV BGR; SAHI takes RGB and its adapter converts back to BGR for YOLO.
    prediction = get_sliced_prediction(frame[:, :, ::-1].copy(), detection_model=adapter,
        slice_height=tile_size, slice_width=tile_size, overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        perform_standard_pred=True, auto_slice_resolution=False, postprocess_type='GREEDYNMM',
        postprocess_match_metric='IOS', postprocess_match_threshold=.5, postprocess_class_agnostic=False,
        verbose=0, progress_bar=False)
    detections = []
    for item in prediction.object_prediction_list:
        x1, y1, x2, y2 = item.bbox.to_xyxy()
        box = [max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))]
        if box[2] > box[0] and box[3] > box[1]:
            detections.append({'label': item.category.name, 'confidence': float(item.score.value), 'bbox_xyxy_px': box})
    return detections, {'mode': 'sliced', 'engine': 'sahi', 'version': sahi.__version__, 'tile_size_px': tile_size,
        'overlap_ratio': overlap, 'tile_count': len(tiles), 'full_image_pass': True,
        'merge': {'method': 'GREEDYNMM', 'metric': 'IOS', 'threshold': .5, 'class_agnostic': False},
        'image_size_px': [width, height], 'confidence_threshold': .25, 'model_input_size': 640,
        'inference_ms': round((time.perf_counter() - started) * 1000, 2)}
