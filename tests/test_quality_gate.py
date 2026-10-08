import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from vision import quality_gate as gate
from scripts.calibrate_quality import split_training,training_sources,read

class QualityGateTests(unittest.TestCase):
    def test_camera_remains_provisional(self):
        rng=np.random.default_rng(42);frame=rng.integers(20,230,(200,200,3),dtype=np.uint8)
        result=gate.check_frame(frame)
        self.assertTrue(result['passed']);self.assertFalse(result['calibrated']);self.assertEqual(result['calibration_version'],'camera-generic-v2')
        self.assertFalse(gate.check_frame(np.full_like(frame,128))['passed'])
    def test_hash_split_is_disjoint_and_training_only(self):
        for profile in ('neu_proxy','casting_proxy'):
            sources=training_sources(profile);fit,held=split_training(sources)
            self.assertFalse({r['sha256'] for r in fit}&{r['sha256'] for r in held})
            self.assertEqual(fit,split_training(list(reversed(sources)))[0][::-1])
            self.assertEqual(len(sources),1260 if profile=='neu_proxy' else 363)
            self.assertTrue(all('/train/' in r['path'] for r in sources) if profile=='neu_proxy' else all('/ok_front/' in r['path'] for r in sources))
    def test_deployed_clean_acceptance_and_severe_corruptions(self):
        for profile in ('neu_proxy','casting_proxy'):
            metadata,digest=gate.load_profile(profile)
            performance=metadata['performance']['held_train_validation']
            self.assertGreaterEqual(performance['clean']['rate'],.95)
            for corruption in ('gaussian_sigma3','gaussian_sigma5','underexposed','overexposed','blank'):
                self.assertGreaterEqual(performance[corruption]['rate'],.90 if corruption in ('overexposed','gaussian_sigma5') else .95)
            frame=read(metadata['sources']['quality_validation'][0]);result=gate.check_frame(frame,profile)
            self.assertEqual(result['calibration_sha256'],digest)
            self.assertEqual(json.loads(json.dumps(result,allow_nan=False)),result)
            self.assertFalse(gate.check_frame(np.full_like(frame,128),profile)['passed'])
    def test_missing_unknown_and_invalid_inputs(self):
        frame=np.zeros((200,200,3),np.uint8)
        with self.assertRaises(ValueError):gate.check_frame(frame,'guess_domain')
        with self.assertRaises(ValueError):gate.check_frame(np.zeros((200,200)), 'camera')
        with tempfile.TemporaryDirectory() as directory,patch.object(gate,'CALIBRATION_DIR',Path(directory)):
            with self.assertRaises(gate.QualityCalibrationUnavailable):gate.check_frame(frame,'neu_proxy')
    def test_calibration_identity_changes_even_same_mtime(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(gate,'CALIBRATION_DIR',Path(directory)):
            path=Path(directory)/'neu_proxy.json';first=json.loads((gate.ROOT/'models/quality/neu_proxy.json').read_text());first['version']='old'
            path.write_text(json.dumps(first));timestamp=path.stat().st_mtime_ns
            old,digest1=gate.load_profile('neu_proxy');first['version']='new';path.write_text(json.dumps(first));os.utime(path,ns=(timestamp,timestamp))
            new,digest2=gate.load_profile('neu_proxy');self.assertNotEqual(digest1,digest2);self.assertEqual(old['version'],'old');self.assertEqual(new['version'],'new')
    def test_corrupt_metadata_fails_closed(self):
        valid=json.loads((gate.ROOT/'models/quality/neu_proxy.json').read_text())
        cases=[]
        for key,value in [('mean_min',float('nan')),('saturation_max',1.1),('blur_ratio_min',-1),('contrast_min','bad'),('mean_max',-1)]:
            case=json.loads(json.dumps(valid));case['thresholds'][key]=value;cases.append(case)
        case=json.loads(json.dumps(valid));case['source_dimension_bounds']=[200,10];cases.append(case)
        case=json.loads(json.dumps(valid));case.pop('version');cases.append(case)
        with tempfile.TemporaryDirectory() as directory,patch.object(gate,'CALIBRATION_DIR',Path(directory)):
            for case in cases:
                (Path(directory)/'neu_proxy.json').write_text(json.dumps(case))
                with self.assertRaises(gate.QualityCalibrationUnavailable):gate.load_profile('neu_proxy')
if __name__=='__main__':unittest.main()
