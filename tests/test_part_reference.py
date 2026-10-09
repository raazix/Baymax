import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from core.part_reference import identify_reference


class PartReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.registry = Path(self.directory.name) / 'references.json'
        self.digest = hashlib.sha256(b'reference image bytes').hexdigest()
        self.registry.write_text(json.dumps({'format_version': 1, 'part_type': 'brake_disc',
            'image_sha256': [self.digest]}), encoding='utf-8')

    def tearDown(self):
        self.directory.cleanup()

    def test_exact_upload_match_has_explicit_reference_provenance(self):
        result = identify_reference(self.digest, 'upload', self.registry)
        self.assertEqual(result['part_type'], 'brake_disc')
        self.assertEqual(result['source'], 'curated_reference_image')
        self.assertFalse(result['model_prediction'])
        self.assertEqual(result['matched_sha256'], self.digest)

    def test_camera_is_unknown_even_if_image_bytes_match(self):
        self.assertEqual(identify_reference(self.digest, 'camera', self.registry)['part_type'], 'unclassified')

    def test_changed_image_is_unknown(self):
        self.assertEqual(identify_reference(hashlib.sha256(b'changed').hexdigest(), 'upload', self.registry)['part_type'], 'unclassified')

    def test_missing_or_invalid_registry_is_unknown(self):
        self.assertEqual(identify_reference(self.digest, 'upload', Path(self.directory.name) / 'absent.json')['part_type'], 'unclassified')
        self.registry.write_text('invalid', encoding='utf-8')
        self.assertEqual(identify_reference(self.digest, 'upload', self.registry)['part_type'], 'unclassified')
