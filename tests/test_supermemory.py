import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from analytics import supermemory

INSPECTION = {'id': 'abc', 'machine_id': 'M-02', 'lot_id': 'M-02-L1008B',
              'defects': [{'label': 'anomaly_unclassified', 'severity': {'level': 'high'}, 'class_hint': {'label': 'crazing'}}],
              'analytics': {'rca': {'hypothesis': 'thermal_process_drift'}},
              'context': {'lot_history': {'drift': [{'sensor': 'temperature_c', 'change': 37.5}]}},
              'action': {'text': 'Inspect the thermocouple.', 'engineer': 'A. Engineer'},
              'verification': {'status': 'demo_criteria_met', 'parts': 20, 'defective': 0}}


class SupermemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_not_configured_stores_nothing(self):
        with patch.dict(os.environ, {'SUPERMEMORY_API_KEY': ''}):
            self.assertEqual(await supermemory.remember_approved(INSPECTION), {'stored': False, 'reason': 'not_configured'})
            self.assertEqual(await supermemory.search_precedents(INSPECTION), [])

    async def test_memory_content_carries_context_and_never_images(self):
        post = AsyncMock(return_value=httpx.Response(200, json={}, request=httpx.Request('POST', 'https://x')))
        with patch.dict(os.environ, {'SUPERMEMORY_API_KEY': 'test'}), patch.object(httpx.AsyncClient, 'post', post):
            self.assertTrue((await supermemory.remember_approved(INSPECTION))['stored'])
            self.assertTrue((await supermemory.remember_verification(INSPECTION))['stored'])
        approved = post.call_args_list[0].kwargs['json']['memories'][0]
        verified = post.call_args_list[1].kwargs['json']['memories'][0]
        for text in ('M-02', 'M-02-L1008B', 'thermal_process_drift', 'temperature_c +37.5', 'crazing', 'not causal proof'):
            self.assertIn(text, approved['content'])
        self.assertIn('0 of 20 follow-up parts defective', verified['content'])
        self.assertEqual(verified['metadata']['type'], 'verification_outcome')
        self.assertNotIn('image', str(post.call_args_list[0].kwargs['json']).lower().replace('images', ''))

    async def test_provider_failure_is_reported_not_raised(self):
        post = AsyncMock(return_value=httpx.Response(401, json={}, request=httpx.Request('POST', 'https://x')))
        with patch.dict(os.environ, {'SUPERMEMORY_API_KEY': 'test'}), patch.object(httpx.AsyncClient, 'post', post):
            self.assertEqual(await supermemory.remember_approved(INSPECTION), {'stored': False, 'reason': 'provider_http_401'})


if __name__ == '__main__':
    unittest.main()
