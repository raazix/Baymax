import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import httpx
from analytics import assistant

class AssistantTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = patch.object(assistant, 'STORE', Path(self.directory.name)); self.store.start()
        self.keys = patch.dict(os.environ, {'NVIDIA_API_KEY': 'test-only', 'ELEVENLABS_API_KEY': 'test-only'}); self.keys.start()
        self.inspection = {'id': 'part-test', 'part_id': 'P1', 'lot_id': 'L1', 'machine_id': 'M1',
            'source': 'uploaded_image', 'image_sha256': 'abc', 'defects': [], 'quality': {'passed': True},
            'telemetry': {}, 'action': {'text': 'Review the part.', 'status': 'pending'},
            'context': {'model': 'casting', 'anomaly': {'score': 3.0, 'threshold': 2.3, 'flagged': True, 'grid': [[99]]}}}

    def tearDown(self):
        self.keys.stop(); self.store.stop(); self.directory.cleanup()

    def provider_answer(self, evidence_ids=None):
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            'answer': 'The proxy anomaly score exceeds its threshold. Review the part.',
            'evidence_ids': evidence_ids or ['inspection', 'action']})}}]})

    async def test_explanation_snapshot_integrity_and_no_action_mutation(self):
        original = json.dumps(self.inspection)
        with patch.object(assistant, 'provider_post', AsyncMock(return_value=self.provider_answer())) as provider:
            result = await assistant.explain(self.inspection, 'Why flagged?')
        self.assertTrue(result['read_only'])
        self.assertEqual(original, json.dumps(self.inspection))
        saved = assistant.read_record(result['id'])
        self.assertEqual(saved['evidence_snapshot']['facts']['anomaly']['threshold'], 2.3)
        self.assertNotIn('grid', saved['evidence_snapshot']['facts']['anomaly'])
        self.assertNotIn('evidence_snapshot', result)
        self.assertFalse(provider.call_args.kwargs['json']['stream'])
        path = assistant.STORE / (result['id'] + '.json')
        saved['answer'] = 'Changed'; path.write_text(json.dumps(saved))
        with self.assertRaises(assistant.AssistantUnavailable) as error:
            assistant.read_record(result['id'])
        self.assertEqual(error.exception.status, 409)

    async def test_unknown_citations_rejected(self):
        with patch.object(assistant, 'provider_post', AsyncMock(return_value=self.provider_answer(['invented']))):
            with self.assertRaises(assistant.AssistantUnavailable):
                await assistant.explain(self.inspection, 'Explain')
        self.assertEqual(list(assistant.STORE.glob('*.json')), [])

    async def test_speech_uses_saved_answer(self):
        response = httpx.Response(200, content=b'ID3audio', headers={'content-type': 'audio/mpeg'})
        with patch.object(assistant, 'provider_post', AsyncMock(return_value=response)) as provider:
            self.assertEqual(await assistant.speech({'answer': 'Review the part.'}), b'ID3audio')
        self.assertEqual(provider.call_args.kwargs['json']['text'], 'AI explanation. Review the part.')

    async def test_transcript_requires_review(self):
        with patch.object(assistant, 'provider_post', AsyncMock(return_value=httpx.Response(200, json={'text': 'Why flagged?'}))):
            result = await assistant.transcribe(b'audio', 'audio/webm')
        self.assertTrue(result['review_before_sending'])
        with self.assertRaises(assistant.AssistantUnavailable):
            await assistant.transcribe(b'audio', 'text/plain')

    async def test_alert_speaks_stored_critical_evidence_without_llm_or_mutation(self):
        self.inspection['source'] = 'synthetic_replay'
        self.inspection['defects'] = [{'label': 'surface_crack', 'severity': {'level': 'critical'}}]
        original = json.dumps(self.inspection)
        response = httpx.Response(200, content=b'ID3alert', headers={'content-type': 'audio/mpeg'})
        with patch.object(assistant, 'provider_post', AsyncMock(return_value=response)) as provider:
            self.assertEqual(await assistant.alert_speech(self.inspection), b'ID3alert')
        self.assertEqual(original, json.dumps(self.inspection))
        self.assertIn('api.elevenlabs.io/v1/text-to-speech/', provider.call_args.args[0])
        text = provider.call_args.kwargs['json']['text']
        self.assertIn('Synthetic demonstration.', text)
        self.assertIn('Critical quality alert.', text)
        self.assertIn('surface crack', text)
        self.assertIn('lot L1', text)

    async def test_alert_rejects_normal_and_rejected_captures_before_provider(self):
        with patch.object(assistant, 'provider_post', AsyncMock()) as provider:
            with self.assertRaises(assistant.AssistantUnavailable):
                await assistant.alert_speech(self.inspection)
            self.inspection['defects'] = [{'label': 'scratch', 'severity': {'level': 'high'}}]
            self.inspection['quality']['passed'] = False
            with self.assertRaises(assistant.AssistantUnavailable):
                await assistant.alert_speech(self.inspection)
        provider.assert_not_called()

    async def test_provider_errors_never_echo_response_or_key(self):
        response = httpx.Response(401, json={'detail': 'secret-provider-error'})
        with patch('httpx.AsyncClient.post', AsyncMock(return_value=response)):
            with self.assertRaises(assistant.AssistantUnavailable) as error:
                await assistant.provider_post('https://example.invalid', provider='NVIDIA')
        self.assertNotIn('secret-provider-error', str(error.exception))

    def test_invalid_response_path(self):
        with self.assertRaises(assistant.AssistantUnavailable):
            assistant.read_record('../../.env')

if __name__ == '__main__': unittest.main()
