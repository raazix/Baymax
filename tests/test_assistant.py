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
        vision = saved['evidence_snapshot']['facts']['vision']
        self.assertEqual(vision['anomaly_threshold'], 2.3)
        self.assertNotIn('grid', json.dumps(saved['evidence_snapshot']))      # compact packet never ships the anomaly grid
        self.assertEqual(saved['evidence_format'], 'compact')
        self.assertEqual(saved['number_grounding'], 'passed')
        self.assertNotIn('evidence_snapshot', result)
        self.assertFalse(provider.call_args.kwargs['json']['stream'])
        path = assistant.STORE / (result['id'] + '.json')
        saved['answer'] = 'Changed'; path.write_text(json.dumps(saved))
        with self.assertRaises(assistant.AssistantUnavailable) as error:
            assistant.read_record(result['id'])
        self.assertEqual(error.exception.status, 409)

    async def test_invented_numbers_trigger_one_corrective_retry(self):
        invented = httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            'answer': 'The defect is 87.4 mm long and the score is 9.81.', 'evidence_ids': ['inspection']})}}]})
        provider = AsyncMock(side_effect=[invented, self.provider_answer()])
        with patch.object(assistant, 'provider_post', provider):
            result = await assistant.explain(self.inspection, 'How big is it?')
        self.assertEqual(provider.await_count, 2)
        self.assertIn('87.4', provider.call_args.kwargs['json']['messages'][-1]['content'])
        self.assertEqual(result['attempts'], 2)
        with patch.object(assistant, 'provider_post', AsyncMock(side_effect=[invented, invented])):
            with self.assertRaises(assistant.AssistantUnavailable):
                await assistant.explain(self.inspection, 'Different question', use_cache=False)

    async def test_identical_question_and_evidence_is_served_from_cache(self):
        provider = AsyncMock(return_value=self.provider_answer())
        with patch.object(assistant, 'provider_post', provider):
            first = await assistant.explain(self.inspection, 'Why flagged?')
            second = await assistant.explain(self.inspection, '  why   FLAGGED? ')
        self.assertEqual(provider.await_count, 1)
        self.assertFalse(first['cached']); self.assertTrue(second['cached'])
        self.assertEqual(first['id'], second['id'])

    async def test_provider_outage_falls_back_to_other_model_once(self):
        outage = assistant.AssistantUnavailable('NVIDIA could not complete the request (HTTP 503).', 502)
        provider = AsyncMock(side_effect=[outage, self.provider_answer()])
        with patch.object(assistant, 'provider_post', provider):
            result = await assistant.explain(self.inspection, 'Fallback?', use_cache=False)
        self.assertTrue(result['fallback_used'])
        self.assertEqual(result['model'], assistant.FALLBACK_MODEL)
        denied = assistant.AssistantUnavailable('NVIDIA denied this request.', 502)
        with patch.object(assistant, 'provider_post', AsyncMock(side_effect=[denied])):
            with self.assertRaises(assistant.AssistantUnavailable):
                await assistant.explain(self.inspection, 'Denied?', use_cache=False)

    def test_grounding_accepts_percent_forms_of_fractions(self):
        packet = {'facts': {'forecast': {'next_lot_defect_fraction': .301}, 'size_mm': [45.74, 35.6]}}
        self.assertEqual(assistant.ungrounded_numbers('Risk about 30.1% for a 45.7 mm region.', packet), [])
        self.assertEqual(assistant.ungrounded_numbers('Risk 64% on a 12.5 mm region.', packet), ['64%', '12.5'])

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

    async def test_provider_reports_missing_elevenlabs_permission_without_echoing_payload(self):
        response = httpx.Response(401, json={'detail': {'message': 'The API key is missing the permission text_to_speech to execute this operation.', 'request_id': 'private'}})
        with patch('httpx.AsyncClient.post', AsyncMock(return_value=response)):
            with self.assertRaises(assistant.AssistantUnavailable) as error:
                await assistant.provider_post('https://example.invalid', provider='ElevenLabs')
        self.assertIn('text_to_speech permission', str(error.exception))
        self.assertNotIn('private', str(error.exception))

    def test_invalid_response_path(self):
        with self.assertRaises(assistant.AssistantUnavailable):
            assistant.read_record('../../.env')

if __name__ == '__main__': unittest.main()
