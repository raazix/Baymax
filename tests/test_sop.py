import json
import unittest

from actions.sop_engine import HYPOTHESIS_ACTIONS, SOP_VERSION, recommend


class SOPTests(unittest.TestCase):
    @staticmethod
    def analytics(hypothesis, confidence=.99):
        return {'rca': {'hypothesis': hypothesis, 'confidence': confidence},
                'forecast': {'predicted_defect_fraction': .72},
                'uncertainty': {'interval_95': [.61, .81], 'tolerance': .5, 'breach_probability': .84}}

    def test_concrete_hypothesis_actions(self):
        for hypothesis, step in HYPOTHESIS_ACTIONS.items():
            with self.subTest(hypothesis=hypothesis):
                action = recommend([{'severity': {'level': 'medium'}}], self.analytics(hypothesis))
                self.assertIn(step, action['steps'])
                self.assertEqual(action['status'], 'pending')
                self.assertTrue(action['required'])
                self.assertEqual(action['sop_version'], SOP_VERSION)

    def test_critical_lot_containment_independent_of_cause_confidence(self):
        for hypothesis in ['nominal_process', 'unknown', *HYPOTHESIS_ACTIONS]:
            for confidence in [None, 0, 1]:
                action = recommend([{'severity': {'level': 'critical'}}], self.analytics(hypothesis, confidence))
                self.assertEqual(action['containment'], 'affected_lot')
                self.assertIn('Quarantine the affected lot', action['text'])
        self.assertEqual(recommend([{'severity': 'CRITICAL'}], None)['containment'], 'affected_lot')

    def test_nominal_with_defect_requires_manual_cause_review(self):
        action = recommend([{'severity': {'level': 'low'}}], self.analytics('nominal_process'))
        self.assertIn('investigate the cause', action['text'])
        self.assertNotIn('Inspect tooling', action['text'])
        self.assertEqual(action['containment'], 'affected_part')

    def test_forecast_sampling_does_not_require_machine_action_without_defects(self):
        action = recommend([], self.analytics('thermal_process_drift'))
        self.assertFalse(action['required'])
        self.assertEqual(action['status'], 'not_required')
        self.assertIn('synthetic forecast', action['text'])
        self.assertNotIn('temperature-control loop', action['text'])
        self.assertIn('no PLC connection', action['execution'])
        self.assertEqual(action['supporting_evidence']['simulated_predictive_interval_95'], [.61, .81])
        self.assertFalse(action['supporting_evidence']['production_validated'])

    def test_bad_forecast_is_omitted_and_strict_json_safe(self):
        action = recommend([], {'forecast': {'risk': float('nan')}, 'uncertainty': {'interval_95': [0, float('inf')]}})
        self.assertNotIn('predicted_defect_fraction', action['supporting_evidence'])
        json.dumps(action, allow_nan=False)

    def test_missing_analytics_still_contains_critical_defect(self):
        action = recommend([{'severity': {'level': 'critical'}}], None)
        self.assertTrue(action['required'])
        self.assertEqual(action['status'], 'pending')
        self.assertEqual(action['containment'], 'affected_lot')


if __name__ == '__main__':
    unittest.main()
