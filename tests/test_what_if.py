import unittest
from fastapi.testclient import TestClient
from apps.api.main import app

NOMINAL = {'temperature_c': 705, 'pressure_bar': 100, 'vibration_mm_s': 2.0, 'machine_speed_rpm': 1200}
THERMAL = {'temperature_c': 753, 'pressure_bar': 103, 'vibration_mm_s': 2.3, 'machine_speed_rpm': 1200}


class WhatIfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def run_case(self, readings):
        response = self.client.post('/api/analytics/what-if', json=readings)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_cause_forecast_and_risk_react_to_readings(self):
        nominal, thermal = self.run_case(NOMINAL), self.run_case(THERMAL)
        self.assertEqual(nominal['rca']['hypothesis'], 'nominal_process')
        self.assertEqual(thermal['rca']['hypothesis'], 'thermal_process_drift')
        self.assertGreater(thermal['forecast']['predicted_defect_fraction'], nominal['forecast']['predicted_defect_fraction'])
        self.assertGreater(thermal['uncertainty']['breach_probability'], nominal['uncertainty']['breach_probability'])
        self.assertEqual(thermal['rca']['feature_contributions'][0]['feature'], 'temperature_c')

    def test_histogram_is_the_same_seeded_sample_as_the_summary(self):
        result = self.run_case(THERMAL)
        counts, edges = result['histogram']['counts'], result['histogram']['edges']
        self.assertEqual(sum(counts), result['uncertainty']['simulations'])
        tolerance = result['uncertainty']['tolerance']
        over = sum(c for c, left in zip(counts, edges) if left >= tolerance)
        self.assertAlmostEqual(over / sum(counts), result['uncertainty']['breach_probability'], delta=.001)

    def test_deterministic_and_validated(self):
        self.assertEqual(self.run_case(THERMAL), self.run_case(THERMAL))
        self.assertEqual(self.client.post('/api/analytics/what-if', json={**NOMINAL, 'temperature_c': 5000}).status_code, 422)
        self.assertEqual(self.client.post('/api/analytics/what-if', json={**NOMINAL, 'extra': 1}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
