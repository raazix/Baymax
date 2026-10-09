import csv
import io
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from analytics import historical_sensors
from analytics.lot_history import build_process_context
from apps.api import main
from database.repository import Repository


def history_csv(machine='M-T', lots=30, drift_from=24, drift=40.0):
    """Synthetic 12-hour lots; temperature steps up from lot `drift_from` onward."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['timestamp', 'machine_id', 'sensor', 'value', 'unit', 'lot_id'])
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    for lot in range(lots):
        for step in range(4):
            at = start + timedelta(hours=lot * 12 + step * 3)
            offset = drift if lot >= drift_from else 0.0
            wobble = (step - 1.5) * .5 + (lot % 3) * .4
            for sensor, unit, value in (('temperature_c', 'C', 705 + offset + wobble), ('pressure_bar', 'bar', 100 + wobble / 2),
                                        ('vibration_mm_s', 'mm/s', 2.0 + wobble / 20), ('machine_speed_rpm', 'rpm', 1200 + wobble * 3)):
                writer.writerow([at.isoformat().replace('+00:00', 'Z'), machine, sensor, f'{value:.3f}', unit, f'{machine}-LOT{lot:03d}'])
    return buffer.getvalue().encode()


class LotHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.repo = Repository('sqlite:///' + (Path(self.directory.name) / 'history.db').as_posix())

    def tearDown(self):
        self.repo.engine.dispose()
        self.directory.cleanup()

    def import_history(self, **kwargs):
        dataset, readings = historical_sensors.parse_csv(history_csv(**kwargs), 'history.csv', 'SYNTHETIC unit-test history')
        return self.repo.import_sensor_dataset(dataset, readings)

    def test_import_stores_every_reading_and_is_idempotent(self):
        # Regression: readings were inserted without their dataset hash and before the manifest row.
        first = self.import_history()
        self.assertEqual(first['rows_inserted'], 30 * 4 * 4)
        self.assertTrue(self.import_history()['already_imported'])
        self.assertEqual(self.repo.sensor_catalog()['reading_count'], 480)

    def test_lot_context_uses_latest_lot_prior_lots_and_flags_drift(self):
        self.import_history()
        context = build_process_context(self.repo.lot_aggregates('M-T', 60), 'M-T')
        self.assertEqual(context['lot_id'], 'M-T-LOT029')
        self.assertEqual(context['prior_lots'], ['M-T-LOT027', 'M-T-LOT028'])
        self.assertEqual(len(context['forecast_history']), 2)
        self.assertGreater(context['telemetry']['temperature_c'], 740)
        self.assertEqual(context['drift'][0]['sensor'], 'temperature_c')
        self.assertGreater(context['drift'][0]['z_score'], 3)
        self.assertTrue(context['synthetic'])
        self.assertIn('SYNTHETIC', context['telemetry_source'])

    def test_stable_machine_has_no_drift_and_unknown_machine_has_no_context(self):
        self.import_history(drift=0.0)
        self.assertEqual(build_process_context(self.repo.lot_aggregates('M-T', 60), 'M-T')['drift'], [])
        self.assertIsNone(build_process_context(self.repo.lot_aggregates('M-NONE', 60), 'M-NONE'))


class LotHistoryApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous = main.repo
        main.repo = Repository('sqlite:///' + (Path(self.directory.name) / 'api.db').as_posix())
        dataset, readings = historical_sensors.parse_csv(history_csv(), 'history.csv', 'SYNTHETIC unit-test history')
        main.repo.import_sensor_dataset(dataset, readings)
        self.client = TestClient(main.app)

    def tearDown(self):
        main.repo.engine.dispose()
        main.repo = self.previous
        self.directory.cleanup()

    def test_lot_context_endpoint(self):
        response = self.client.get('/api/history/lot-context', params={'machine_id': 'M-T'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['lot_id'], 'M-T-LOT029')
        self.assertEqual(self.client.get('/api/history/lot-context', params={'machine_id': 'M-NONE'}).status_code, 404)


if __name__ == '__main__':
    unittest.main()


class FleetRiskTests(unittest.TestCase):
    def test_drifting_machine_ranks_first_with_cause_and_action(self):
        from analytics.fleet_risk import rank_machines
        with tempfile.TemporaryDirectory() as directory:
            repo = Repository('sqlite:///' + (Path(directory) / 'fleet.db').as_posix())
            for machine, drift in (('M-HOT', 50.0), ('M-OK', 0.0)):
                dataset, readings = historical_sensors.parse_csv(history_csv(machine=machine, drift=drift), f'{machine}.csv', 'SYNTHETIC test')
                repo.import_sensor_dataset(dataset, readings)
            result = rank_machines(repo)
            repo.engine.dispose()
        first, second = result['ranking']
        self.assertEqual((first['machine_id'], first['rank']), ('M-HOT', 1))
        self.assertEqual(first['root_cause'], 'thermal_process_drift')
        self.assertGreater(first['predicted_defect_fraction'], second['predicted_defect_fraction'])
        self.assertIn('thermocouple', first['recommended_action'])
        self.assertEqual(first['drift'][0]['sensor'], 'temperature_c')
        self.assertEqual(len(first['batches']), 6)
        self.assertIn('uncalibrated', result['limitations'])
