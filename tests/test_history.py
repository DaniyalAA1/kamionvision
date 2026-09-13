"""History lookup, identity isolation and price arithmetic, without live API calls."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.history import HistoryReport, PlateObservation, apply_history, lookup, scan
from app.schema import Appraisal, Detection, GateReport, PhotoCheck, PriceEstimate, VehicleRead

VIN = 'NM0TEST1234567891'


def record(**changes):
    r = dict(plate='34 ABC 123', country='TR', vin=VIN, commercial=True,
             source='Synthetic test records', record_id='test-1', as_of=date.today().isoformat(),
             events=[dict(id='accident-1', date='2025-01-02', type='accident',
                          severity='moderate', repaired=True, description='Synthetic repaired collision')])
    r.update(changes)
    return r


def matched(r=None, **changes):
    return PlateObservation(0, 0, [0, 0, 100, 100], True, '34ABC123', 'TR', .99,
                            status='matched', record=r or record(), **changes)


def price():
    return PriceEstimate(point=100000, low=80000, high=120000, baseline_point=110000,
                         point_usd=2000, low_usd=1600, high_usd=2400)


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'records.json'

    def query(self, rows):
        self.path.write_text(json.dumps({'vehicles': rows}))
        return lookup('34ABC123', 'TR', self.path)

    def test_country_and_normalized_plate_match(self):
        status, r, _ = self.query([record()])
        self.assertEqual(status, 'matched')
        self.assertEqual(r['record_id'], 'test-1')
        self.assertEqual(lookup('34ABC123', 'BG', self.path)[0], 'not_found')

    def test_duplicate_stale_invalid_and_noncommercial(self):
        self.assertEqual(self.query([record(), record()])[0], 'ambiguous')
        self.assertEqual(self.query([record(as_of='2020-01-01')])[0], 'stale')
        self.assertEqual(self.query([record(as_of=(date.today()+timedelta(days=1)).isoformat())])[0], 'stale')
        self.assertEqual(self.query([record(source='')])[0], 'error')
        self.assertEqual(self.query([record(commercial=False)])[0], 'not_commercial')
        self.assertEqual(self.query([record(events=[{}])])[0], 'error')
        self.assertEqual(self.query([record(events=record()['events']*2)])[0], 'error')

    def test_unavailable_is_not_clean(self):
        self.assertEqual(lookup('34ABC123', 'TR', None)[0], 'unavailable')
        self.assertIn('does not establish', self.query([])[2])

    def test_owner_data_not_exposed(self):
        _, r, _ = self.query([record(owner_name='DO NOT DISPLAY')])
        self.assertNotIn('owner_name', r)


class AdjustmentTests(unittest.TestCase):
    def test_repaired_accident_calculation_keeps_baseline_and_conversions(self):
        p, h = price(), HistoryReport(observations=[matched()])
        apply_history(p, h, VehicleRead(vin=VIN))
        self.assertEqual((p.low, p.point, p.high), (73600, 92000, 110400))
        self.assertEqual(p.point_usd, 1840)
        self.assertEqual(p.baseline_point, 110000)
        self.assertIn('accident-1', h.reasoning[-1])
        self.assertIn('uncalibrated', h.reasoning[-1])
        apply_history(p, h, VehicleRead(vin=VIN))
        self.assertEqual(p.point, 92000)
        self.assertEqual(Appraisal(history=h).to_dict()['history']['adjustment_pct'], -8)

    def test_background_vehicle_and_mismatched_vin_cannot_change_price(self):
        for vin, is_subject in [(VIN, False), ('DIFFERENT', True), (None, True)]:
            p, o = price(), matched()
            o.is_subject = is_subject
            apply_history(p, HistoryReport(observations=[o]), VehicleRead(vin=vin))
            self.assertEqual(p.point, 100000)

    def test_conflicting_plates_and_mixed_sets_do_not_adjust(self):
        p, other = price(), matched()
        other.plate = '06ABC123'
        apply_history(p, HistoryReport(observations=[matched(), other]), VehicleRead(vin=VIN))
        self.assertEqual(p.point, 100000)
        apply_history(p, HistoryReport(observations=[matched()]), VehicleRead(vin=VIN), same_vehicle=False)
        self.assertEqual(p.point, 100000)

    def test_multiple_photos_and_accidents_are_not_stacked(self):
        r = record()
        r['events'].append(dict(r['events'][0], id='accident-2', severity='major'))
        p, h = price(), HistoryReport(observations=[matched(r), matched(r)])
        apply_history(p, h, VehicleRead(vin=VIN))
        self.assertEqual(p.point, 85000)

    def test_title_flood_mileage_and_unrepaired_accident_block_price(self):
        for kind in ['total_loss', 'salvage', 'flood', 'odometer_discrepancy', 'accident']:
            r = record(events=[dict(id='risk', date='2025-01-02', type=kind,
                                    repaired=False, description='Requires inspection')])
            p, h = price(), HistoryReport(observations=[matched(r)])
            apply_history(p, h, VehicleRead(vin=VIN))
            self.assertFalse(p.ok)
            self.assertTrue(h.blocked)
            self.assertEqual(p.point, 0)

    def test_no_events_no_premium(self):
        p, h = price(), HistoryReport(observations=[matched(record(events=[]))])
        apply_history(p, h, VehicleRead(vin=VIN))
        self.assertEqual(p.point, 100000)
        self.assertIn('not proof', h.reasoning[-1])


class PipelineTests(unittest.TestCase):
    def test_history_adjustment_reaches_final_pipeline_result(self):
        from app import pipeline
        from app.schema import EvidenceReport, IdentityVerdict, ReconcileReport
        h, p = HistoryReport(observations=[matched()]), price()
        ev = EvidenceReport(vehicle=VehicleRead(vin=VIN))
        with ExitStack() as stack:
            for target, value in [
                ('app.pipeline.gate_stage.run', GateReport()),
                ('app.pipeline.history_stage.scan', h),
                ('app.pipeline.history_stage.enabled', True),
                ('app.pipeline.evidence_stage.run', ev),
                ('app.pipeline.reconcile_stage.apply', ReconcileReport()),
                ('app.pipeline.identity_stage.decide', IdentityVerdict(status='confirmed')),
                ('app.pipeline.perception_stage.available', False),
                ('app.pipeline.price_model', None),
                ('app.pipeline.listings', None),
                ('app.pipeline.pricing.price_from_evidence', p),
            ]:
                stack.enter_context(patch(target, return_value=value))
            result = pipeline.appraise([])
        self.assertEqual(result.price.point, 92000)
        self.assertEqual(result.to_dict()['history']['after_point'], 92000)
        self.assertIn('73,600–110,400', result.headline)

    def test_history_disabled_keeps_existing_pipeline_without_extra_calls(self):
        from app import pipeline
        from app.schema import GateDecision
        with patch.dict('os.environ', {}, clear=True), \
             patch('app.pipeline.gate_stage.run', return_value=GateReport(decision=GateDecision.REFUSE_NOT_A_TRUCK)), \
             patch('app.pipeline.history_stage.scan') as history_scan:
            result = pipeline.appraise([])
        history_scan.assert_not_called()
        self.assertIsNone(result.history)

    def test_foreign_record_does_not_apply_turkish_policy(self):
        p = price()
        h = HistoryReport(observations=[matched(record(country='BG'))])
        apply_history(p, h, VehicleRead(vin=VIN))
        self.assertEqual(p.point, 100000)
        self.assertIn('other countries', h.reasoning[-1])


class ScanTests(unittest.TestCase):
    def test_multiple_vehicles_remain_bound_to_their_crops(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'two.jpg'
            Image.new('RGB', (200, 100), 'white').save(path)
            gate = GateReport(photos=[PhotoCheck(7, str(path), 'two.jpg', detections=[
                Detection('truck', .99, [0,0,100,100], .5, True),
                Detection('car', .99, [100,0,200,100], .5, False)])])
            class Backend:
                def complete(self, prompt, images, **kwargs):
                    with Image.open(images[0]) as img:
                        assert img.size == (100,100)
                    return SimpleNamespace(text=json.dumps(dict(plate='34 ABC 123', country='TR', confidence=.99)))
            with patch('app.history.lookup', return_value=('matched', record(), 'Matched')) as db:
                h = scan(gate, backend=Backend())
            self.assertEqual([o.is_subject for o in h.observations], [True, False])
            self.assertEqual([o.photo_id for o in h.observations], [7, 7])
            self.assertEqual(db.call_count, 1)

    def test_uncertain_reads_and_backend_failures_do_not_lookup(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'truck.jpg'
            Image.new('RGB', (100,100)).save(path)
            gate = GateReport(photos=[PhotoCheck(0, str(path), 'truck.jpg', detections=[
                Detection('truck', .99, [0,0,100,100], 1, True)])])
            for country, conf, expected in [('TR', .7, 'needs_confirmation'), (None, .99, 'needs_confirmation'), ('TR', float('nan'), 'error')]:
                backend = SimpleNamespace(complete=lambda *a, **k: SimpleNamespace(
                    text=json.dumps(dict(plate='34ABC123', country=country, confidence=conf))))
                with patch('app.history.lookup') as db:
                    h = scan(gate, backend=backend)
                self.assertEqual(h.observations[0].status, expected)
                db.assert_not_called()


if __name__ == '__main__':
    unittest.main()
