"""Grounded part locations and real worker-activity contracts; no inference."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from app.schema import PhotoCheck, PhotoFinding, EvidenceReport
from app.evidence import passes, prompts, sampling, stage


class InspectionLocations(unittest.TestCase):
    def setUp(self):
        self.check = PhotoCheck(photo_id=9, path='x.jpg', filename='x.jpg',
                               width=1000, height=800, subject_box=[200, 100, 800, 700])

    def read(self, regions, cropped=False):
        return passes.parse_closeup(json.dumps({'component_regions': regions}), self.check, cropped=cropped)

    def test_visible_parts_are_not_issues_or_condition_claims(self):
        f = self.read([{'component':'windscreen_glass', 'box':[.2,.2,.3,.3]}])
        self.assertEqual(f.photo_id, 9)
        self.assertEqual(f.component_regions[0]['component'], 'windscreen_glass')
        self.assertEqual(f.issues, [])
        self.assertEqual(f.strengths, [])
        self.assertEqual(f.to_dict()['component_regions'][0]['box'], [.2,.2,.3,.3])

    def test_malformed_and_unknown_regions_are_not_drawn(self):
        for raw in [None, 'invalid', {}, [None], [{'component':'imaginary', 'box':[0,0,1,1]}],
                    [{'component':'engine_bay', 'box':[-1,0,.3,.3]}],
                    [{'component':'engine_bay', 'box':[0,0,0,.3]}]]:
            self.assertEqual(self.read(raw).component_regions, [], raw)

    def test_crop_locations_map_to_original_pixels(self):
        box = [.2,.2,.3,.3]
        actual = self.read([{'component':'engine_bay', 'box':box}], cropped=True).component_regions[0]['box']
        rect = passes.subject_crop_rect(self.check.subject_box, 1000, 800)
        self.assertEqual(actual, passes.map_box_to_original(box, rect, 1000, 800))
        self.assertNotEqual(actual, box)

    def test_sampling_preserves_geometry_without_averaging(self):
        a = self.read([{'component':'engine_bay','box':[.1,.1,.3,.3]}])
        b = self.read([{'component':'engine_bay','box':[.6,.6,.3,.3]}])
        merged, _ = sampling.combine_samples([a,b])
        self.assertEqual(merged.component_regions, a.component_regions)

    def test_schema_requires_locations_and_legacy_payloads_still_parse(self):
        schema = prompts.closeup_schema()
        self.assertIn('component_regions', schema['required'])
        self.assertEqual(passes.parse_closeup('{}', self.check, cropped=False).component_regions, [])

    def test_worker_announces_photo_before_result(self):
        events = []
        f = PhotoFinding(photo_id=9, view='engine_bay')
        with patch.object(sampling, 'closeup_consensus', return_value=(f, [], [])):
            stage._fan_out([], [self.check], '', Path('/tmp'),
                           lambda finding: events.append(('result', finding.photo_id)),
                           EvidenceReport(), on_activity=lambda msg: events.append(('started', msg['photo_id'])))
        self.assertEqual(events, [('started', 9), ('result', 9)])
