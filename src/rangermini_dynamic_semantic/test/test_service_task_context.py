import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from rangermini_dynamic_semantic.service_task_context import load_service_directory


class ServiceDirectoryTests(unittest.TestCase):
    def load(self, entries):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'directory.json'
            p.write_text(json.dumps(dict(version=1, source='simulation registration', entities=entries)))
            return load_service_directory(p, SimpleNamespace(doors={'D617': {}, 'D614': {}}))

    def test_registration_keeps_visual_identity_unverified(self):
        result = self.load([dict(id='student', name='小陈', kind='person', room='617')])
        self.assertEqual(result['entities'][0]['knowledge_source'], 'registration')
        self.assertFalse(result['entities'][0]['visual_identity_verified'])

    def test_unknown_room_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unregistered room'):
            self.load([dict(id='student', name='小陈', kind='person', room='999')])

    def test_duplicate_alias_requires_clarification_context(self):
        result = self.load([dict(id='a', name='陈同学', aliases=['小陈'], kind='person', room='617'),
                            dict(id='b', name='小陈', kind='person', room='614')])
        self.assertEqual(result['ambiguous_labels']['小陈'], ['a', 'b'])
        self.assertEqual(len(result['entities']), 2)

    def test_modes_are_not_registration_entities(self):
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            self.load([dict(id='m', name='横移', kind='motion_mode', room='617')])

if __name__ == '__main__':
    unittest.main()
