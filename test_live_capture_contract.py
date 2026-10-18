import json
from pathlib import Path
import tempfile
import unittest
import live_dev_bridge as b
from test_live_dev_bridge import response


class CaptureContractTests(unittest.TestCase):
    def test_capture_keeps_answer_not_provider_reasoning(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            client = root / 'frontier_client.py'
            client.write_text('# Existing client is not executed for explicit replay\n', 'utf-8')
            source = root / 'source.json'
            prompt = root / 'prompt.txt'
            data = response()
            data['choices'][0]['message']['reasoning_content'] = 'PRIVATE_TRACE_NOT_FOR_PUBLIC_EVIDENCE'
            source.write_text(json.dumps(data), 'utf-8')
            prompt.write_text('same public task', 'utf-8')
            model = b.FrontierModel(client, root / 'captured', source, prompt)
            model.generate(b.MODEL, 'same public task')
            capture = (root / 'captured' / 'generation-1.json').read_text('utf-8')
            self.assertNotIn('PRIVATE_TRACE_NOT_FOR_PUBLIC_EVIDENCE', capture)
            self.assertIn('candidate', capture)
            self.assertTrue(json.loads(capture)['replayed_actual_capture'])


if __name__ == '__main__': unittest.main()
