"""Public acceptance contract; fixtures contain synthetic data only."""
import ast
import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from render_analysis import render_report

SOURCE = Path(__file__).resolve().parents[1] / 'render_analysis.py'
DOCUMENT = {
    'schema': 'redline-upper-analysis/v1',
    'created_at': '2026-10-18T09:00:00+00:00',
    'source_model': 'Qwen/Qwen3.8-Flash-Next',
    'evidence_ids': ['case:tenant-check', 'artifact:' + 'a' * 64],
    'model_analysis': {
        'summary': '所有者確認の不足が疑われます。実行時の影響は未確認です。',
        'findings': [{
            'evidence_ids': ['case:tenant-check'],
            'assessment': '資料上の所有者判定を確認してください。',
            'recommendation': '認可middlewareと対照ケースを確認してください。',
        }, {
            'evidence_ids': ['artifact:' + 'a' * 64],
            'assessment': '保存済み結果と実行対象の版の対応は未確認です。',
            'recommendation': '実行版のhashと独立再検証receiptを照合してください。',
        }],
        'limitations': ['実環境での再現は確認していません。',
                        '静的資料だけでは実際のアクセス権を確定できません。'],
    },
    'site_safe': None,
    'target_actions_executed': False,
    'generation_attempts': [{'receipt': {'final_content': 'DO_NOT_RENDER_RAW_RECEIPT'}}],
}


class PassiveHTML(HTMLParser):
    allowed = {'html','head','meta','title','body','main','h1','h2','h3','section',
               'article','p','ul','ol','li','dl','dt','dd','table','thead','tbody',
               'tr','th','td','pre','code','span','div','br','hr','strong','em','small'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.text = []
        self.html_lang = None
        self.utf8 = False
        self.body_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag not in self.allowed:
            raise AssertionError('Active or external-resource tag: ' + tag)
        self.tags.append(tag)
        keys = [key for key, _ in attrs]
        if len(keys) != len(set(keys)):
            raise AssertionError('Duplicate attribute')
        for key, value in attrs:
            if not ((tag == 'html' and key == 'lang') or (tag == 'meta' and key == 'charset')):
                raise AssertionError('Attribute is not allowlisted: ' + key)
            if not isinstance(value, str) or not value:
                raise AssertionError('Missing attribute value')
        attrs = dict(attrs)
        if tag == 'html':
            self.html_lang = attrs.get('lang')
        if tag == 'meta' and attrs.get('charset', '').casefold().replace('-', '') == 'utf8':
            self.utf8 = True
        if tag == 'body':
            self.body_depth += 1

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == 'body':
            self.body_depth = max(0, self.body_depth - 1)

    def handle_data(self, data):
        if self.body_depth:
            self.text.append(data)


class ReportAcceptance(unittest.TestCase):
    def test_passive_guard_rejects_css_hidden_and_resource_attributes(self):
        rejected = [
            '<body background="https://example.invalid/pixel">text</body>',
            '<style>body {background-image:u\\72l(https://example.invalid/pixel)}</style>',
            '<p style="display:none">finding</p>',
            '<p hidden>finding</p>',
            '<p class="hidden">finding</p>',
            '<html lang="ja" lang="en">',
            '<meta charset="UTF-8" http-equiv="refresh" content="0;url=https://example.invalid">',
            '<p lang="ja">unexpected attribute</p>',
        ]
        for markup in rejected:
            with self.subTest(markup=markup), self.assertRaises(AssertionError):
                parser = PassiveHTML()
                parser.feed(markup)
                parser.close()

    def test_passive_guard_accepts_minimal_ordinary_html(self):
        parser = PassiveHTML()
        parser.feed('<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8"><title>報告</title></head>'
                    '<body><main><h1>概要</h1><p>資料の指摘です。</p><ul><li>根拠</li></ul></main></body></html>')
        parser.close()
        self.assertEqual(parser.html_lang, 'ja')
        self.assertTrue(parser.utf8)

    def page(self, document):
        html = render_report(document)
        self.assertIsInstance(html, str)
        self.assertLessEqual(len(html.encode('utf-8')), 256000)
        self.assertTrue(html.lstrip().casefold().startswith('<!doctype html>'))
        parsed = PassiveHTML()
        parsed.feed(html)
        parsed.close()
        self.assertEqual(parsed.html_lang, 'ja')
        self.assertTrue(parsed.utf8)
        for tag in ('html','head','title','body'):
            self.assertIn(tag, parsed.tags)
        return html, '\n'.join(parsed.text)

    def test_standalone_japanese_report_has_real_analysis_and_sections(self):
        _, visible = self.page(copy.deepcopy(DOCUMENT))
        required = ['概要','指摘','証拠','モデル','制約', DOCUMENT['source_model'],
                    DOCUMENT['model_analysis']['summary'], *DOCUMENT['model_analysis']['limitations']]
        for finding in DOCUMENT['model_analysis']['findings']:
            required.extend([finding['assessment'], finding['recommendation'], *finding['evidence_ids']])
        for value in required:
            self.assertIn(value, visible)

    def test_evidence_ids_are_visible_as_text(self):
        _, visible = self.page(copy.deepcopy(DOCUMENT))
        for value in DOCUMENT['evidence_ids']:
            self.assertIn(value, visible)

    def test_every_displayed_string_is_escaped_not_active_markup(self):
        value = copy.deepcopy(DOCUMENT)
        attacks = [
            '<script>alert("summary")</script>',
            '<img src="https://example.invalid/pixel" onerror="alert(1)">',
            '\"><svg/onload=alert(2)> & 日本語',
            '<iframe srcdoc="<script>alert(3)</script>"></iframe>',
        ]
        value['model_analysis']['summary'] = attacks[0]
        value['model_analysis']['findings'][0]['assessment'] = attacks[1]
        value['model_analysis']['findings'][0]['recommendation'] = attacks[2]
        value['model_analysis']['limitations'] = [attacks[3]]
        _, visible = self.page(value)
        for attack in attacks:
            self.assertIn(attack, visible)

    def test_model_name_and_evidence_ids_are_escaped(self):
        value = copy.deepcopy(DOCUMENT)
        value['source_model'] = '<script>alert("model")</script>'
        value['evidence_ids'] = ['case:"><img src=x onerror=alert(4)>']
        for finding in value['model_analysis']['findings']:
            finding['evidence_ids'] = value['evidence_ids'][:]
        _, visible = self.page(value)
        self.assertIn(value['source_model'], visible)
        self.assertIn(value['evidence_ids'][0], visible)

    def test_input_is_not_mutated_and_raw_receipts_are_not_rendered(self):
        value = copy.deepcopy(DOCUMENT)
        before = copy.deepcopy(value)
        html, _ = self.page(value)
        self.assertEqual(value, before)
        self.assertNotIn('DO_NOT_RENDER_RAW_RECEIPT', html)

    def test_empty_findings_is_valid_without_a_safety_claim(self):
        value = copy.deepcopy(DOCUMENT)
        value['model_analysis']['findings'] = []
        _, visible = self.page(value)
        self.assertIn('安全判定は未実施', visible)
        self.assertIsNone(value['site_safe'])

    def test_unknown_or_private_bundle_schema_is_rejected(self):
        for schema in ('redline-dossier/v2', 'mj.company-db-evidence/1.0', 'other/v1', None):
            value = copy.deepcopy(DOCUMENT)
            value['schema'] = schema
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                render_report(value)

    def test_required_top_level_fields_are_validated(self):
        for key in ('schema','source_model','evidence_ids','model_analysis'):
            value = copy.deepcopy(DOCUMENT)
            del value[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                render_report(value)

    def test_summary_limitations_and_finding_shapes_are_validated(self):
        for field, invalid in [('summary', []), ('summary', ''), ('findings', {}),
                               ('limitations', []), ('limitations', [None])]:
            value = copy.deepcopy(DOCUMENT)
            value['model_analysis'][field] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                render_report(value)
        for key in ('assessment','recommendation','evidence_ids'):
            value = copy.deepcopy(DOCUMENT)
            del value['model_analysis']['findings'][0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                render_report(value)

    def test_unknown_finding_evidence_is_rejected(self):
        value = copy.deepcopy(DOCUMENT)
        value['model_analysis']['findings'][0]['evidence_ids'] = ['case:invented']
        with self.assertRaises(ValueError):
            render_report(value)

    def test_non_string_source_and_evidence_are_rejected(self):
        for key, invalid in [('source_model', None), ('source_model',''),
                             ('evidence_ids', 'case:tenant-check'), ('evidence_ids',[None])]:
            value = copy.deepcopy(DOCUMENT)
            value[key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                render_report(value)

    def test_source_uses_only_approved_stdlib_without_dynamic_execution(self):
        allowed = {'argparse','html','json','pathlib','re','sys','typing','collections','copy'}
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split('.')[0], allowed)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn((node.module or '').split('.')[0], allowed | {'__future__'})
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {'eval','exec','compile','__import__'})

    def cli(self, input_file, output_file):
        return subprocess.run([sys.executable, str(SOURCE), '--input', str(input_file), '--out', str(output_file)],
                              shell=False, capture_output=True, text=True, timeout=10)

    def test_cli_writes_fresh_utf8_html_for_space_and_japanese_paths(self):
        with tempfile.TemporaryDirectory(prefix='REDLINE 報告 ') as folder:
            root = Path(folder)
            source, output = root/'公開 解析.json', root/'確認 報告.html'
            source.write_text(json.dumps(DOCUMENT, ensure_ascii=False), encoding='utf-8')
            before = source.read_bytes()
            result = self.cli(source, output)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(source.read_bytes(), before)
            parsed = PassiveHTML()
            parsed.feed(output.read_text(encoding='utf-8'))
            self.assertIn(DOCUMENT['model_analysis']['summary'], '\n'.join(parsed.text))

    def test_cli_does_not_overwrite_existing_output_or_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/'analysis.json', root/'existing.html'
            source.write_text(json.dumps(DOCUMENT), encoding='utf-8')
            original = source.read_bytes()
            output.write_text('PRESERVE EXISTING USER FILE', encoding='utf-8')
            self.assertNotEqual(self.cli(source, output).returncode, 0)
            self.assertEqual(output.read_text(), 'PRESERVE EXISTING USER FILE')
            self.assertNotEqual(self.cli(source, source).returncode, 0)
            self.assertEqual(source.read_bytes(), original)

    def test_cli_invalid_json_never_creates_a_successful_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/'invalid.json', root/'report.html'
            source.write_text('not JSON', encoding='utf-8')
            self.assertNotEqual(self.cli(source, output).returncode, 0)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
