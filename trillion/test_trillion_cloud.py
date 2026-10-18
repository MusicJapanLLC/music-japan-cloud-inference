import contextlib
import io
import json
import unittest
from unittest.mock import Mock, patch

import trillion_cloud as cloud

POSIX_REQUIRED = unittest.skipUnless(hasattr(cloud.signal, 'setitimer'), 'Cloud execution requires POSIX wall timeout')


def pricing(version=cloud.VERSION, output_peak='3.96'):
    return ('''<table>
    <tr><th colspan="3">MODEL</th><th>deepseek-flash</th><th>deepseek-v4-pro</th></tr>
    <tr><td colspan="3">BASE URL (OpenAI Format)</td><td colspan="2">https://api.deepseek.com</td></tr>
    <tr><td colspan="3">MODEL VERSION</td><td>unrelated-model</td><td>''' + version + '''</td></tr>
    <tr><td rowspan="6">PRICING(2)</td><td rowspan="2">1M INPUT TOKENS(CACHE HIT)</td><td>OFF-PEAK</td><td>$0.003</td><td>$0.022</td></tr>
    <tr><td>PEAK</td><td>$0.006</td><td>$0.044</td></tr>
    <tr><td rowspan="2">1M INPUT TOKENS(CACHE MISS)</td><td>OFF-PEAK</td><td>$0.15</td><td>$0.66</td></tr>
    <tr><td>PEAK</td><td>$0.30</td><td>$1.32</td></tr>
    <tr><td rowspan="2">1M OUTPUT TOKENS</td><td>OFF-PEAK</td><td>$0.60</td><td>$1.98</td></tr>
    <tr><td>PEAK</td><td>$1.20</td><td>$''' + output_peak + '''</td></tr>
    </table>''').encode()


def prepared(**changes):
    options = dict(public=True, allow_charge=True, max_input_bytes=1024,
                   max_output_tokens=128, timeout_seconds=30, max_cost_usd='0.01')
    options.update(changes)
    return cloud.prepare('Return only the integer result of 37 + 58.', **options)


def completion(**changes):
    value = {'id': 'test-completion-1', 'model': cloud.API_MODEL, 'created': 1792339200,
             'choices': [{'message': {'role': 'assistant', 'content': '95'}, 'finish_reason': 'stop'}],
             'usage': {'prompt_tokens': 25, 'completion_tokens': 1, 'total_tokens': 26}}
    value.update(changes)
    return value


class FakeTransport:
    def __init__(self, html=None, response=None):
        self.html = html if html is not None else pricing()
        self.response = response if response is not None else completion()
        self.posts = 0

    def mapping(self):
        return self.html

    def complete(self, body, key):
        self.posts += 1
        return cloud.canonical(self.response)


class CloudTests(unittest.TestCase):
    def test_plan_never_reads_credentials_files_or_network(self):
        with patch('builtins.open', side_effect=AssertionError('file read')), \
             patch.object(cloud, 'read_key', side_effect=AssertionError('secret read')), \
             patch.object(cloud, 'FixedHTTPS', side_effect=AssertionError('network')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cloud.main(['--api-key-file', '/missing']), 0)
        self.assertEqual(json.loads(output.getvalue())['network_requests'], 0)

    def test_explicit_public_charge_and_budget_guards(self):
        for options in ({'public': False}, {'allow_charge': False}, {'max_cost_usd': '0.00001'},
                        {'max_cost_usd': 'NaN'}, {'max_output_tokens': None},
                        {'timeout_seconds': 121}, {'max_input_bytes': 0}):
            with self.subTest(options=options), self.assertRaises(cloud.CloudError):
                prepared(**options)

    def test_input_byte_limit_includes_json_message_encoding(self):
        with self.assertRaisesRegex(cloud.CloudError, 'INPUT_BYTE_LIMIT'):
            cloud.prepare('あ' * 50, public=True, allow_charge=True, max_input_bytes=128,
                          max_output_tokens=128, timeout_seconds=30, max_cost_usd='0.01')

    def test_mapping_metadata_never_counts_as_inference(self):
        evidence = cloud.verify_mapping(pricing())
        self.assertEqual(evidence['model_version'], cloud.VERSION)
        self.assertFalse(evidence['inference_performed'])
        self.assertFalse(evidence['weight_revision_attested'])

    def test_neighboring_model_cannot_satisfy_version_check(self):
        html = pricing('New-Pro-Model').replace(b'unrelated-model', cloud.VERSION.encode())
        with self.assertRaisesRegex(cloud.CloudError, 'NO_LONGER_0813'):
            cloud.verify_mapping(html)

    @POSIX_REQUIRED
    def test_price_increase_or_model_drift_prevents_any_inference(self):
        for html in (pricing('different-model'), pricing(output_peak='4.00')):
            transport = FakeTransport(html=html)
            body, limits = prepared()
            result = cloud.execute_once(body, limits, 'example-test-key-value', transport=transport)
            self.assertEqual(transport.posts, 0)
            self.assertFalse(result['inference_verified'])
            self.assertFalse(result['charge_may_have_occurred'])

    @POSIX_REQUIRED
    def test_real_response_fields_separate_from_mapping_evidence(self):
        body, limits = prepared(); transport = FakeTransport()
        result = cloud.execute_once(body, limits, 'example-test-key-value', transport=transport)
        self.assertTrue(result['inference_verified'])
        self.assertFalse(result['weight_revision_attested'])
        self.assertEqual(result['response_evidence']['content'], '95')
        self.assertEqual(result['response_evidence']['id'], 'test-completion-1')
        self.assertEqual(result['response_evidence']['usage']['total_tokens'], 26)
        self.assertEqual(result['inference_requests_attempted'], 1)
        self.assertEqual(transport.posts, 1)
        self.assertEqual(result['request_sha256'], cloud.hashlib.sha256(cloud.canonical(body)).hexdigest())
        self.assertIn('inference_elapsed_seconds', result)
        self.assertNotIn('example-test-key-value', json.dumps(result))

    @POSIX_REQUIRED
    def test_truncation_wrong_model_and_missing_usage_are_not_success(self):
        truncated = completion(); truncated['choices'][0]['finish_reason'] = 'length'
        for response in (truncated, completion(model='smaller-model'), completion(usage=None)):
            body, limits = prepared(); transport = FakeTransport(response=response)
            result = cloud.execute_once(body, limits, 'example-test-key-value', transport=transport)
            self.assertFalse(result['inference_verified'])
            self.assertTrue(result['charge_may_have_occurred'])
            self.assertEqual(result['status'], 'inference_response_rejected')
            self.assertEqual(result['response_evidence']['content'], '95')
            self.assertEqual(transport.posts, 1)

    @POSIX_REQUIRED
    def test_reflected_credential_is_not_logged(self):
        body, limits = prepared(); response = completion()
        response['choices'][0]['message']['content'] = 'example-test-key-value'
        result = cloud.execute_once(body, limits, 'example-test-key-value', transport=FakeTransport(response=response))
        self.assertEqual(result['error_code'], 'AUTH_KEY_REFLECTED_BY_PROVIDER')
        self.assertNotIn('example-test-key-value', json.dumps(result))

    def test_redirect_is_not_followed_or_body_logged_and_second_post_refused(self):
        response = Mock(status=307); connection = Mock(); connection.getresponse.return_value = response
        transport = cloud.FixedHTTPS(10)
        with patch.object(cloud.http.client, 'HTTPSConnection', return_value=connection) as factory:
            with self.assertRaisesRegex(cloud.CloudError, 'HTTP_307'):
                transport.complete(b'{}', 'example-test-key-value')
            with self.assertRaisesRegex(cloud.CloudError, 'INFERENCE_REQUEST_LIMIT'):
                transport.complete(b'{}', 'example-test-key-value')
        self.assertEqual(factory.call_count, 1)
        factory.assert_called_with('api.deepseek.com', 443, timeout=10)
        response.read.assert_not_called(); connection.close.assert_called_once()

    @POSIX_REQUIRED
    def test_timeout_attempt_is_not_retried_and_cannot_claim_free(self):
        body, limits = prepared(); transport = FakeTransport()
        def timeout(_body, _key):
            transport.posts += 1
            raise cloud.CloudError('WALL_TIMEOUT')
        transport.complete = timeout
        result = cloud.execute_once(body, limits, 'example-test-key-value', transport=transport)
        self.assertEqual(result['error_code'], 'WALL_TIMEOUT')
        self.assertTrue(result['charge_may_have_occurred'])
        self.assertFalse(result['inference_verified'])
        self.assertEqual(transport.posts, 1)

    def test_execution_without_main_thread_wall_timeout_fails_before_network(self):
        body, limits = prepared(); transport = FakeTransport()
        with patch.object(cloud.threading, 'current_thread', return_value=object()):
            result = cloud.execute_once(body, limits, 'example-test-key-value', transport=transport)
        self.assertEqual(result['error_code'], 'WALL_TIMEOUT_UNAVAILABLE')
        self.assertEqual(result['metadata_requests_attempted'], 0)
        self.assertEqual(transport.posts, 0)

    def test_late_timeout_or_exception_clears_previously_set_success(self):
        for error in (cloud.CloudError('WALL_TIMEOUT'), RuntimeError('private upstream detail')):
            @contextlib.contextmanager
            def late_failure(_seconds):
                yield
                raise error
            body, limits = prepared(); transport = FakeTransport()
            with patch.object(cloud, 'wall_timeout', side_effect=late_failure):
                result = cloud.execute_once(body, limits, 'example-test-key-value', transport=transport)
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['inference_verified'])
            self.assertTrue(result['charge_may_have_occurred'])
            self.assertNotIn('private upstream detail', json.dumps(result))

    @POSIX_REQUIRED
    def test_json_escaped_secret_in_values_or_keys_is_not_persisted(self):
        key = 'example-test-key-value'
        for response in (completion(extra={'nested': [{'sensitive': key}]}), completion(**{key: 'marker'})):
            raw = cloud.canonical(response).replace(b'example', b'\\u0065xample')
            self.assertNotIn(key.encode(), raw)
            body, limits = prepared(); transport = FakeTransport()
            transport.complete = lambda _body, _key: raw
            result = cloud.execute_once(body, limits, key, transport=transport)
            self.assertEqual(result['error_code'], 'AUTH_KEY_REFLECTED_BY_PROVIDER')
            self.assertIsNone(result['response_evidence'])
            self.assertNotIn(key, json.dumps(result))


if __name__ == '__main__':
    unittest.main()
