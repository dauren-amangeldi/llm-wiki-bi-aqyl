"""Offline checks: no requests to AQYL or SSO are made.

Run with the Locust virtualenv: python tests/load/check_auth_modes.py

Keep this out of the backend's pytest discovery: Locust has separate optional
dependencies and applies gevent monkey-patching when imported.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from locust.env import Environment

from locustfile import AqylSmokeUser, TEST_HOST


class AuthenticationModesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "credential"
        self.user = AqylSmokeUser(Environment(host=TEST_HOST))
        self.user.on_start()
        self.user.request = Mock(return_value={"access_token": "test.token.signature"})
        self.environ = patch.dict(os.environ, {}, clear=True)
        self.environ.start()
        self.addCleanup(self.environ.stop)

    def test_existing_session_only_calls_identity_route(self):
        for value in ("test.token.signature", json.dumps("test.token.signature")):
            with self.subTest(value_type="json" if value.startswith('"') else "raw"):
                self.user.request.reset_mock()
                self.path.write_text(value)
                os.environ["AQYL_ACCESS_TOKEN_FILE"] = str(self.path)
                self.user.authenticate()
                self.assertEqual(self.user.client.headers["Authorization"], "Bearer test.token.signature")
                self.user.request.assert_called_once()
                self.assertEqual(self.user.request.call_args.args, ("A01", "/api/v1/auth/me"))
                validator = self.user.request.call_args.kwargs["validator"]
                self.assertTrue(validator({"email": "operator@example.invalid", "role": "admin"}))
                self.assertFalse(validator({"role": "employee"}))
                self.assertEqual(self.user.auth_source, "existing_sso_session")

    def test_dedicated_login_keeps_employee_identity_check(self):
        self.path.write_text("fake-login-secret")
        os.environ["AQYL_LOAD_LOGIN_SECRET_FILE"] = str(self.path)
        self.user.authenticate()
        self.assertEqual(self.user.request.call_args_list[0].args[1], "/api/v1/auth/load-test/token")
        validator = self.user.request.call_args.kwargs["validator"]
        self.assertTrue(validator({"email": "loadtest-0001@aqyl.test.invalid", "role": "employee"}))
        self.assertFalse(validator({"email": "loadtest-0001@aqyl.test.invalid", "role": "admin"}))
        self.assertFalse(validator({"email": "operator@example.invalid", "role": "employee"}))

    def test_ambiguous_or_missing_credentials_do_not_send_requests(self):
        for env in ({}, {"AQYL_ACCESS_TOKEN_FILE": "a", "AQYL_LOAD_LOGIN_SECRET_FILE": "b"}):
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(RuntimeError):
                    self.user.authenticate()
        self.user.request.assert_not_called()

    def test_malformed_token_is_never_sent_or_echoed(self):
        for value in ("", "Bearer test.token.signature", "private-secret\nsecond-line", '""'):
            self.path.write_text(value)
            os.environ["AQYL_ACCESS_TOKEN_FILE"] = str(self.path)
            with self.assertRaisesRegex(RuntimeError, "^Access token file must contain one JWT$"):
                self.user.authenticate()
        self.user.request.assert_not_called()
        self.assertNotIn("Authorization", self.user.client.headers)


if __name__ == "__main__":
    unittest.main()
