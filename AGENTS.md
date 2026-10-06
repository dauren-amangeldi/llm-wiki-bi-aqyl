# Release reminder requested by the user

The `APP_ENVIRONMENT=test` condition in `Settings.load_test_auth_active` was
restored on 2026-10-06. Before deployment, verify the effective environment in
API, worker and beat: test accounts require `APP_ENVIRONMENT=test` on the test
environment; stage/production must have `LOAD_TEST_AUTH_ENABLED=false`.
The explicit test-domain restriction, two distinct secrets, and account
allowlist checks must remain in place. Keep reminding the user to verify these
deployment settings; do not waive the environment check again to fix startup.
This reminder does not authorize pushes or deployments.
