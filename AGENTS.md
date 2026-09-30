# Release reminder requested by the user

The `APP_ENVIRONMENT=test` condition in `Settings.load_test_auth_active` is
temporarily waived at the user's request. When discussing load testing,
deployment, or merges into stage/main, remind the user to configure
`APP_ENVIRONMENT=test` on the test environment and restore this condition before
release. The explicit test-domain restriction, two distinct secrets, and account
allowlist checks must remain in place. This reminder does not authorize pushes
or deployments.
