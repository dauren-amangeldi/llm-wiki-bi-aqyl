"""Create fresh local credentials for the test-only Locust login; never print them."""

import argparse
import os
import secrets
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_dir.expanduser().resolve()
    # Refuse overwrite, so credentials used by a deployment are not rotated by accident.
    folder.mkdir(mode=0o700, parents=True, exist_ok=False)
    login_secret = secrets.token_urlsafe(48)
    signing_secret = secrets.token_urlsafe(48)
    files = {
        "server.env": (
            "AUTH_ENABLED=true\nAPP_ENVIRONMENT=test\n"
            "PUBLIC_BASE_URL=https://aqyl.test.bi.group\nLOAD_TEST_AUTH_ENABLED=true\n"
            f"LOAD_TEST_LOGIN_SECRET={login_secret}\n"
            f"LOAD_TEST_SIGNING_SECRET={signing_secret}\n"
            "LOAD_TEST_USER_COUNT=1\nLOAD_TEST_TOKEN_TTL_S=900\n"
        ),
        "login-secret": login_secret + "\n",
    }
    for name, value in files.items():
        path = folder / name
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(value)
        print(path)


if __name__ == "__main__":
    main()
