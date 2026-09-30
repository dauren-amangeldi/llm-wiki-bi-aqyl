"""One-user, one-pass smoke test using real AQYL routes; no AI generations."""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import gevent
from locust import HttpUser, events, task
from locust.exception import StopUser

TEST_HOST = "https://aqyl.test.bi.group"
RUN_ID = os.environ.get("AQYL_RUN_ID", "aqyl-locust-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
PROFILE = os.environ.get("AQYL_PROFILE", "preflight")
OUTPUT = Path(os.environ.get("AQYL_RESULTS_DIR", "/tmp/aqyl-locust-results")) / RUN_ID
if PROFILE not in {"preflight", "readonly"} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", RUN_ID):
    raise SystemExit("Invalid AQYL_PROFILE or AQYL_RUN_ID")


@events.init.add_listener
def validate_run(environment, **kwargs):
    options = environment.parsed_options
    if environment.host != TEST_HOST:
        raise SystemExit("This smoke test only accepts https://aqyl.test.bi.group")
    if options.num_users != 1 or not options.headless or options.master or options.worker:
        raise SystemExit("Use --headless -u 1; distributed/load runs need a separate profile")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    environment.aqyl_completed = False


@events.test_stop.add_listener
def require_completed_pass(environment, **kwargs):
    if not getattr(environment, "aqyl_completed", False):
        environment.process_exit_code = 1


class AqylSmokeUser(HttpUser):
    host = TEST_HOST

    def on_start(self):
        self.count = 0
        self.operations = {}
        self.skipped = []
        self.completed = False
        self.client.headers.update({"Accept-Language": "ru"})

    def record(self, event):
        with (OUTPUT / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def request(self, scenario, route, *, params=None, expected=200, shape=None,
                validator=None, method="GET", path=None, body=None, extra_headers=None):
        if method != "GET" and (method, route) != ("POST", "/api/v1/auth/load-test/token"):
            raise RuntimeError("Only read routes and test login are allowed")
        self.count += 1
        if self.count > 60:
            raise RuntimeError("Smoke request limit exceeded")
        if self.count > 1:
            gevent.sleep(1)  # Human pacing even between requests inside a task.
        request_id = f"{RUN_ID}-{self.count:03d}"
        operation = self.operations.setdefault(scenario, f"{RUN_ID}-{scenario}-{uuid4().hex[:6]}")
        headers = {"X-Run-ID": RUN_ID, "X-Scenario-ID": scenario,
                   "X-Operation-ID": operation, "X-Request-ID": request_id,
                   **(extra_headers or {})}
        start = time.perf_counter()
        with self.client.request(
            method, path or route, params=params, json=body, headers=headers,
            name=f"{scenario} {route}", timeout=(10, 30),
            allow_redirects=False, catch_response=True,
        ) as response:
            failure = None
            data = None
            if response.status_code != expected:
                failure = f"Expected HTTP {expected}, got {response.status_code}"
            elif response.headers.get("X-Request-ID") != request_id:
                failure = "Response lost X-Request-ID"
            elif shape is not None or validator is not None:
                try:
                    data = response.json()
                except ValueError:
                    failure = "Response is not JSON"
                if failure is None and shape is not None and not isinstance(data, shape):
                    failure = "Unexpected JSON shape"
                if failure is None and validator is not None and not validator(data):
                    failure = "Response failed scenario assertion"
            if failure:
                response.failure(failure)
            else:
                response.success()  # Expected 401 is success for the negative auth check.
            self.record({"timestamp": datetime.now(timezone.utc).isoformat(),
                         "run_id": RUN_ID, "scenario_id": scenario,
                         "operation_id": operation, "request_id": request_id,
                         "method": method, "route": route, "status_code": response.status_code,
                         "client_duration_ms": round((time.perf_counter() - start) * 1000, 2),
                         "response_bytes": len(response.content), "failure": failure})
        if failure:
            raise RuntimeError(f"{scenario} {route}: {failure}")
        return data

    def skip(self, scenario, reason):
        self.skipped.append({"scenario": scenario, "reason": reason})

    def preflight(self):
        self.request("SYS", "/api/healthz", shape=dict, validator=lambda d: d.get("status") == "ok")
        self.request("SYS", "/api/readyz", shape=dict,
                     validator=lambda d: d.get("status") == "ready" and
                     all(d.get("checks", {}).get(k) == "ok" for k in ("postgres", "redis", "object_store")))
        self.request("A01", "/api/v1/auth/config", shape=dict, validator=lambda d: d.get("enabled") is True)
        self.request("A01-negative", "/api/v1/cases", params={"limit": 200, "offset": 0}, expected=401)
        self.request("A01-negative", "/api/v1/auth/me", expected=401)

    def authenticate(self):
        token_path = os.environ.get("AQYL_ACCESS_TOKEN_FILE")
        secret_path = os.environ.get("AQYL_LOAD_LOGIN_SECRET_FILE")
        if bool(token_path) == bool(secret_path):
            raise RuntimeError("Set exactly one of AQYL_ACCESS_TOKEN_FILE or AQYL_LOAD_LOGIN_SECRET_FILE")
        if token_path:
            token = Path(token_path).read_text().strip()
            # Browser storage encodes strings as JSON; accept that copy as well.
            if token.startswith('"'):
                token = json.loads(token)
            if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token):
                raise RuntimeError("Access token file must contain one JWT")
            self.auth_source = "existing_sso_session"
        else:
            secret = Path(secret_path).read_text().strip()
            result = self.request("A01", "/api/v1/auth/load-test/token", method="POST",
                                  body={"email": "loadtest-0001@aqyl.test.invalid"},
                                  extra_headers={"X-Load-Test-Secret": secret}, shape=dict,
                                  validator=lambda d: bool(d.get("access_token")))
            token = result["access_token"]
            self.auth_source = "synthetic_test_account"
        self.client.headers["Authorization"] = "Bearer " + token
        self.request("A01", "/api/v1/auth/me", shape=dict,
                     validator=lambda d: bool(d.get("email")) if token_path else
                     d.get("email") == "loadtest-0001@aqyl.test.invalid" and d.get("role") == "employee")

    def readonly(self):
        self.authenticate()
        cases = []
        for offset in range(0, 1000, 200):
            page = self.request("A02", "/api/v1/cases", params={"limit": 200, "offset": offset}, shape=list)
            cases.extend(page)
            if len(page) < 200:
                break
        else:
            raise RuntimeError("More than 1000 cases: prepare a smaller smoke fixture")
        self.request("A02", "/api/v1/documents", params={"language": "ru"}, shape=list)
        self.request("A02", "/api/v1/tags", shape=list)
        sessions = self.request("A02", "/api/v1/twin/sessions", shape=list)
        self.request("POLL", "/api/v1/twin/readiness", shape=dict)
        consultations = self.request("A07", "/api/v1/advisor/consultations",
                                     params={"include_brief": "true", "completed_only": "true", "limit": 11, "offset": 0}, shape=list)
        self.request("A08-read", "/api/v1/notifications", shape=dict,
                     validator=lambda d: isinstance(d.get("items"), list) and "unread" in d)

        ready = next((c for c in cases if c.get("council", {}).get("ready_doc_ids")), None)
        if ready:
            case_id = quote(ready["id"], safe="")
            self.request("A04", "/api/v1/documents",
                         params=[("language", "ru"), *[("ids", d) for d in ready["doc_ids"]]], shape=list)
            self.request("A04", "/api/v1/cases/{case_id}/chat", path=f"/api/v1/cases/{case_id}/chat", shape=list)
            self.request("A04", "/api/v1/notes/{doc_id}", path=f"/api/v1/notes/{case_id}", shape=dict)
            artifacts = self.request("A04", "/api/v1/artifacts", params={"document_id": ready["id"], "language": "ru"}, shape=list)
            doc_id = quote(ready["council"]["ready_doc_ids"][0], safe="")
            document = self.request("A05", "/api/v1/documents/{document_id}/text",
                                    path=f"/api/v1/documents/{doc_id}/text", shape=dict,
                                    validator=lambda d: bool(d.get("content")))
            if document.get("slug"):
                self.request("A05", "/api/v1/wiki/{slug}/full",
                             path=f"/api/v1/wiki/{quote(document['slug'], safe='')}/full", shape=dict,
                             validator=lambda d: bool(d.get("content")))
            # FTS indexes bodies, not just titles: take an actual word from ready text.
            words = re.findall(r"[A-Za-zА-Яа-яЁё]{5,}", document["content"])
            if words:
                self.request("A03", "/api/v1/wiki", params={"q": words[0], "limit": 8}, shape=list,
                             validator=lambda d: len(d) > 0)
            else:
                self.skip("A03-positive", "Ready source has no suitable search word")
            artifact = next((a for a in artifacts if a.get("status") == "ready" and a.get("has_content")), None)
            if artifact:
                artifact_id = quote(artifact["artifact_id"], safe="")
                self.request("A06-read", "/api/v1/artifacts/{artifact_id}",
                             path=f"/api/v1/artifacts/{artifact_id}", params={"language": "ru"},
                             shape=dict, validator=lambda d: bool(d.get("versions")))
            else:
                self.skip("A06-read", "Selected ready case has no saved artifact")
        else:
            for scenario in ("A03-positive", "A04", "A05", "A06-read"):
                self.skip(scenario, "No accessible case with a ready material")
        self.request("A03-empty", "/api/v1/wiki", params={"q": "aqylnomatch" + uuid4().hex, "limit": 8},
                     shape=list, validator=lambda d: d == [])
        self.request("A07", "/api/v1/twin/personas", shape=dict,
                     validator=lambda d: isinstance(d.get("personas"), list) and
                     isinstance(d.get("presets"), list))
        if sessions:
            session_id = quote(sessions[0]["id"], safe="")
            self.request("A07", "/api/v1/twin/sessions/{session_id}/messages",
                         path=f"/api/v1/twin/sessions/{session_id}/messages", shape=list)
            summary = self.request("A07", "/api/v1/twin/sessions/{session_id}/summary",
                                   path=f"/api/v1/twin/sessions/{session_id}/summary", shape=dict)
            if not summary.get("summary"):
                self.skip("A07-saved-summary", "Selected session has no saved summary")
        else:
            self.skip("A07-session", "No accessible saved council session")
        if not consultations:
            self.skip("A07-consultation", "No saved consultations for the test account")

    @task
    def run_once(self):
        error = None
        try:
            self.preflight()
            if PROFILE == "readonly":
                self.readonly()
            self.completed = True
            self.environment.aqyl_completed = True
        except Exception as exc:
            # Exceptions contain only our fixed assertions; never persist bodies/credentials.
            error = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
            self.environment.process_exit_code = 1
        finally:
            (OUTPUT / "coverage.json").write_text(json.dumps({
                "run_id": RUN_ID, "profile": PROFILE, "completed": self.completed,
                "auth_source": getattr(self, "auth_source", None),
                "requests": self.count, "skipped": self.skipped, "error": error,
                "scope": "API smoke only; excludes UI, SSO, exports, writes and AI",
            }, indent=2, ensure_ascii=False))
            # Locust writes CSV once per second. Allow its final snapshot before
            # quitting this very short run; otherwise CSV can miss the last call.
            gevent.spawn_later(2, self.environment.runner.quit)
        raise StopUser()
