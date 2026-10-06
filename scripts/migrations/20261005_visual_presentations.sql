-- Additive migration; also applied by the existing create_all startup path.

-- Run before new API/worker/beat. No existing content is rewritten.

BEGIN;

CREATE TABLE IF NOT EXISTS artifact_revisions (
	artifact_id VARCHAR NOT NULL,
	language VARCHAR NOT NULL,
	revision INTEGER NOT NULL,
	content JSON NOT NULL,
	sources JSON NOT NULL,
	requested_by VARCHAR,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (artifact_id, language, revision),
	FOREIGN KEY(artifact_id) REFERENCES artifacts (artifact_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS visual_jobs (
	id VARCHAR NOT NULL,
	artifact_id VARCHAR NOT NULL,
	language VARCHAR NOT NULL,
	requested_by VARCHAR NOT NULL,
	request_key VARCHAR NOT NULL,
	status VARCHAR NOT NULL,
	snapshot JSON NOT NULL,
	plan JSON,
	error VARCHAR,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	deadline TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(artifact_id) REFERENCES artifacts (artifact_id) ON DELETE CASCADE,
	UNIQUE (request_key)
);

CREATE INDEX IF NOT EXISTS ix_visual_jobs_artifact_id ON visual_jobs (artifact_id);

CREATE INDEX IF NOT EXISTS ix_visual_jobs_requested_by ON visual_jobs (requested_by);

CREATE INDEX IF NOT EXISTS ix_visual_jobs_status ON visual_jobs (status);

CREATE TABLE IF NOT EXISTS visual_units (
	job_id VARCHAR NOT NULL,
	index INTEGER NOT NULL,
	status VARCHAR NOT NULL,
	token VARCHAR,
	attempts INTEGER NOT NULL,
	lease_until TIMESTAMP WITH TIME ZONE,
	output JSON,
	PRIMARY KEY (job_id, index),
	FOREIGN KEY(job_id) REFERENCES visual_jobs (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_visual_units_status ON visual_units (status);

COMMIT;
