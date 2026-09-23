CREATE TABLE search_memory_snapshots (
    run_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    snapshot TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (run_id, version)
);
