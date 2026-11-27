# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities privately via [GitHub Private Vulnerability Reporting](https://github.com/beautifulrem/FinSight/security/advisories/new) (Security tab → **Report a vulnerability**) rather than opening a public issue or pull request.

Include affected components, minimal reproduction steps, and expected impact. Never paste live credentials into a report—describe where the secret lives instead. Reports receive an acknowledgement within 7 days.

**In scope**: `query_intelligence/`, `sentiment/`, `frontend/`, `docker/`, and `deploy/`. Findings in upstream market-data or LLM providers should be reported to those vendors.

## Supported Versions

Security fixes are shipped on the latest commit on `master`.

## Secrets Policy

- **No secrets in git**: API keys, tokens, database passwords, and cookies must come from environment variables (`DEEPSEEK_API_KEY`, `TUSHARE_TOKEN`, `QI_API_KEYS`, `QI_ANON_COOKIE_SECRET`), a git-ignored local `.env`, or external Kubernetes Secrets ([docs/deployment.md](docs/deployment.md#secrets)).
- **Automated scanning**: `gitleaks` runs on staged diffs via [`.pre-commit-config.yaml`](.pre-commit-config.yaml) and over the **entire git history** in CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). [`.gitleaks.toml`](.gitleaks.toml) allows only narrowly scoped false positives and the single revoked historical finding below.
- **Historical disclosure (`25cac09`)**: An LLM API key was briefly committed to `config/app_config.json` in `25cac09` (2026-05-02) and removed 7 minutes later in `df385ba`. The repository owner confirmed provider-side revocation on 2026-09-30.
- **Credential redaction**: Logs, `/ready`, SSE/A2A error payloads, and evaluation provenance snapshots redact database DSNs, bearer tokens, and keys (`describe_error` / secret filters). Caller identities are logged only as SHA-256 prefixes.

## Security Boundaries

- **Authentication & tenant isolation**: `QI_API_KEYS` enables `X-API-Key` / `Authorization: Bearer` authentication. In `QI_PROFILE=production`, the FastAPI app and HTTP MCP server fail closed unless keys are configured (or `QI_ALLOW_ANONYMOUS=1` is set explicitly); non-loopback HTTP MCP binds also enforce key auth and optional `QI_MCP_ALLOWED_HOSTS`. Sessions, traces, and A2A contexts are strictly owner-scoped and return uniform `404` / `"Context not found."` responses to prevent cross-tenant ID enumeration. Keyless callers receive `403` on `/agent/traces`. The React UI stores API keys in `sessionStorage` by default (`localStorage` only on explicit opt-in).
- **Request caps, rate limiting & DoS defense**: Eager `BodyLimitMiddleware` caps all request bodies at `QI_MAX_REQUEST_BYTES` (default `1 MiB`) across REST, A2A, and HTTP MCP—including `Transfer-Encoding: chunked` payloads (`413`). Per-principal / per-IP rate limiting (`QI_RATE_LIMIT_PER_MINUTE`) is shared across replicas via Postgres (`finsight_rate_buckets`) with automatic per-replica fallback. Input length (`MAX_QUERY_LENGTH=2000`), recursion depth, regex backtracking, and in-memory stores/caches are strictly bounded.
- **Prompt injection & untrusted evidence**: Every tool is read-only. Retrieved text is wrapped in an untrusted-data envelope and filtered by NFKC/confusable normalization, lexical rules, and a character n-gram classifier. User inputs pass an input guard before NLU/LLM routing. Sentence-level claim verification (`verifier.py`) ties numeric statements to per-sentence `evidence_id` citations, while output safety and compliance guards (`output_safety.py`, `text_safety.py`, `lookalike.py`, `stated_dates.py`) strip trading instructions, attribute single-source or disputed document figures, hedge price/earnings/market-cap forecasts, and refuse out-of-coverage foreign listings and lookalike company names.
- **Supply-chain, archive & container hardening**: Shipped classical ML models are verified against `models/manifest.json` SHA-256 digests where present; archive extraction validates member paths and symlinks before unpacking; `docker/docker-compose.yml` binds local service ports to `127.0.0.1` without anonymous Grafana Admin; Docker/Kubernetes workloads run as non-root (`uid 10001`) with `readOnlyRootFilesystem`, dropped capabilities, and `NetworkPolicy` isolation (see [docs/deployment.md](docs/deployment.md)).
- **Audit & known limits**: Refusals, redactions, and output-safety edits are emitted as structured audit events and Prometheus counters (`finsight_audit_events_total`, `finsight_output_safety_edits_total`). Detailed guard rules, red-team benchmarks, and open trade-offs live in [docs/agent.md](docs/agent.md#safety-compliance-and-guardrails), [docs/agent-eval.md](docs/agent-eval.md), and [docs/known-issues.md](docs/known-issues.md).
