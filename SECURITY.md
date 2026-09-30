# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue or pull request:

- use GitHub's **private vulnerability reporting** on this repository (Security tab → "Report a
  vulnerability"), or
- email the maintainer at the address on their GitHub profile, with "FinSight security" in the subject.

Include what you found, how to reproduce it (request, input or commit), and the impact you expect. If
the report involves a credential, describe where it is and do **not** paste the value. You should get
an acknowledgement within 7 days. Please give us a reasonable time to fix the issue before disclosing
it publicly; we will credit you in the fix unless you prefer otherwise.

In scope: the API and agent (`query_intelligence/`), the web UI, the container image and the
deployment manifests in this repository. Findings in upstream data providers or third-party
services should go to those vendors.

## Secrets policy

- **No secrets in the repository**, in any file or commit: API keys, tokens, passwords, DSNs with
  passwords, private keys, cookies. `config/app_config.json` keeps `api_key` empty; keys come from the
  environment (`DEEPSEEK_API_KEY`, `TUSHARE_TOKEN`, ...), a local `.env` (git-ignored, see
  `.env.example`), or a Kubernetes Secret created outside the repository
  ([docs/deployment.md](docs/deployment.md#secrets)).
- **Enforced twice**:
  - pre-commit: `pre-commit install` enables a gitleaks hook on the staged diff
    ([`.pre-commit-config.yaml`](.pre-commit-config.yaml));
  - CI: the `secrets` job scans the **full git history** with gitleaks on every push and pull request
    ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
- **Allowlist entries must be narrow.** [`.gitleaks.toml`](.gitleaks.toml) extends the default rules
  with one exception for the historical finding below (matched by commit, file and rule together) and
  two content-matched false positives (a public model id and the hashed caller-id format). Never allowlist by path wildcard; a real secret is revoked
  and removed, not allowlisted.
- **If a secret is committed**: treat it as compromised as soon as it is pushed. Revoke or rotate it at
  the provider first, then remove it from the code, then decide whether to rewrite history (removal
  from history does not un-leak a key that was public).
- Logs, traces and `/ready` errors must not contain credentials; the API reports the SHA-256 prefix of
  an API key as the caller identity, never the key.

## Known historical exposure

An LLM API key (a 32-character `sk-...` value) was committed to `config/app_config.json` in commit
`302077a` (2026-05-02) and removed from the file in the next commit to that file, `d1d2958`, seven
minutes later. Both commits are part of the published history of `master`.

- The maintainers decided **not to rewrite history**: the commits were public, so rewriting cannot make
  the key secret again, and it would break every existing clone and reference.
- **The key must be considered compromised and must be revoked by its owner at the provider.** It is not
  used by any current code or configuration, and this repository does not reproduce its value anywhere.
- The CI and pre-commit scans allowlist this single finding (commit `302077a`, file
  `config/app_config.json`, rule `generic-api-key`) so that every other secret still fails the build.

## Runtime security (what the deployment enforces, and its limits)

**Authentication and tenancy.**
- `QI_API_KEYS` enables key authentication (`X-API-Key` or `Authorization: Bearer`).
- The Kubernetes manifest sets `QI_PROFILE=production` and reads the keys from a required Secret. In that profile the app refuses to start without keys unless `QI_ALLOW_ANONYMOUS=1` is set explicitly ([docs/deployment.md](docs/deployment.md#authentication)).
- Sessions, traces and A2A tasks are scoped to the caller: the SHA-256 prefix of the key, or for keyless callers a per-browser id from an HMAC-signed HttpOnly cookie.
- Keyless callers never see `/agent/traces` (403).

**Web UI API key.**
- The key is kept in `sessionStorage`: it is gone when the tab closes and is not shared with other tabs. It moves to `localStorage` only if the user ticks "Remember on this device"; that option is off by default and the dialog states the trade-off.
- Keys stored in `localStorage` by older builds are moved to `sessionStorage` on first load.
- Trade-off: both storages are readable by any script on the page's origin, so an XSS bug would expose the key either way. `sessionStorage` only limits how long the key stays on disk and in which tabs it is visible.
- A login endpoint that exchanges the key for an HttpOnly session cookie (with CSRF protection) would hide it from scripts. It is not implemented.

**Rate limiting.**
- `QI_RATE_LIMIT_PER_MINUTE` applies per validated key, or per remote address.
- When a Postgres DSN is configured, the bucket is shared by all replicas. If the database fails, it degrades to a per-replica bucket ([results](docs/results/security/shared-rate-limiter.md)).

**Prompt injection in retrieved documents.** Layers, from structural to heuristic:
1. Read-only tools.
2. An untrusted-data envelope around tool output.
3. Claim-level number and citation verification.
4. The deterministic (no-LLM) answer never quotes document titles or text. It cites documents by category, publisher and date. The evidence list hides titles that fail a positive shape check: NFKC and confusable folding; no links, domains, phones or messaging handles; no instructions and no advice or guarantee wording.
5. A compliance guard removes guaranteed-return claims (稳赚不赔, 保本, 保证收益, 资金翻倍+亏损赔付), stock-tip solicitation (荐股, 带单, 加微信, 私信), hype (直接拉升, 错过再等) and contact details from every answer. Before it, an output-side layer (`query_intelligence/agent/output_safety.py`) replaces any such sentence that came from a document with a neutral note, attributes single-source regulatory claims and disputed figures ("据一篇文档称…（未经其他来源证实）"), and drops document figures that contradict structured fundamentals.
6. A lexical filter, plus a small character n-gram classifier on document text.

The classifier catches about 4 in 10 held-out attacks and flags 0.5% of clean documents ([results](evaluation/results/injection_classifier-r4.json)). It does not generalise to attacks that are hype or planted facts rather than instructions. The filters are defense in depth; the structural layers carry the guarantee.

**Audit.**
- Refusals, compliance edits and injection-filter redactions are written to the audit log and counted in `finsight_audit_events_total` and `finsight_injection_redactions_total`. This includes a redacted user message whose turn was still answered.
- The audit log stores principal and query hashes, never text.

**Not covered.**
- The LLM-path red team (holdout3 and holdout4) has not been rerun at the current commit: it needs LLM quota.
- Key revocation for the historical exposure above must be confirmed by the owner at the provider.
