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
  with exactly two exceptions: the historical finding below (matched by commit, file and rule together)
  and a false positive on a public model id. Never allowlist by path wildcard; a real secret is revoked
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
