# Security Policy

## Supported Version

Until versioned releases are established, security maintenance targets the current `main` branch. Historical branches and older commits are not separately supported.

## Reporting Security Issues

Do not post sensitive security details, credentials, tokens, personal data, or private reproduction material in a public issue or pull request.

Use GitHub's private vulnerability reporting for this repository when that option is available under the **Security** tab. Include the affected version or commit, the observed impact, and enough information for the maintainer to reproduce and assess the report.

If private vulnerability reporting is unavailable, use a non-sensitive contact method published by the repository owner to request a private reporting channel. A public issue may request private contact, but should not include sensitive technical details.

There is currently no formal response-time SLA. Reports are prioritized according to impact, reproducibility, and exposure.

## Project Security Expectations

- Never commit real `SECRET_KEY`, `TMDB_API_KEY`, database credentials, tokens, or other secrets. Use `.env.example` only for safe placeholders and configuration names.
- Treat local model and cache files as administrator-controlled artifacts. Do not accept serialized model/cache data from untrusted users or remote inputs.
- Keep authentication, authorization, CSRF, persistence, external-request, dependency, and logging changes covered by tests and CI checks.
- Resolve dependency and static-analysis findings at the narrowest appropriate boundary rather than disabling checks globally.
- If a credential is accidentally exposed, rotate or revoke it; removing it only from a later commit does not remove it from repository history.

## Automated Checks

Pull requests run application tests, static correctness checks, dependency auditing, Bandit security scanning, and an explicit guard for the repository's trusted local model/cache deserialization sites.
