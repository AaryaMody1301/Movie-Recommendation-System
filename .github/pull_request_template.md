## Summary

Describe the problem and the change.

## Validation

- [ ] Tests added or updated where behavior changed.
- [ ] `pytest` passes locally, or CI is the validation source.
- [ ] Relevant quality/security checks pass.

## Impact

Check all that apply and explain below when relevant.

- [ ] Configuration / environment variables
- [ ] Database or persisted user data
- [ ] Movie catalog / DataLoader contract
- [ ] Recommendation behavior or model/cache format
- [ ] TMDb / external HTTP behavior
- [ ] Authentication / authorization / CSRF
- [ ] Deployment / health / logging
- [ ] Dependencies
- [ ] Documentation only

## Data and Security Checklist

- [ ] No credentials, `.env` values, local databases, model/cache artifacts, or logs are committed.
- [ ] Online personalization still uses persisted application-user ratings rather than synthetic/baseline CSV behavior unless this PR explicitly changes and documents that contract.
- [ ] New external requests are bounded and failure-tolerant.
- [ ] No new untrusted serialized-data deserialization path is introduced.

## Documentation

List any README, deployment, troubleshooting, configuration, or contributor documentation updated by this change. If none is needed, state why.
