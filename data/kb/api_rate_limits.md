# Internal API — Rate Limits

**Last updated:** 2025-11-15

Rate limits are applied per workspace.

## 1. Default limits

- `GET` endpoints: 120 requests/min
- `POST` endpoints: 60 requests/min

## 2. Burst behavior

- short bursts up to 2x are allowed
- sustained bursts will be throttled

## 3. Headers

Responses include:

- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`

## 4. 429 responses

If you receive `429 Too Many Requests`:

- apply exponential backoff (start at 250ms)
- respect `Retry-After` when present

