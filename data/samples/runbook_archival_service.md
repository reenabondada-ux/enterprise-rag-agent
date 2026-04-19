# Runbook: Document Archival Timeout (document-archival-service)

## Incident Pattern
Document archival requests time out when document-archival-service service calls griffin-archiver during peak load.

## Typical Signals
- Logs show `ReadTimeout`, `ConnectTimeout`, and `504 Gateway Timeout`.
- Retry attempts increase with exponential backoff.
- Queue depth and request latency rise during load spikes.

## Initial Triage
1. Confirm affected service: `document-archival-service`.
2. Confirm error class: timeout (not auth/validation).
3. Check current downstream health and queue backlog.

## Recommended Response
1. Retry archival request with exponential backoff.
2. Temporarily increase downstream timeout for archival path.
3. Restart document-archival-service only if retries continue to fail.
4. Validate health + latency trend after each retry loop.

## Example Retry Payload
```json
{
  "service": "document-archival-service",
  "action": "retry_downstream_request",
  "operation": "archive_document",
  "retry_strategy": "exponential_backoff",
  "max_attempts": 3
}
```

## Validation Criteria
- Service health check returns healthy.
- 504/timeout errors stop for new archival requests.
- Queue depth and latency begin returning toward baseline.

## Escalation
Escalate to platform/on-call if retries fail for all attempts or timeout rate remains elevated for more than 15 minutes.