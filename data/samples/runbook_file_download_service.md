# Runbook: gateway_service - File Not Ready / Download Failed

## Summary
This runbook covers incidents where gateway_service fails to download a file because the file is not yet available or still processing in upstream storage.

## Symptoms
- Errors like `FileNotAvailable` or `FileDownloadFailed` in logs.
- Retries are scheduled but downloads continue to fail.
- No significant platform metric anomalies.

## Likely Root Cause
Upstream data pipeline has not materialized the requested file yet (or it is still processing).

## Triage Checklist
1. Identify the failed file name from logs (e.g., `file_name=notification_data.txt`).
2. Confirm the upstream pipeline state for that file (processing / complete / failed).
3. If the file is still processing, wait for completion before retrying.

## Recommended Actions
### 1) Verify file readiness
Confirm that the file is present in upstream storage and marked "ready":
- Storage location: `nfs://notifications/ready/`
- Expected object: `<file_name>`

### 2) Retry the download
Once the file is ready, re-trigger gateway_service with the correct payload:

```json
{
  "service": "gateway_service",
  "action": "retry_file_download",
  "payload": {
    "file_name": "notification_data.txt",
    "priority": "standard",
    "requested_by": "incident-response"
  }
}
```

### 3) Confirm success
Verify a `download_status=success` message in logs and that the file was delivered.

## Escalation
If the file remains unavailable after 2 retry windows, escalate to the upstream data pipeline owner.