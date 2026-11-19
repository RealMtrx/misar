# Security Policy

## Supported Versions

| Version | Supported |
| :--- | :---: |
| `1.0.x` | ✅ |

## Built-in Security Controls

When running `misar serve` or embedding the workbench UI:

- **Payload Bounds:** `/api/route` and `/api/batch` reject payloads exceeding `1 MiB` (`413 Payload Too Large`) and batches exceeding `500` items (`400 Bad Request`).
- **Strict Type & Header Validation:** Malformed UTF-8, invalid JSON, non-object JSON payloads, non-string `text` entries, and invalid `Content-Length` headers return clean `4xx` responses.
- **HTTP Hardening Headers:** All responses include `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, and `Referrer-Policy: no-referrer`.
- **XSS-Safe UI:** The interactive web workbench renders user text and token outputs strictly via DOM `.textContent` nodes with a 1,500-chip DOM render cap.
- **Local Binding by Default:** `misar serve` binds to `127.0.0.1` unless explicitly overridden via `--host`.

## Reporting a Vulnerability

Please open a private security advisory on [GitHub Security Advisories](https://github.com/RealMtrx/misar/security/advisories) or open a minimal issue without publishing exploit payloads publicly until patched.
