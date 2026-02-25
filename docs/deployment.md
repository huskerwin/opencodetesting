# Deployment Notes

This project is currently optimized for local and internal use. For production
deployment, plan for authentication, persistence, and observability.

## Recommended baseline

- Run behind an authenticated gateway (SSO, OAuth, or internal identity)
- Store document artifacts/indexes in persistent storage
- Centralize logs and monitor errors/latency
- Restrict outbound network access to approved endpoints

## Containerization

When containerizing, include:

- Python runtime matching repository requirements
- Tesseract binary and language data if OCR is required
- Environment variable injection for OpenAI and OCR settings

## Security and data handling

- Do not commit secrets; use environment variables or secret manager
- Apply least-privilege access to document storage
- Evaluate retention and deletion policies for uploaded content

## Scaling considerations

- OCR can be CPU-intensive and should be queued for high-volume workloads
- In-memory indexing is session-scoped; move to persistent/vector storage for
  larger multi-user deployments
