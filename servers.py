import os
import hmac
import hashlib
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from tasks import analyze_pr_diff, analyze_push_diff


app = FastAPI(title="AI Code Reviewer Webhook Server")
GITHUB_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "super_secret_webhook_token")

def verify_signature(payload: bytes, signature: str):
    if not signature:
        raise HTTPException(status_code=401, detail="X-Hub-Signature-256 missing")
    hash_object = hmac.new(GITHUB_SECRET.encode(), msg=payload, digestmod=hashlib.sha256)
    expected_signature = "sha256=" + hash_object.hexdigest()
    if not hmac.compare_digest(expected_signature, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

@app.post("/webhook/github")
async def github_webhook_listener(request: Request, x_hub_signature_256: str = Header(None)):
    payload_bytes = await request.body()
    verify_signature(payload_bytes, x_hub_signature_256)

    payload = await request.json()
    event_type = request.headers.get("X-GitHub-Event", "unknown")

    if event_type == "pull_request":
        action = payload.get("action")
        if action in ["opened", "synchronize"]:
            analyze_pr_diff.delay(payload)
            return JSONResponse(
                status_code=202,
                content={"status": "Enqueued in Celery worker stream", "action": action}
            )
        
    elif event_type == "push":
        analyze_push_diff.delay(payload)
        return JSONResponse(status_code=202, content={"status": "Enqueued", "event": "push"})


    return {"status": "Ignored event stream", "event": event_type}