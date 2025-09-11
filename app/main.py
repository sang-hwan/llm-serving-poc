from fastapi import FastAPI, Header, Response
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import datetime as dt
import time, uuid, json, csv

APP_TITLE = "LLM Serving PoC"
BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"
METRICS_DIR = BASE_DIR / "metrics"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = METRICS_DIR / "requests.csv"
METRICS_HEADER = ["timestamp","route","request_id","ttft_ms","latency_ms","status"]

# CSV 헤더 보장
if not METRICS_CSV.exists():
    with METRICS_CSV.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(METRICS_HEADER)

SERVER_STARTED_AT = dt.datetime.now(dt.timezone.utc)
app = FastAPI(title=APP_TITLE)

def _today_jsonl() -> Path:
    today = dt.datetime.now().strftime("%Y%m%d")
    return LOGS_DIR / f"requests-{today}.jsonl"

def _now_iso() -> str:
    # 로컬 타임존 포함 ISO8601 (밀리초)
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")

def _append_jsonl(record: dict) -> None:
    try:
        with _today_jsonl().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[jsonl-write-error] {e}")

def _append_metrics(route: str, request_id: str, ttft_ms: int, latency_ms: int, status: int) -> None:
    try:
        with METRICS_CSV.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([_now_iso(), route, request_id, ttft_ms, latency_ms, status])
    except Exception as e:
        print(f"[csv-write-error] {e}")

def _coalesce_request_id(x_request_id: Optional[str]) -> str:
    if x_request_id:
        rid = x_request_id.strip()
        if 1 <= len(rid) <= 120:
            return rid
    return str(uuid.uuid4())

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 32

class GenerateResponse(BaseModel):
    id: str
    output: str

@app.get("/health")
def health(x_request_id: Optional[str] = Header(None)):
    t0 = time.perf_counter()                     # 1) 성능 측정 시작(고해상도, 단조 증가 타이머)
    rid = _coalesce_request_id(x_request_id)     # 2) 요청 ID: 헤더 있으면 그대로, 없으면 UUID

    # 3) 상태 페이로드 구성
    uptime_s = (dt.datetime.now(dt.timezone.utc) - SERVER_STARTED_AT).total_seconds()
    payload = {"status": "ok", "uptime_s": round(uptime_s, 3), "version": "0.1.0"}

    # 4) TTFT/총 지연 측정(비스트리밍이라 사실상 거의 동일)
    t_first = time.perf_counter()
    t_end = time.perf_counter()

    ttft_ms   = int((t_first - t0) * 1000)       # 요청 수신→첫 바이트 준비까지(추정)
    latency_ms= int((t_end   - t0) * 1000)       # 요청 수신→응답 준비 완료까지
    status = 200

    # 5) 라인로그(JSONL) + 지표(CSV) 기록
    record = {"ts": _now_iso(), "route": "/health", "request_id": rid,
              "ttft_ms": ttft_ms, "latency_ms": latency_ms, "status": status}
    _append_jsonl(record)                         # logs/requests-YYYYMMDD.jsonl로 한 줄 append
    _append_metrics("/health", rid, ttft_ms, latency_ms, status)  # metrics/requests.csv로 누적

    # 6) 응답 반환(직접 직렬화)
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        status_code=status,
        media_type="application/json",
        headers={"X-Request-ID": rid},
    )

@app.post("/v1/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest, x_request_id: Optional[str] = Header(None)):
    t0 = time.perf_counter()
    rid = _coalesce_request_id(x_request_id)

    # 최소 placeholder "생성": prompt 일부를 잘라서 반환
    max_tokens = body.max_tokens or 32
    output = body.prompt[:max_tokens]

    t_first = time.perf_counter()  # 첫 바이트 준비 시점(비스트리밍이라 거의 즉시)
    resp_obj = {"id": rid, "output": output}
    t_end = time.perf_counter()

    ttft_ms = int((t_first - t0) * 1000)
    latency_ms = int((t_end - t0) * 1000)
    status = 200

    record = {"ts": _now_iso(), "route": "/v1/generate", "request_id": rid,
              "ttft_ms": ttft_ms, "latency_ms": latency_ms, "status": status}
    _append_jsonl(record)
    _append_metrics("/v1/generate", rid, ttft_ms, latency_ms, status)

    return Response(
        content=json.dumps(resp_obj, ensure_ascii=False),
        status_code=status,
        media_type="application/json",
        headers={"X-Request-ID": rid},
    )
