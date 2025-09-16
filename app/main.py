from fastapi import FastAPI, Header, Response, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import datetime as dt
import time, uuid, json, csv, os, asyncio

APP_TITLE = "LLM Serving PoC"
BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"
METRICS_DIR = BASE_DIR / "metrics"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# 공통 CSV 스키마
METRICS_CSV = METRICS_DIR / "requests.csv"
METRICS_HEADER_V3 = [
    "timestamp","route","request_id","status","is_stream",
    "ttft_ms","latency_ms","chunks","bytes","reason",
    "prompt_chars","output_chars","timeout_ms",
    "decode_ms","tpot_chunks_per_sec","delay_ms","token_size"
]
METRICS_HEADER = METRICS_HEADER_V3  # 현재 활성 스키마

# 헤더 자동 마이그레이션(레거시 파일은 보존)
def _ensure_metrics_header():
    if not METRICS_CSV.exists():
        with METRICS_CSV.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(METRICS_HEADER)
        return

    try:
        with METRICS_CSV.open("r", encoding="utf-8", newline="") as f:
            first = f.readline().strip()
    except Exception:
        first = ""

    expected = ",".join(METRICS_HEADER)
    if not first or first.replace(" ", "") != expected.replace(" ", ""):
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        legacy = METRICS_DIR / f"requests_legacy_{ts}.csv"
        try:
            os.replace(METRICS_CSV, legacy)
        except Exception as e:
            print(f"[metrics-rotate-warn] {e}")
        with METRICS_CSV.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(METRICS_HEADER)

_ensure_metrics_header()

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

# 공통 CSV 기록. extras는 미지정 시 기본값 채움
def _append_metrics(route: str, request_id: str, ttft_ms: int, latency_ms: int,
                    status: int, **extras) -> None:
    row = [
        _now_iso(),
        route,
        request_id,
        status,
        extras.get("is_stream", False),
        ttft_ms,
        latency_ms,
        extras.get("chunks", 0),
        extras.get("bytes", 0),
        extras.get("reason", "ok"),
        extras.get("prompt_chars", None),
        extras.get("output_chars", None),
        extras.get("timeout_ms", None),
        extras.get("decode_ms", None),
        extras.get("tpot_chunks_per_sec", None),
        extras.get("delay_ms", None),
        extras.get("token_size", None),
    ]
    try:
        with METRICS_CSV.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        print(f"[csv-write-error] {e}")

def _coalesce_request_id(x_request_id: Optional[str]) -> str:
    if x_request_id:
        rid = x_request_id.strip()
        if 1 <= len(rid) <= 120:
            return rid
    return f"req-{uuid.uuid4().hex[:12]}"

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 32

class GenerateResponse(BaseModel):
    id: str
    output: str

# 컨테이너/프로세스가 죽었는지, 메인 루프가 응답하는지만 확인
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
    record = {
        "ts": _now_iso(), "route": "/health", "request_id": rid,
        "status": status, "is_stream": False,
        "ttft_ms": ttft_ms, "latency_ms": latency_ms,
        "chunks": 0, "bytes": 0, "reason": "ok"
    }
    _append_jsonl(record)                         # logs/requests-YYYYMMDD.jsonl로 한 줄 append
    _append_metrics("/health", rid, ttft_ms, latency_ms, status,
                    is_stream=False, chunks=0, bytes=0, reason="ok",
                    decode_ms=None, tpot_chunks_per_sec=None,
                    delay_ms=None, token_size=None)  # metrics/requests.csv로 누적

    # 6) 응답 반환(직접 직렬화)
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        status_code=status,
        media_type="application/json",
        headers={"X-Request-ID": rid},
    )

# 텍스트 생성 API
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

    ttft_ms    = int((t_first - t0) * 1000)
    latency_ms = int((t_end   - t0) * 1000)
    status = 200

    record = {
        "ts": _now_iso(), "route": "/v1/generate", "request_id": rid,
        "status": status, "is_stream": False,
        "ttft_ms": ttft_ms, "latency_ms": latency_ms,
        "chunks": 0, "bytes": 0, "reason": "ok",
        "prompt_chars": len(body.prompt) if getattr(body, "prompt", None) else None,
        "output_chars": len(output)
    }
    _append_jsonl(record)
    _append_metrics("/v1/generate", rid, ttft_ms, latency_ms, status,
                    is_stream=False, chunks=0, bytes=0, reason="ok",
                    prompt_chars=len(body.prompt) if getattr(body, "prompt", None) else None,
                    output_chars=len(output),
                    decode_ms=None, tpot_chunks_per_sec=None,
                    delay_ms=None, token_size=None)

    return Response(
        content=json.dumps(resp_obj, ensure_ascii=False),
        status_code=status,
        media_type="application/json",
        headers={"X-Request-ID": rid},
    )

# SSE(Server-Sent Events) 방식으로 prompt를 8자 단위로 쪼개 증분 전송
@app.get("/v1/stream")
async def stream(
    request: Request,
    prompt: str = Query(..., description="text prompt"),
    max_tokens: int = Query(32, ge=1, le=4096),
    delay_ms: int = Query(40, ge=0, le=2000),
    timeout_ms: int = Query(60_000, ge=1_000, le=600_000),
    x_request_id: Optional[str] = Header(None),
):
    t0   = time.perf_counter()
    rid  = _coalesce_request_id(x_request_id)
    route = "/v1/stream"

    # placeholder 토큰화: prompt 일부를 고정 길이 조각으로 스트리밍
    token_size = 8
    text = prompt[:max_tokens]
    chunks_total = (len(text) + token_size - 1) // token_size

    state = {
        "first_sent": False,
        "t_first": None,
        "chunks": 0,
        "bytes": 0,
        "reason": "ok",
        "status": 200,
        "output_acc": []
    }

    async def event_gen():
        try:
            # 시작 알림(옵션)
            yield f"event: start\ndata: {json.dumps({'id': rid, 'route': route})}\n\n"

            start = time.perf_counter()
            for idx in range(chunks_total):
                # 클라이언트 취소 감지
                if await request.is_disconnected():
                    state["reason"] = "client_cancel"; state["status"] = 499
                    break
                # 타임아웃
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                if elapsed_ms > timeout_ms:
                    state["reason"] = "timeout"; state["status"] = 504
                    break

                piece = text[idx*token_size:(idx+1)*token_size]
                payload = {"id": rid, "index": idx, "delta": piece, "finished": False}
                line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                if not state["first_sent"]:
                    state["first_sent"] = True
                    state["t_first"] = time.perf_counter()

                state["chunks"] += 1
                state["bytes"]  += len(line.encode("utf-8"))
                state["output_acc"].append(piece)
                yield line

                # 마지막 청크 시각 기록(정확한 decode_ms/TPOT 계산용)
                if idx == chunks_total - 1:
                    state["t_last"] = time.perf_counter()

                # 마지막 청크 뒤에는 sleep 생략 → (chunks−1)번만 지연
                if delay_ms and idx < chunks_total - 1:
                    await asyncio.sleep(delay_ms / 1000)

            # 종료 이벤트
            done = {"id": rid, "finished": True, "reason": state["reason"]}
            yield f"event: done\ndata: {json.dumps(done, ensure_ascii=False)}\n\n"

        except asyncio.CancelledError:
            state["reason"] = "client_cancel"; state["status"] = 499
            raise
        except Exception as e:
            state["reason"] = "error"; state["status"] = 500
            try:
                err = {"id": rid, "finished": True, "reason": "error", "message": str(e)}
                yield f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"
            finally:
                raise
        finally:
            # 공통 스키마 로깅
            t_end = time.perf_counter()
            ttft_ms    = int(((state["t_first"] or t_end) - t0) * 1000)
            latency_ms = int((t_end - t0) * 1000)
            
            # 첫 청크 → 마지막 청크 구간(디코딩 시간)
            t_last = state.get("t_last", None)
            decode_ms = int(((t_last or t_end) - (state["t_first"] or t_end)) * 1000) if state["first_sent"] else 0

            # TPOT: 청크/초 (간격 개수 = chunks−1)
            tpot = (state["chunks"] - 1) / (decode_ms / 1000) if state["chunks"] > 1 and decode_ms > 0 else None

            rec = {
                "ts": _now_iso(), "route": route, "request_id": rid,
                "status": state["status"], "is_stream": True,
                "ttft_ms": ttft_ms, "latency_ms": latency_ms,
                "chunks": state["chunks"], "bytes": state["bytes"],
                "reason": state["reason"],
                "prompt_chars": len(prompt), "output_chars": len("".join(state["output_acc"])),
                "timeout_ms": timeout_ms,

                # (선택) 분석 편의용 추가 컬럼
                "decode_ms": decode_ms,
                "tpot_chunks_per_sec": tpot,
                "delay_ms": delay_ms,
                "token_size": token_size,
            }
            _append_jsonl(rec)
            _append_metrics(route, rid, ttft_ms, latency_ms, state["status"],
                            is_stream=True, chunks=state["chunks"], bytes=state["bytes"],
                            reason=state["reason"], prompt_chars=len(prompt),
                            output_chars=len("".join(state["output_acc"])),
                            timeout_ms=timeout_ms,
                            decode_ms=decode_ms,
                            tpot_chunks_per_sec=tpot,
                            delay_ms=delay_ms,
                            token_size=token_size)

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-Request-ID": rid,
    }
    return StreamingResponse(event_gen(), headers=headers, status_code=200)
