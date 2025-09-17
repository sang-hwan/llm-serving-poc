import time, subprocess, json, uuid, argparse

def call_generate(prompt, max_tokens=12):
    rid = str(uuid.uuid4())
    out = subprocess.check_output([
        "curl.exe","-s",
        "-H", f"X-Request-ID: {rid}",
        "-H","Content-Type: application/json",
        "-d", json.dumps({"prompt": prompt, "max_tokens": max_tokens}),
        "http://127.0.0.1:8000/v1/generate"
    ]).decode("utf-8","ignore")
    return rid, out

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--attempts",type=int,default=3); args=ap.parse_args()
    backoff=0.25
    for i in range(args.attempts):
        rid, out = call_generate("hello structured world",12)
        try:
            ok = set(json.loads(out).keys()) >= {"id","output"}
        except Exception:
            ok = False
        print(f"[try={i}] rid={rid} ok={ok}")
        if ok: break
        time.sleep(backoff); backoff*=2
    print("[demo] fallback rule = keep only core fields")
