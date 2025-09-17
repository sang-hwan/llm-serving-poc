import argparse, json, csv
from pathlib import Path
from datetime import datetime
import fastjsonschema

CSV_HEADER = ["timestamp","source","kind","file","request_id","schema_ok","error_path","reason","retriable"]

def compile_schema(p: Path):
    return fastjsonschema.compile(json.loads(p.read_text(encoding="utf-8")))

def iter_inputs(in_file: Path=None, in_dir: Path=None, ndjson=False):
    if in_file:
        text = Path(in_file).read_text(encoding="utf-8")
        if ndjson:
            for i, line in enumerate(text.splitlines()):
                line=line.strip()
                if line: yield Path(in_file), i+1, line
        else:
            yield Path(in_file), None, text
    else:
        for fp in sorted(Path(in_dir).rglob("*.json")):
            yield fp, None, fp.read_text(encoding="utf-8")

def classify_reason(msg:str):
    if any(s in msg for s in ["required property","is not of type","under any of the given schemas"]):
        return "invalid_schema", True
    if "Expecting value" in msg:
        return "non_json", True
    return "invalid_schema", True

def find_request_id(obj):
    for k in ("request_id","id","req_id"):
        v = obj.get(k)
        if isinstance(v,str): return v
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--kind", required=True, choices=["completion","chunk","error"])
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in_file")
    g.add_argument("--in_dir")
    ap.add_argument("--ndjson", action="store_true")
    ap.add_argument("--csv", default="metrics/structured_output.csv")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    validate = compile_schema(Path(args.schema))
    out = Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists() or not args.append
    mode = "a" if args.append and out.exists() else "w"

    with out.open(mode, "w" if not out.exists() else mode, encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        if write_header: wr.writerow(CSV_HEADER)

        src = Path(args.in_file).name if args.in_file else Path(args.in_dir).as_posix()
        for p, lineno, text in iter_inputs(args.in_file, args.in_dir, args.ndjson):
            ts = datetime.now().isoformat(timespec="seconds")
            file_disp = p.name if lineno is None else f"{p.name}:{lineno}"
            try:
                obj = json.loads(text)
            except Exception as e:
                reason, retriable = classify_reason(str(e))
                wr.writerow([ts, src, args.kind, file_disp, "", False, "", reason, retriable])
                continue
            try:
                validate(obj)
                wr.writerow([ts, src, args.kind, file_disp, find_request_id(obj), True, "", "ok", False])
            except Exception as e:
                msg = str(e)
                path_hint = ""
                for token in (" at ", " data."):
                    if token in msg:
                        path_hint = msg.split(token,1)[-1][:80]; break
                reason, retriable = classify_reason(msg)
                wr.writerow([ts, src, args.kind, file_disp, find_request_id(obj), False, path_hint, reason, retriable])

if __name__ == "__main__":
    main()
