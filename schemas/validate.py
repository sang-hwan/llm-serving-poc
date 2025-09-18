import argparse, json, csv, sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, List
from datetime import datetime

try:
    import fastjsonschema
except Exception as e:
    print("fastjsonschema 가 필요합니다: pip install fastjsonschema", file=sys.stderr)
    raise

CSV_HEADER = [
    "timestamp","kind","path","is_valid","reason",
    "request_id","schema_version","fields_missing",
    "fields_extra","type_errors","size_bytes"
]

def load_schema(schema_path: Path):
    raw = json.loads(schema_path.read_text(encoding="utf-8"))
    compile_fn = fastjsonschema.compile(raw)
    # 스키마 버전 표기 우선순위: version > $id > title > 파일명
    version = raw.get("version") or raw.get("$id") or raw.get("title") or schema_path.stem
    return compile_fn, str(version)

def is_ndjson(p: Path) -> bool:
    return p.suffix.lower() in {".jsonl", ".ndjson"}

def iter_json_objects(p: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    if p.is_dir():
        for q in sorted(p.rglob("*.json")):
            yield from iter_json_objects(q)
        for q in sorted(p.rglob("*.jsonl")):
            yield from iter_json_objects(q)
        for q in sorted(p.rglob("*.ndjson")):
            yield from iter_json_objects(q)
        return
    if is_ndjson(p):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                yield p.with_suffix(p.suffix + f":{i}"), obj
            except Exception:
                yield p.with_suffix(p.suffix + f":{i}"), {"__non_json__": line}
        return
    # 일반 JSON
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                yield p.with_suffix(p.suffix + f":{i}"), item
        else:
            yield p, obj
    except Exception:
        yield p, {"__non_json__": p.read_text(encoding="utf-8", errors="ignore")}

def detect_request_id(d: Dict[str, Any]) -> str:
    # 최상위 → 흔한 중첩(meta/request/header) → 실패 시 빈 문자열
    for k in ("request_id", "id", "rid"):
        if isinstance(d.get(k), str) and d[k]:
            return d[k]
    meta = d.get("meta") or d.get("header") or {}
    if isinstance(meta, dict):
        for k in ("request_id", "id", "rid"):
            v = meta.get(k)
            if isinstance(v, str) and v:
                return v
    return ""

def classify_reason(errs: List[str], obj: Dict[str, Any]) -> str:
    if "__non_json__" in obj:
        return "non_json"
    if not errs:
        return "ok"
    # 간단 분류 규칙
    for e in errs:
        if "is a required property" in e or "required" in e:
            return "missing_field"
        if "is not of type" in e or "is not of" in e or "is not a" in e:
            return "type_mismatch"
        if "Additional properties are not allowed" in e or "additionalProperties" in e:
            return "extra_field"
    return "invalid_schema"

def validate_obj(validator, obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    try:
        validator(obj)
        return True, []
    except Exception as e:
        # fastjsonschema는 ValidationError.messages 가 없을 수 있어 문자열만 기록
        return False, [str(e)]

def open_csv(out: Path, append: bool) -> Tuple[csv.writer, Any]:
    write_header = (not out.exists()) or (not append)
    mode = "a" if append and out.exists() else "w"
    f = out.open(mode, encoding="utf-8", newline="")
    wr = csv.writer(f)
    if write_header:
        wr.writerow(CSV_HEADER)
    return wr, f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True, help="JSON Schema 파일 경로")
    ap.add_argument("--kind", required=True, choices=["completion","chunk","error"])
    ap.add_argument("--in_file", help="검증 대상 파일(.json/.jsonl/.ndjson)")
    ap.add_argument("--in_dir",  help="검증 대상 디렉터리(재귀)")
    ap.add_argument("--out_csv", default="metrics/structured_output.csv")
    ap.add_argument("--append", action="store_true", help="CSV 이어쓰기")
    args = ap.parse_args()

    targets = []
    if args.in_file:
        targets.append(Path(args.in_file))
    if args.in_dir:
        targets.append(Path(args.in_dir))
    if not targets:
        print("검증 대상이 없습니다: --in_file 또는 --in_dir 지정", file=sys.stderr)
        sys.exit(2)

    schema_path = Path(args.schema)
    validator, schema_version = load_schema(schema_path)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wr, f = open_csv(out_path, args.append)

    total = ok = 0
    try:
        for t in targets:
            for path, obj in iter_json_objects(t):
                total += 1

                # 크기(바이트)
                size = (
                    len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
                    if "__non_json__" not in obj
                    else len(str(obj["__non_json__"]).encode("utf-8"))
                )

                # 요청 ID
                rid = "" if "__non_json__" in obj else detect_request_id(obj)

                # 스키마 검증
                if "__non_json__" in obj:
                    valid, errs = False, ["non_json"]
                else:
                    valid, errs = validate_obj(validator, obj)

                # 분류 및 CSV용 필드들(가독성 분리)
                reason = classify_reason(errs, obj)
                fields_missing = "" if valid else ";".join([e for e in errs if "required" in e])
                fields_extra   = "" if valid else ("extra" if reason == "extra_field" else "")
                type_errors    = "" if valid else ";".join([e for e in errs if "is not of" in e or "is not a" in e])

                if valid:
                    ok += 1

                wr.writerow([
                    datetime.utcnow().isoformat(),
                    args.kind,
                    str(path),
                    1 if valid else 0,
                    reason,
                    rid,
                    schema_version,
                    fields_missing,
                    fields_extra,
                    type_errors,
                    size
                ])
    finally:
        f.close()

    print(f"[validate] kind={args.kind} schema={schema_version} total={total} ok={ok} nok={total-ok}")

if __name__ == "__main__":
    main()
