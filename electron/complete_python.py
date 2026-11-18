import json
import sys

try:
    import jedi
except ImportError:
    sys.path.append(sys.argv[1])
    import jedi

for raw in sys.stdin:
    request = {}
    try:
        request = json.loads(raw)
        code = request["code"]
        script = jedi.Script(code, path=request.get("path") or None)
        values = script.complete(request["line"], request["column"])
        matches = [{"label": item.name, "type": item.type, "detail": item.description}
                   for item in values[:70]]
        response = {"id": request["id"], "matches": matches}
    except Exception:
        response = {"id": request.get("id"), "matches": []}
    print(json.dumps(response), flush=True)
