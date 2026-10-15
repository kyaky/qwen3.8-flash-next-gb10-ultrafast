#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import json
import sys

trace = sys.argv[1]
d = json.load(open(trace))
seen = set()
for e in d["traceEvents"]:
    if e.get("cat") in ("kernel", "Kernel") and "Marlin" in (e.get("name") or ""):
        a = e.get("args", {})
        key = json.dumps([e["name"][:70], a.get("grid"), a.get("block"),
                          a.get("registers per thread"),
                          a.get("shared memory")], sort_keys=True)
        if key not in seen:
            seen.add(key)
            print(key)
