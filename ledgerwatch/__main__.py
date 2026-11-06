"""Command line:

    python -m ledgerwatch build          rebuild the simulated ledger (wipes it)
    python -m ledgerwatch ensure         build the ledger and train model v1 only if missing
    python -m ledgerwatch eval           train v1 on months 1-9, measure on 10-12, check the gates
    python -m ledgerwatch close [YYYY-MM | previous]
                                         score a close and let the agent investigate the top flags
                                         (the monthly scheduled job runs "close previous")
    python -m ledgerwatch agent-eval [--live] [--limit N]
                                         run the investigating agent on the current close's flags
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta

from .config import CURRENT_CLOSE, Settings


def main() -> int:
    args = sys.argv[1:]
    cmd = args[0] if args else "help"
    s = Settings()
    if cmd == "build":
        from .simulate import build

        labels = build(s.data_dir)
        print(f"Built the ledger in {s.data_dir} with {len(labels)} planted problem entries")
        return 0
    if cmd == "ensure":
        from .service import ensure_ready

        ensure_ready(s)
        print(f"Ledger and model v1 ready in {s.data_dir}")
        return 0
    if cmd == "eval":
        from .evals import run
        from .simulate import ensure_built
        from .tracking import log_run

        ensure_built(s.data_dir)
        _, m = run(s.data_dir, s.models)
        log_run(m, s.models)
        keep = ("version", "entries", "problems", "found", "recall", "high_severity_recall", "precision", "flags_per_1000",
                "per_kind", "baselines", "gates", "passed")
        print(json.dumps({k: m[k] for k in keep}, indent=1))
        with open("eval_report.json", "w") as fh:
            json.dump(m, fh, indent=1, default=str)
        return 0 if m["passed"] else 1
    if cmd == "close":
        from .service import open_service

        month = args[1] if len(args) > 1 else CURRENT_CLOSE
        if month == "previous":  # the scheduled job runs early in the month for the month that just closed
            month = (date.today().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
        svc = open_service(s.data_dir)
        r = svc.score_close(month)
        live = s.llm_provider == "bedrock" or bool(os.environ.get("ANTHROPIC_API_KEY"))
        mode = ("bedrock" if s.llm_provider == "bedrock" else "live") if live else "scripted"
        kw = {"api_key": os.environ.get("ANTHROPIC_API_KEY")} if mode == "live" else {}
        n = svc.investigate_top(month, mode=mode, **kw)
        svc.hub.close()
        print(f"{month}: {r['entries']} entries, {r['flags']} flags, {n} investigated ({mode}), model {r['version']}")
        return 0
    if cmd == "agent-eval":
        from . import agent_evals
        from .service import open_service

        live = "--live" in args
        limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
        svc = open_service(s.data_dir)
        kw = {"api_key": os.environ.get("ANTHROPIC_API_KEY")} if live else {}
        out = agent_evals.run(svc, CURRENT_CLOSE, "live" if live else "scripted", limit, **kw)
        svc.hub.close()
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
        with open(f"eval_report_agent_{'live' if live else 'scripted'}.json", "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        return 0 if out["passed"] else 1
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
