"""Keep-warm scheduler — pings both containers every 5 minutes with warmup=True.

Deploy manually with:
    modal deploy modal_backend/keepwarm.py

Stop manually with:
    modal app stop f1-paddock-oracle-keepwarm

Do NOT include this in the main app deploy.
"""

import modal

from modal_backend.commentary import generate_commentary
from modal_backend.strategy import reason_strategy

app = modal.App("f1-paddock-oracle-keepwarm")


@app.function(schedule=modal.Period(minutes=5))
def ping_containers():
    generate_commentary.remote(prompt="", warmup=True)
    reason_strategy.remote(prompt="", warmup=True)
