"""
Edge Computing Architecture Simulator — compact version
Run: python edge_simulator.py
"""

import asyncio, random, logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("edge")

now = lambda: datetime.now(timezone.utc).isoformat()

# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class Reading:
    device: str
    kind:   str
    value:  Any
    unit:   str
    ts:     str = field(default_factory=now)

@dataclass
class Event:
    reading:  Reading
    node:     str
    alerts:   list = field(default_factory=list)
    to_cloud: bool = False

# ── IoT Devices ───────────────────────────────────────────────────────────────

SENSORS = {
    "temp":     (lambda: round(random.gauss(22, 6), 2),    "°C"),
    "humidity": (lambda: round(random.uniform(30, 90), 2), "%"),
    "motion":   (lambda: random.random() < 0.25,            "bool"),
    "power":    (lambda: round(random.uniform(0.5, 15), 3), "kWh"),
    "gps":      (lambda: {"lat": round(42.87 + random.uniform(-0.01, 0.01), 5),
                          "lon": round(74.57 + random.uniform(-0.01, 0.01), 5)}, "coords"),
}

def read(device_id: str, kind: str) -> Reading:
    gen, unit = SENSORS[kind]
    return Reading(device_id, kind, gen(), unit)

# ── Service Plugins ───────────────────────────────────────────────────────────

class Plugin:
    name = "base"
    async def on_data(self, e: Event): pass
    async def on_alert(self, alert: str, e: Event): pass

class MQTT(Plugin):
    name = "mqtt"
    async def on_data(self, e: Event):
        await asyncio.sleep(0.01)
        log.info(f"MQTT → edge/{e.node}/{e.reading.kind}  val={e.reading.value}")
    async def on_alert(self, alert, e):
        log.warning(f"MQTT ALERT edge/alerts/{e.node}: {alert}")

class InfluxDB(Plugin):
    name = "influxdb"
    async def on_data(self, e: Event):
        if isinstance(e.reading.value, (int, float)):
            await asyncio.sleep(0.005)
            log.info(f"InfluxDB ← {e.reading.kind}={e.reading.value}{e.reading.unit}")

class Webhook(Plugin):
    name = "webhook"
    def __init__(self, url="https://api.example.com/iot"):
        self.url = url
    async def on_alert(self, alert, e):
        await asyncio.sleep(0.02)
        log.warning(f"WEBHOOK POST {self.url} | {alert} | device={e.reading.device}")

class Slack(Plugin):
    name = "slack"
    def __init__(self, channel="#alerts"):
        self.channel = channel
    async def on_alert(self, alert, e):
        log.warning(f"SLACK {self.channel} alert: {alert} | device={e.reading.device}")

# ── Edge Node ─────────────────────────────────────────────────────────────────

ALERT_RULES = {
    "temp":     lambda v: v > 40 or v < -10,
    "humidity": lambda v: v > 85,
    "motion":   lambda v: v is True,
    "power":    lambda v: v > 12,
}

class EdgeNode:
    def __init__(self, node_id, cloud_every=5):
        self.id      = node_id
        self.plugins = []
        self.count   = 0
        self.alerts  = 0
        self._nth    = cloud_every

    def use(self, *plugins):
        self.plugins.extend(plugins)
        return self

    async def process(self, r: Reading):
        rule   = ALERT_RULES.get(r.kind)
        alerts = [f"{r.kind}={r.value}{r.unit} exceeded threshold"] if rule and rule(r.value) else []
        e      = Event(r, self.id, alerts, to_cloud=(self.count % self._nth == 0))
        self.count  += 1
        self.alerts += len(alerts)

        tasks = [p.on_data(e) for p in self.plugins]
        for a in alerts:
            tasks += [p.on_alert(a, e) for p in self.plugins]
        await asyncio.gather(*tasks)

        if e.to_cloud:
            await asyncio.sleep(0.03)
            log.info(f"☁  {self.id} → cloud | {r.device} alerts={alerts}")

# ── Simulation ────────────────────────────────────────────────────────────────

async def run(ticks=6, interval=0.4):
    print("╔═══════════════════════════════════╗")
    print("║  Edge Computing Simulator (lite)  ║")
    print("╚═══════════════════════════════════╝\n")

    n1 = EdgeNode("EN-factory",   cloud_every=3).use(MQTT(), InfluxDB(), Slack())
    n2 = EdgeNode("EN-warehouse", cloud_every=5).use(MQTT(), InfluxDB(), Webhook())
    n3 = EdgeNode("EN-parking",   cloud_every=4).use(MQTT(), Slack())

    devices = {
        "T1": ("temp",     n1), "T2": ("temp",     n1),
        "H1": ("humidity", n1), "P1": ("power",    n2),
        "M1": ("motion",   n2), "G1": ("gps",      n3),
    }

    for tick in range(1, ticks + 1):
        print(f"\n── Tick {tick}/{ticks} " + "─" * 30)
        await asyncio.gather(*[
            node.process(read(dev, kind))
            for dev, (kind, node) in devices.items()
        ])
        await asyncio.sleep(interval)

    print("\n" + "═" * 45)
    for n in (n1, n2, n3):
        plugins = ", ".join(p.name for p in n.plugins)
        print(f"  {n.id}  processed={n.count}  alerts={n.alerts}  [{plugins}]")
    print("═" * 45)

if __name__ == "__main__":
    asyncio.run(run())