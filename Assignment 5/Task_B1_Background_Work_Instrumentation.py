# TASK B1: Implement and Instrument Background Architecture
# Simulates background sensor uploads with constraints and two scheduling orders.

from dataclasses import dataclass

@dataclass
class Work:
    name: str
    priority: int
    critical: bool
    needs_network: bool = True

def run_scheduler(items, states, priority_mode=False):
    queue = items[:]
    if priority_mode:
        queue.sort(key=lambda w: (w.critical, w.priority), reverse=True)

    enqueued = len(queue)
    deferred = 0
    cancelled = 0
    successful = 0
    peak_workers = 0
    log = []

    for index, work in enumerate(queue):
        state = states[index % len(states)]
        battery = state["battery"]
        network = state["network"]
        doze = state["doze"]
        permission = state["permission"]

        # At most one worker runs at a time in this simple simulation.
        peak_workers = max(peak_workers, 1)

        if not permission:
            cancelled += 1
            log.append(work.name + ": CANCELLED (permission revoked)")
        elif work.needs_network and not network:
            deferred += 1
            log.append(work.name + ": DEFERRED (no network)")
        elif battery < 20 and not work.critical:
            deferred += 1
            log.append(work.name + ": DEFERRED (low battery)")
        elif doze and not work.critical:
            deferred += 1
            log.append(work.name + ": DEFERRED (Doze mode)")
        else:
            successful += 1
            log.append(work.name + ": UPLOADED")

    return {
        "enqueued": enqueued, "deferred": deferred, "cancelled": cancelled,
        "successful": successful, "peak": peak_workers, "log": log
    }

items = [
    Work("Routine sensor A", 2, False),
    Work("Critical heart alert", 10, True),
    Work("Routine sensor B", 3, False),
    Work("Daily summary", 1, False),
    Work("Critical fall alert", 9, True),
    Work("Routine sensor C", 2, False)
]

good_states = [
    {"battery": 85, "network": True, "doze": False, "permission": True}
]
restricted_states = [
    {"battery": 12, "network": False, "doze": True, "permission": True},
    {"battery": 60, "network": True, "doze": True, "permission": True}
]
permission_test = [
    {"battery": 80, "network": True, "doze": False, "permission": True},
    {"battery": 80, "network": True, "doze": False, "permission": False}
]

print("TASK B1 - BACKGROUND WORK INSTRUMENTATION")
for label, states in [
    ("Good connectivity and battery", good_states),
    ("Doze / low battery / network restriction", restricted_states),
    ("Permission revoked boundary test", permission_test)
]:
    result = run_scheduler(items, states, priority_mode=True)
    print("\\nScenario:", label)
    print("Enqueued:", result["enqueued"], "Deferred:", result["deferred"],
          "Cancelled:", result["cancelled"], "Successful:", result["successful"],
          "Peak workers:", result["peak"])
    for line in result["log"]:
        print(" ", line)

fifo = run_scheduler(items, good_states, priority_mode=False)
priority = run_scheduler(items, good_states, priority_mode=True)
print("\\nFIFO vs priority under good conditions")
print("FIFO successful:", fifo["successful"], "| Priority successful:", priority["successful"])
print("Priority order:", [x.split(":")[0] for x in priority["log"]])
print("Note: Simplified single-worker model; Android WorkManager is not invoked.")
