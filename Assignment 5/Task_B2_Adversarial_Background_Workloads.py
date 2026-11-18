# TASK B2: Engineer an Adversarial Workload
# Same number of records, but different connectivity / battery / permission conditions.

import random

def simulate_workload(name, records, states):
    processed = 0
    deferred = 0
    cancelled = 0
    delivered = 0
    retries = 0

    for i, record in enumerate(records):
        processed += 1
        state = states[i]
        if not state["permission"]:
            cancelled += 1
        elif not state["network"] or state["battery"] < 20 or state["doze"]:
            deferred += 1
            retries += 1
        else:
            delivered += 1

    print("\\n", name)
    print("Records processed:", processed)
    print("Deferred:", deferred, "| Cancelled:", cancelled,
          "| Successful deliveries:", delivered, "| Retry opportunities:", retries)

records = ["record_" + str(i) for i in range(1, 21)]

# Workload A: stable conditions, all records can upload.
workload_a = [
    {"network": True, "battery": 80, "doze": False, "permission": True}
    for _ in records
]

# Workload B: same record count, alternating poor network, Doze and low battery.
workload_b = []
for i in range(len(records)):
    workload_b.append({
        "network": i % 3 == 0,
        "battery": 15 if i % 2 == 0 else 55,
        "doze": i % 4 == 0,
        "permission": True
    })

print("TASK B2 - SAME INPUT SIZE, DIFFERENT WORKLOAD STRUCTURE")
print("Prediction: Workload B will have more deferrals because its constraints are often unsatisfied.")
simulate_workload("Workload A - easy / stable", records, workload_a)
simulate_workload("Workload B - adversarial / unstable", records, workload_b)

print("\\nInterpretation:")
print("Input size is 20 in both cases, but constraint failures cause more waiting and retries in B.")
print("The structure of network, battery and Doze states determines practical effort.")
