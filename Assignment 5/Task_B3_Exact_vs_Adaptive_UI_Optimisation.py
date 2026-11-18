# TASK B3: Exact vs Approximate UI / Graphics Optimisation
# Exhaustive configuration search compared with a simple adaptive heuristic.

from itertools import product

def cost(frame_time, battery_cost, quality):
    # Lower cost is better. Frame time above 16 ms is penalised heavily.
    frame_penalty = max(0, frame_time - 16) * 5
    return frame_penalty + battery_cost + (5 - quality) * 1.5

def evaluate_config(items, quality, cache):
    # Educational cost model.
    frame_time = 7 + items * quality * 0.12 - cache * 0.35
    battery = items * quality * 0.08 + cache * 0.6
    return cost(frame_time, battery, quality), frame_time, battery

def exact_search(items, qualities, caches):
    best = None
    evaluated = 0
    for quality, cache in product(qualities, caches):
        evaluated += 1
        result = evaluate_config(items, quality, cache)
        if best is None or result[0] < best[0]:
            best = (result[0], quality, cache, result[1], result[2])
    return best, evaluated

def adaptive_search(items):
    quality = 3
    cache = 2
    changes = 0
    # Greedy rule: reduce quality if frame time exceeds 16 ms;
    # otherwise increase cache only while memory/battery budget allows.
    while changes < 10:
        _, frame_time, battery = evaluate_config(items, quality, cache)
        if frame_time > 16 and quality > 1:
            quality -= 1
        elif battery < items * 0.35 and cache < 5:
            cache += 1
        else:
            break
        changes += 1
    result = evaluate_config(items, quality, cache)
    return (result[0], quality, cache, result[1], result[2]), changes

print("TASK B3 - EXACT vs ADAPTIVE UI OPTIMISATION")
tests = [8, 15, 25]
for items in tests:
    exact, count = exact_search(items, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    adaptive, changes = adaptive_search(items)
    print("\\nFeed items:", items)
    print("Exact: cost =", round(exact[0], 2), "quality =", exact[1],
          "cache =", exact[2], "frame ms =", round(exact[3], 2),
          "configurations evaluated =", count)
    print("Adaptive: cost =", round(adaptive[0], 2), "quality =", adaptive[1],
          "cache =", adaptive[2], "frame ms =", round(adaptive[3], 2),
          "parameter changes =", changes)
    print("Costs match:", abs(exact[0] - adaptive[0]) < 0.001)

# Counter-example / larger scale operation count.
items = 40
exact, count = exact_search(items, list(range(1, 11)), list(range(1, 11)))
adaptive, changes = adaptive_search(items)
print("\\nCounter-example / larger instance (40 feed items)")
print("Exact minimum cost:", round(exact[0], 2), "with", count, "configurations evaluated")
print("Adaptive cost:", round(adaptive[0], 2), "with", changes, "parameter changes")
print("Approximation cost difference:", round(adaptive[0] - exact[0], 2))
print("Conclusion: adaptive search checks fewer choices but does not guarantee the global minimum.")
