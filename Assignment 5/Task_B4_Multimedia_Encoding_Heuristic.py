# TASK B4: Multimedia Framework Selection under Production Constraints
# Select video resolution, bitrate and upload timing for sample device profiles.

profiles = [
    {"name": "Low-end phone", "cpu": 2, "battery": 25, "network": 2},
    {"name": "Mid-range phone", "cpu": 5, "battery": 60, "network": 5},
    {"name": "Flagship phone", "cpu": 9, "battery": 90, "network": 9},
    {"name": "Weak network challenge", "cpu": 8, "battery": 70, "network": 1}
]

def choose_settings(device):
    # Heuristic: use the weakest of CPU, battery and network as the limiting score.
    limit = min(device["cpu"], device["battery"] // 10, device["network"])
    if limit <= 2:
        resolution, bitrate, upload = "480p", 700, "Wi-Fi / when stable"
    elif limit <= 5:
        resolution, bitrate, upload = "720p", 1800, "when network is stable"
    else:
        resolution, bitrate, upload = "1080p", 3500, "immediately if charging or battery is healthy"

    # Simple illustrative estimates.
    duration_seconds = 20
    size_mb = bitrate * duration_seconds / 8 / 1000
    energy_units = bitrate / 500 + device["cpu"] * 0.4
    upload_time = size_mb * 8 / max(device["network"], 0.5)
    battery_impact = energy_units / max(device["battery"], 1) * 100
    return resolution, bitrate, upload, size_mb, energy_units, upload_time, battery_impact

print("TASK B4 - MULTIMEDIA ENCODING HEURISTIC")
for device in profiles:
    result = choose_settings(device)
    print("\\nDevice:", device["name"])
    print("Selected resolution:", result[0])
    print("Bitrate:", result[1], "kbps")
    print("Upload timing:", result[2])
    print("Estimated clip size:", round(result[3], 2), "MB")
    print("Estimated energy units:", round(result[4], 2))
    print("Estimated upload time:", round(result[5], 2), "seconds")
    print("Estimated battery impact:", round(result[6], 2), "%")

print("\\nChallenge case:")
print("The weak-network phone has strong CPU but poor network, so the heuristic selects 480p.")
print("This may be suboptimal if it could safely encode at 720p and upload later over Wi-Fi.")
print("Production decision: monitor upload success, quality, energy, latency and user experience.")
print("Note: Estimates are illustrative; this is not a real camera encoder.")
