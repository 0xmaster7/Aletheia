import time
import sys
import platform
import subprocess

sys.path.insert(0, './scripts')
from _pipeline import native_route

def get_cpu_info():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8").strip()
    except:
        return platform.processor()

# Prepare log file
log_file = open("routing_latency_test.log", "w")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")

log("=== Aletheia Routing Latency Benchmark ===")
log(f"CPU Model: {get_cpu_info()}")
log(f"Platform: {platform.platform()}\n")

log("Warming up model (loading PyTorch weights)...")
t0 = time.perf_counter()
native_route("What is the current value?")
t1 = time.perf_counter()
log(f"Warmup call took: {(t1 - t0) * 1000:.2f} ms\n")

queries = [
    "How many different cities has Keshav lived in?",
    "Did Keshav ever work for Microsoft?",
    "What was the original recorded value for the entity?",
    "Who is the current person in charge?"
]

log("Running 100 benchmark iterations...")
latencies = []
for i, q in enumerate(queries * 25):
    start = time.perf_counter()
    native_route(q)
    end = time.perf_counter()
    lat_ms = (end - start) * 1000
    latencies.append(lat_ms)

log(f"\n--- Results ---")
log(f"Total Runs: {len(latencies)}")
log(f"Average Latency: {sum(latencies) / len(latencies):.2f} ms")
log(f"Min Latency: {min(latencies):.2f} ms")
log(f"Max Latency: {max(latencies):.2f} ms (Iteration {latencies.index(max(latencies)) + 1})")
log(f"Median Latency: {sorted(latencies)[len(latencies)//2]:.2f} ms")

log_file.close()
