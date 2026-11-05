import pandas as pd
import sys

name_pattern = sys.argv[1]
trace_path = sys.argv[2]

trace = pd.read_csv(trace_path)
buckets = {}
for i, kernel_name in enumerate(trace["Kernel_Name"]):
    if name_pattern in kernel_name:
        elapse = (trace["End_Timestamp"][i] - trace["Start_Timestamp"][i]).item()
        if kernel_name in buckets:
            buckets[kernel_name].append(elapse)
        else:
            buckets[kernel_name] = [elapse]

for k, v in buckets.items():
    print(k[:200])
    print(v, sum(v[-5:]) / 5)
