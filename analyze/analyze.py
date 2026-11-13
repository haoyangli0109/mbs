import sys
import json
import os
 
att_trace_path = sys.argv[1]
filter_pattern = sys.argv[2]
 
def parse_one_file(att_trace_file: str):
    with open(att_trace_file, "r") as f:
        att_trace = json.load(f)
    if filter_pattern not in att_trace["code"][0][0]:
        return
    rows = [["instruction", " Hit", " Latency", " Stall", " Idle"]]
 
    for item in att_trace["code"]:
        rows.append([item[0], str(item[-4]), str(item[-3]), str(item[-2]), str(item[-1])])
 
    max_len = [0, 0, 0, 0, 0]
    for r in rows:
        for i in range(5):
            max_len[i] = max(max_len[i], len(r[i]))
 
    for r in rows:
        msg = ""
        for i in range(5):
            msg += " " * (max_len[i] - len(r[i]) + 1) + r[i]
        print(msg)
    print("========================" * 4)
 
if os.path.isdir(att_trace_path):
    for item in os.listdir(att_trace_path):
        if item.startswith("ui_output_agent_"):
            parse_one_file(os.path.join(att_trace_path, item, "code.json"))
else:
    parse_one_file(att_trace_path)
 
 