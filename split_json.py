import json

with open("bfcl/bfcl_eval/data/BFCL_v4_multiple.json", "r") as f:
    objs = [json.loads(line) for line in f if line.strip()]
print(len(objs))
print(objs[0])