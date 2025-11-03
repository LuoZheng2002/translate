import json

def load_json_lines(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
    return data

# # Example usage:
# items = load_json_lines("dataset/BFCL_v4_multiple.json")
# print(items[0:2])  # Print first two items to verify
