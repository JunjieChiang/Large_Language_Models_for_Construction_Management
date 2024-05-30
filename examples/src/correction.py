import json

def check_data_consistency(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            try:
                data = json.loads(line)
                # 确保使用正确的键名 'query' 来检查类型是否为字符串
                if not isinstance(data['name'], str):
                    print(f"Type mismatch in line {i+1}: expected str for 'name', found {type(data['name']).__name__} instead.")
            except json.JSONDecodeError as e:
                print(f"JSON decode error in line {i + 1}: {e}")
            except KeyError as e:
                # 如果某个预期的键不存在
                print(f"Key error in line {i + 1}: '{e}' key is missing")
            except Exception as e:
                # 处理可能的其他异常
                print(f"Unexpected error in line {i + 1}: {e}")

# 替换 'path_to_file.jsonl' 为你的文件路径
check_data_consistency('examples/bim_kb.jsonl')