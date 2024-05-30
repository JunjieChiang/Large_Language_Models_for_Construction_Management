import json

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    walls = data.get("walls", [])
    doors = data.get("doors", [])
    windows = data.get("windows", [])
    columns = data.get("columns", [])
    slabs = data.get("slabs", [])
    beams = data.get("beams", [])

    # Open a new file to write the processed data
    with open('examples/bim_kb.jsonl', 'w', encoding='utf-8') as output_file:
        for wall in walls:
            output_file.write(json.dumps(wall, ensure_ascii=False) + '\n')

        for door in doors:
            output_file.write(json.dumps(door, ensure_ascii=False) + '\n')

        for window in windows:
            output_file.write(json.dumps(window, ensure_ascii=False) + '\n')

        for column in columns:
            output_file.write(json.dumps(column, ensure_ascii=False) + '\n')

        for slab in slabs:
            output_file.write(json.dumps(slab, ensure_ascii=False) + '\n')

        for beam in beams:
            output_file.write(json.dumps(beam, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    file_path = 'examples/BIM.json' # 替换为你的JSON文件路径
    process_json_file(file_path)
