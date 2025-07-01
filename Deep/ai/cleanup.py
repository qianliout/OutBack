
import os
import shutil

base_dir = "/Users/liuqianli/Library/Mobile Documents/iCloud~md~obsidian/Documents/OutBack/Deep/ai/ai_note/"
project_root = "/Users/liuqianli/Library/Mobile Documents/iCloud~md~obsidian/Documents/OutBack/Deep/ai/"

# Delete original subdirectories
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path):
        print(f"Deleting directory: {item_path}")
        shutil.rmtree(item_path)

# Delete temporary Python scripts
scripts_to_delete = [
    os.path.join(project_root, "merge_and_map.py"),
    os.path.join(project_root, "update_links.py")
]
for script_path in scripts_to_delete:
    if os.path.exists(script_path):
        print(f"Deleting script: {script_path}")
        os.remove(script_path)

# Delete the mapping JSON file
map_file_path = os.path.join(base_dir, "original_to_merged_map.json")
if os.path.exists(map_file_path):
    print(f"Deleting map file: {map_file_path}")
    os.remove(map_file_path)

print("清理完成。")
