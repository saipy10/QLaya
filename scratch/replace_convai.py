import os

TARGET_DIRS = ["qlaya-dart", "qlaya-ts", "qlaya", "docs", "examples", "tests", "scripts", "benchmarks"]
INDIVIDUAL_FILES = ["pyproject.toml", "HF_README.md", "README.md", "BENCHMARKS.md", "Dockerfile"]

SKIP_DIRS = {".git", ".dart_tool", "node_modules", "dist", "build", ".venv", "qlaya.egg-info", ".pytest_cache", "models"}
SKIP_EXTS = {".onnx", ".data", ".bin", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".pdf", ".lock"}

files_to_check = []
for f in INDIVIDUAL_FILES:
    if os.path.exists(f):
        files_to_check.append(f)

for d in TARGET_DIRS:
    if not os.path.exists(d):
        continue
    for root, dirs, files in os.walk(d):
        dirs[:] = [sub for sub in dirs if sub not in SKIP_DIRS]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SKIP_EXTS:
                continue
            files_to_check.append(os.path.join(root, f))

total_replaced = 0
modified_files = []

for filepath in files_to_check:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        continue

    new_content = content
    # Replace convaiinnovations -> saipy10
    if "convaiinnovations" in new_content:
        new_content = new_content.replace("convaiinnovations", "saipy10")
    if "Convai Innovations" in new_content:
        new_content = new_content.replace("Convai Innovations", "saipy10")

    if new_content != content:
        count = content.count("convaiinnovations") + content.count("Convai Innovations")
        total_replaced += count
        modified_files.append((filepath, count))
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

print(f"Total replacements: {total_replaced} across {len(modified_files)} files.")
for path, count in sorted(modified_files, key=lambda x: x[0]):
    print(f"  {count:2d}  {path}")
