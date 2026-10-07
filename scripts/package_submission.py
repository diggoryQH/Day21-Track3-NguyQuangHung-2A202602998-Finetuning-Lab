import shutil
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
dist = root / "lab21_2A202602998"
zip_path = root / "lab21_2A202602998.zip"

if dist.exists():
    shutil.rmtree(dist)
if zip_path.exists():
    zip_path.unlink()

(dist / "submission").mkdir(parents=True, exist_ok=True)
(dist / "results").mkdir(parents=True, exist_ok=True)

# Copy submission docs
shutil.copy2(root / "submission" / "REPORT.md", dist / "submission" / "REPORT.md")
if (root / "submission" / "REFLECTION.md").exists():
    shutil.copy2(root / "submission" / "REFLECTION.md", dist / "submission" / "REFLECTION.md")

# Copy LINKS.md
shutil.copy2(root / "LINKS.md", dist / "LINKS.md")

# Copy all result files except .gitkeep
for f in (root / "results").glob("*"):
    if f.is_file() and f.name != ".gitkeep":
        shutil.copy2(f, dist / "results" / f.name)

# Create zip archive
shutil.make_archive(str(root / "lab21_2A202602998"), "zip", root, "lab21_2A202602998")
print(f"Packaged successfully to: {zip_path}")
