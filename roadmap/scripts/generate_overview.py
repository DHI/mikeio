# /// script
# requires-python = ">=3.10"
# dependencies = ["jinja2", "pyyaml"]
# ///
"""Generate roadmap/README.md from feature page YAML frontmatter."""

from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

ROADMAP_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROADMAP_DIR / "features"
TEMPLATE_DIR = ROADMAP_DIR / "templates"

STATUSES = [
    "Delivered",
    "In Development",
    "Planned",
    "Under Consideration",
    "Not Planned",
]


def parse_frontmatter(path: Path) -> dict:
    """Extract YAML frontmatter from a markdown file."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"No YAML frontmatter found in {path}")
    _, frontmatter, _ = text.split("---", 2)
    meta = yaml.safe_load(frontmatter)
    meta["filename"] = path.name
    return meta


def main():
    features = []
    for path in sorted(FEATURES_DIR.glob("*.md")):
        features.append(parse_frontmatter(path))

    features_by_status: dict[str, list[dict]] = {s: [] for s in STATUSES}
    for f in features:
        status = f["status"]
        if status in features_by_status:
            features_by_status[status].append(f)

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), keep_trailing_newline=True)
    template = env.get_template("overview.md.j2")
    readme = template.render(statuses=STATUSES, features_by_status=features_by_status)

    output_path = ROADMAP_DIR / "README.md"
    output_path.write_text(readme, encoding="utf-8")
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
