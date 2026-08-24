"""Generate llms.txt for the MIKE IO documentation site.

This script produces a curated markdown index following the llms.txt spec
(https://llmstxt.org/) that helps LLMs understand and navigate the project.

Links are extracted from _quarto.yml (sidebar pages and quartodoc API sections).
Page titles are read from YAML frontmatter or first H1 heading in each source file.

Run as part of the docs build (after quarto render):
    python generate_llms_txt.py
"""

import re
from pathlib import Path

import yaml

BASE_URL = "https://dhi.github.io/mikeio"
QUARTO_YML = Path("_quarto.yml")
OUTPUT = Path("_site/llms.txt")


def get_title(filepath: str) -> str:
    """Extract the title from a .qmd or .md file.

    Checks YAML frontmatter 'title' field first, then falls back to
    the first H1 heading. Returns the filename stem as a last resort.
    """
    path = Path(filepath)
    if not path.exists():
        return path.stem.replace("-", " ").title()

    text = path.read_text()

    # Check YAML frontmatter
    fm_match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if fm_match:
        try:
            fm = yaml.safe_load(fm_match.group(1))
            if isinstance(fm, dict) and fm.get("title"):
                return str(fm["title"])
        except yaml.YAMLError:
            pass

    # Fall back to first H1
    h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if h1_match:
        # Strip Quarto shortcodes like {{< fa icon >}}
        title = re.sub(r"\{\{<.*?>}}", "", h1_match.group(1)).strip()
        # Strip markdown bold
        title = re.sub(r"\*\*(.+?)\*\*", r"\1", title)
        return title

    return path.stem.replace("-", " ").title()


def qmd_to_url(filepath: str) -> str:
    """Convert a .qmd/.md source path to its URL on the docs site."""
    return BASE_URL + "/" + re.sub(r"\.(qmd|md)$", ".html", filepath)


def get_description(filepath: str) -> str:
    """Extract the description from YAML frontmatter, if present."""
    path = Path(filepath)
    if not path.exists():
        return ""

    text = path.read_text()
    fm_match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if fm_match:
        try:
            fm = yaml.safe_load(fm_match.group(1))
            if isinstance(fm, dict) and fm.get("description"):
                return str(fm["description"])
        except yaml.YAMLError:
            pass
    return ""


def main() -> None:
    """Generate llms.txt and write it to _site/llms.txt."""
    config = yaml.safe_load(QUARTO_YML.read_text())

    lines = [
        "# MIKE IO",
        "",
        "> Python package for reading, writing, and manipulating"
        " MIKE files (dfs0, dfs1, dfs2, dfs3, dfsu, mesh) from DHI.",
        "",
    ]

    # Sidebar sections (User Guide, Examples)
    for sidebar in config["website"]["sidebar"]:
        title = sidebar["title"]
        lines.append(f"## {title}")
        lines.append("")
        for item in sidebar["contents"]:
            if not isinstance(item, str):
                continue
            if not Path(item).exists():
                continue
            page_title = get_title(item)
            url = qmd_to_url(item)
            desc = get_description(item)
            entry = f"- [{page_title}]({url})"
            if desc:
                entry += f": {desc}"
            lines.append(entry)
        lines.append("")

    # API Reference from quartodoc sections
    api_dir = config["quartodoc"].get("dir", "api")
    lines.append("## API Reference")
    lines.append("")
    for section in config["quartodoc"]["sections"]:
        for item in section["contents"]:
            # Strip inline comments (e.g. "dfsu.DfsuSpectral # comment")
            name = item.split("#")[0].strip()
            # The API page filename uses the dotted name
            url = f"{BASE_URL}/{api_dir}/{name}.html"
            lines.append(f"- [{name}]({url})")
    lines.append("")

    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
