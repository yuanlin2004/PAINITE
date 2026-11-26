"""Build the PAINITE user guide into site/_site/ for GitHub Pages.

Each page in pages/ starts with `title:` and `description:` lines, then a line
holding `---`, then the page's HTML body. This script wraps each body in the
shared header and footer and copies assets/. {{version}} in a page becomes the
release its download links point at (see release_version). No dependencies.

    python3 site/build.py && open site/_site/index.html
"""
import html
import json
import os
import pathlib
import re
import shutil

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "_site"

NAV = [
    ("index.html", "Home"),
    ("install.html", "Install"),
    ("mac.html", "Mac app"),
    ("cli.html", "Command line"),
    ("agents.html", "AI agents"),
    ("enhancements.html", "Enhancements"),
    ("whatsnew.html", "What's new"),
    ("help.html", "Help"),
]
CURRENT = ' aria-current="page"'

# Google Tag Manager container for the guide (analytics).
GTM_ID = "GTM-MR9QX638"
GTM_HEAD = f"""  <!-- Google Tag Manager -->
  <script>(function(w,d,s,l,i){{w[l]=w[l]||[];w[l].push({{'gtm.start':
  new Date().getTime(),event:'gtm.js'}});var f=d.getElementsByTagName(s)[0],
  j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
  'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
  }})(window,document,'script','dataLayer','{GTM_ID}');</script>
  <!-- End Google Tag Manager -->"""
GTM_BODY = f"""<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id={GTM_ID}"
height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->"""


def page(fname: str, title: str, body: str, desc: str) -> str:
    nav = "\n".join(f'      <a href="{h}"{CURRENT if h == fname else ""}>{t}</a>' for h, t in NAV)
    t = "PAINITE" if fname == "index.html" else f"{title} · PAINITE"
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
{GTM_HEAD}
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{t}</title>
  <meta name="description" content="{desc}">
  <link rel="icon" href="assets/icon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
{GTM_BODY}
<header class="site-header">
  <div class="inner">
    <a class="brand" href="index.html"><img src="assets/icon.svg" alt="">PAINITE</a>
    <nav aria-label="Site">
{nav}
    </nav>
  </div>
</header>
<main>
{body}
</main>
<footer class="site-footer">
  <div class="inner">
    <span>PAINITE © 2026 Yuan Lin · <a href="https://github.com/yuanlin2004/PAINITE/blob/main/LICENSE">Apache License 2.0</a></span>
    <a href="https://github.com/yuanlin2004/PAINITE">Source code</a>
    <a href="https://github.com/yuanlin2004/PAINITE/releases">Releases</a>
    <a href="https://github.com/yuanlin2004/PAINITE/issues">Report a problem</a>
  </div>
</footer>
<script src="assets/compare.js"></script>
</body>
</html>
'''


def compare(op: str, sample: dict) -> str:
    """A before/after slider for assets/samples/<op>/."""
    base = f"assets/samples/{op}"
    return f'''<div class="compare">
      <img src="{base}/before.jpg" alt="{html.escape(sample["title"])}: before" loading="lazy">
      <div class="after-wrap"><img src="{base}/after.jpg" alt="{html.escape(sample["title"])}: after PAINITE" loading="lazy"></div>
      <input type="range" min="0" max="100" value="50" aria-label="Drag to compare before and after">
      <div class="divider"></div>
      <span class="tag before">BEFORE</span><span class="tag after">AFTER</span>
    </div>'''


def expand_samples(body: str, samples: dict) -> str:
    """<!--SAMPLES--> becomes every sample with its caption; <!--SAMPLE:op--> one slider."""
    def one(op: str) -> str:
        s = samples[op]
        return f'''<figure class="sample">
    <h3>{s["title"]}</h3>
    <p class="meta">{s["meta"]}</p>
    {compare(op, s)}
    <figcaption>{s["caption"]}</figcaption>
  </figure>'''
    body = body.replace("<!--SAMPLES-->", "\n".join(one(op) for op in samples))
    return re.sub(r"<!--SAMPLE:(\w+)-->", lambda m: f"<figure>{compare(m[1], samples[m[1]])}"
                  f"<figcaption>{samples[m[1]]['caption']}</figcaption></figure>", body)


def release_version() -> str:
    """The version the download links point at, without the leading v.

    PAINITE_VERSION (set by the Pages workflow to the latest published
    release) wins; otherwise the newest docs/release-notes/v*.md.
    """
    if v := os.environ.get("PAINITE_VERSION", "").strip():
        return v.removeprefix("v")
    notes = (ROOT.parent / "docs" / "release-notes").glob("v*.md")
    versions = [tuple(int(x) for x in n.stem[1:].split(".")) for n in notes if re.fullmatch(r"v\d+\.\d+\.\d+", n.stem)]
    return ".".join(map(str, max(versions)))


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(ROOT / "assets", OUT / "assets")
    (OUT / ".nojekyll").touch()
    samples = json.loads((ROOT / "assets" / "samples" / "samples.json").read_text())
    version = release_version()
    for f in sorted((ROOT / "pages").glob("*.html")):
        head, body = f.read_text().split("\n---\n", 1)
        body = expand_samples(body, samples).replace("{{version}}", version)
        meta = dict(line.split(": ", 1) for line in head.strip().splitlines())
        (OUT / f.name).write_text(page(f.name, meta["title"], body.strip("\n"), meta["description"]))
    print(f"built {OUT} (downloads: v{version})")


if __name__ == "__main__":
    main()
