# PAINITE.app

The macOS app: open a photo, compare the server's enhancement against the
original at matched zoom, and export the full-resolution result. It also works
as a Lightroom desktop **Edit in** target. Design: [`docs/DESIGN.md`](../docs/DESIGN.md) §7.
Contract: [`api/openapi.yaml`](../api/openapi.yaml).

**Status: M2 first cut, built against the placeholder models.** Denoise and
sharpen return the original and upscale is a plain resize until the real
models land; the app marks them as placeholders.

## Build and run

Needs Swift 6 (Xcode or its command line tools) on Apple silicon. The PRD
targets macOS 27; the code builds for macOS 14 and later.

```sh
cd mac
./build.sh              # build/PAINITE.app
./build.sh --install    # also copy to ~/Applications and register it with Launch Services
open build/PAINITE.app
swift test              # unit tests (no server needed)
PAINITE_LIVE=1 swift test --filter LiveServerTests   # job flow against the configured server
```

`rsvg-convert` (`brew install librsvg`) is used to make the app icon from
`docs/logo/painite-icon.svg`; without it the app builds with no icon.

## Server settings

Settings (⌘,) holds the server address, token, and optional CA certificate.
They live in the same file as the CLI's settings,
`~/Library/Application Support/painite/config.json` (mode 0600), so configuring
either client configures both. Create a token on the server with
`painite-server token create --name <name>`.

The test server on the LAN speaks plain HTTP until TLS setup is built; the app
allows that only for local-network hosts. macOS asks for **Local Network**
permission the first time the app connects.

## How it works

| Area | Behavior |
| --- | --- |
| Steps | The sidebar holds up to 4 steps (Denoise, Sharpen, Upscale 2×) that run top to bottom as one server job. Toggle, reorder, or remove each step; Upscale can appear once and is added last. |
| Open | Finder, File › Open, drag onto the welcome window, or Lightroom **Edit in**. JPEG and TIFF only, as the server supports. |
| Upload | Starts as soon as the photo opens. Content-addressed by SHA-256: `HEAD /sources/{sha}` first, so a photo the server already has is not sent again. If the server dropped it, the app uploads again and retries the job. |
| Preview area | A box you pick, at most 2048 px square (1024 with upscale), starting in the middle of the photo. Zooming and panning never change it. **Select Area** (⌘E) then drag to draw a new box, drag inside the box to move it, or click to center it there; Shift-drag does the same at any time; Esc cancels. **Preview Area in View** (⇧⌘E) picks what is on screen. A preview is rendered only when the area or the steps change. Lossless TIFF, so compression doesn't hide noise. The last six previews are cached. |
| Preview box | Shown on both sides, snapped to the server's tile grid. It pulses yellow while the preview is rendering, turns white when the preview inside it is current, and orange if it failed. |
| Compare | Original and enhanced side by side, always at the same zoom and position. 100% is one image pixel per screen pixel. The enhanced pane shows the original dimmed outside the previewed region. Upscaled previews are drawn into the same source region so the framing matches. Display is color managed through Core Animation in the file's own color space. |
| Export | Same steps, strengths, and models as the preview (the model is pinned from the preview). Progress, cancel, and retry. Downloads resume with HTTP Range and are checked against the result's SHA-256, written to a hidden temp file beside the destination, then renamed into place, so a failed or canceled export never touches an existing file. The original is never overwritten. |
| Model gone | If the previewed model is no longer installed at export, the app says so and asks for a new preview. |

## Lightroom desktop

From [`spikes/lightroom-handoff/RESULTS.md`](../spikes/lightroom-handoff/RESULTS.md):
Lightroom 10.0 hands over a fresh 16-bit ProPhoto TIFF in `$TMPDIR` for every
Edit in. The app treats a TIFF whose Software tag starts with "Adobe Lightroom"
and that sits in the temporary folder as a handoff.

- **Save** (⌘S) appears only when that handoff file and its folder are
  writable. It replaces the file atomically, in the same format; then click
  **Finish** in Lightroom.
- **Save As** (⇧⌘S) is always offered, defaulting to `~/Pictures/<name>-painite.tif`.
  Bring it back with **Browse…** (JPEG sources) or **Select…** (RAW sources,
  where Lightroom offers no Finish).
- Opens are handled whenever they arrive, before or after launch finishes; the
  app does not rely on telling cold and warm launches apart.

Setup: in Lightroom, **Edit in › Browse…** and choose `~/Applications/PAINITE.app`
(after `./build.sh --install`).
