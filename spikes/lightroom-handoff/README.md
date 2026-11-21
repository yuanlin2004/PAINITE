# Lightroom handoff spike (M0)

Throwaway code to prove the Adobe Lightroom desktop **Edit in** round trip before the Mac app's file-format and save rules are fixed (PRD, "Adobe Lightroom desktop external editor"; DESIGN §7 and §10). Lightroom Classic and the **Local** tab are out of scope.

The spike is a tiny AppKit app, **PAINITE Handoff Spike**, that registers as an image editor, logs everything about the file Lightroom hands it, and can return a visibly changed image (magenta frame) in each way the PRD asks about. It does no enhancement and talks to no server.

## Build and install (Mac)

Needs the Xcode command line tools (`xcode-select --install`).

```sh
cd spikes/lightroom-handoff
./build.sh --install     # builds, copies to ~/Applications, registers with Launch Services
```

The app is ad-hoc signed. If macOS blocks the first launch, open it once from Finder with right-click → Open.

Every event goes to `~/Library/Logs/PainiteHandoffSpike/handoff.jsonl` (one JSON object per line) and to the app window.

## What the app logs on each open

- Launch phase: `cold` (the file arrived with the launch) or `warm` (app already running).
- File: path, directory listing, inode, size, mtime, mode, owner, `file_writable`, `dir_writable`, extended attributes, UTType, SHA-256.
- Image: container type, pixel size, bit depth, color model, ICC profile name and size, orientation, alpha, TIFF software and compression, EXIF color space, IPTC presence, and any `crs:` (Camera Raw / Lightroom develop) XMP tags.

## Buttons

| Button | What it does | PRD question it answers |
| --- | --- | --- |
| Save (atomic) | Writes a marked copy to a temp file beside the handoff file, then `rename(2)`s it over the handoff path (new inode). This is the save the design proposes. | Does **Finish** pick up an atomic replace? |
| Save (in place) | Truncates and rewrites the handoff file (same inode). | Fallback if Lightroom tracks the file by inode or handle. |
| Save As… | Writes a marked copy to a separate path (default `~/Pictures`). | Minimum round trip: **Edit in → Save As → Browse**. |
| Simulate failed write | Writes a temp file, deletes it before the replace, and checks the handoff hash is unchanged. | Original preserved on a failed write. |
| Re-inspect | Logs the handoff file again, e.g. after **Finish** or **Cancel** in Lightroom. | What Lightroom does to the copy afterwards. |

## Test protocol

Record the macOS build (`sw_vers`) and the Lightroom desktop build (Lightroom → About) first. Baseline is Lightroom desktop 10.0 on macOS 27.

Inputs, both from the **cloud library** (not the Local tab), each with a visible edit such as exposure +1 and a crop:

- **R**: a camera RAW photo
- **J**: a JPEG

For each input, run each case below. "Cold" means quit the spike app first (Cmd-Q); "warm" means leave it running.

1. In Grid or Detail view select the photo, then **Edit in** → choose the spike app (first time: **Edit in other apps** / choose application). Note every option Lightroom offers in its dialog (file type, bit depth, color space, resolution) and what you picked. Start with Lightroom's defaults.
2. Confirm the spike window shows an `open` record with the right launch phase.
3. Return path, one per run:
   - a. **Save As…** to `~/Pictures`, then in Lightroom use **Browse** to pick that file. Does the magenta frame show in Lightroom? Is it a new photo or a new version?
   - b. **Save (atomic)**, then **Finish** in Lightroom. Does the frame show?
   - c. **Save (in place)**, then **Finish**. Does the frame show?
   - d. Don't save; **Cancel** (or Finish without a save) in Lightroom. Press **Re-inspect**. Is the cloud original unchanged?
   - e. **Simulate failed write**, then **Finish**. `handoff_unchanged` must be `true` and Lightroom must not show a frame.
4. After each run press **Re-inspect** to see whether Lightroom deleted or changed the handoff copy.

Repeat step 1 once with 16-bit TIFF and once with a wide-gamut color space (ProPhoto or Display P3) if Lightroom offers those, to see what reaches the app.

When done, copy `~/Library/Logs/PainiteHandoffSpike/handoff.jsonl` into `results/` and fill in [RESULTS.md](RESULTS.md).
