# Lightroom handoff spike results

Status: **round trip proven for edited cloud JPEGs; edited RAW run pending**

- Date: 2026-11-21
- macOS build: macOS 27.2.1 (26B131), Apple silicon
- Swift: Apple Swift 6.2
- Lightroom desktop build: 10.0 (com.adobe.lightroomCC, CFBundleVersion 10.0). Lightroom Classic not installed.
- Spike commit for the runs: eced50b (the alpha and cold-launch fixes below came after)
- Raw log: `spikes/lightroom-handoff/results/handoff-2026-11-21.jsonl` in the Mac clone (not committed; it lists temp-folder contents)

## Smoke test without Lightroom

`open -a` on a 6016×6016 Display P3 JPEG: the cold open arrived through `application(_:open:)` before `didFinishLaunching` and was logged `cold`. A second open in the same process was logged `warm`. Both files were writable and had no `crs:` XMP tags. The build needed no code changes.

## Getting into the Edit in menu

Lightroom desktop builds the third-party part of **Edit in** from one JSON file per app in `<library>.lrlibrary/ExternalEditors/`, e.g.

```json
{
    "appPath": "/Applications/PAINITE Handoff Spike.app",
    "displayName": "PAINITE Handoff Spike",
    "creator": "Adobe Lightroom 10.0",
    "guid": "5360D535-918C-4F7E-8350-468823931608"
}
```

The submenu normally shows the listed apps, then **Browse...** (choose an app) and **Clear...**. On the test Mac the submenu was completely blank at first, with no Browse item, even though Lightroom's code always adds Browse. It filled in only after the PAINITE entry was added by hand and Lightroom was relaunched. The cause of the blank menu is not known.

**Implication for the Mac app:** document **Edit in → Browse... → PAINITE** as the setup step. Consider having the app's installer or first run write the ExternalEditors entry, but only after checking that this is acceptable and stable across Lightroom versions.

## Edit in dialog

Lightroom offered **no export options** (no file type, bit depth, color space or size). After choosing the app, Lightroom shows an "Edit in PAINITE Handoff Spike" dialog with **Cancel** and **Finish** buttons, then exports a file and opens it in the app.

## What arrives

The inputs were three edited cloud-library JPEGs from a Sony DSC-RX10M4 (YL_04697/98/99.JPG: 5472×3648, 8-bit).

| Property | Value |
| --- | --- |
| Location | `$TMPDIR` (`/private/var/folders/…/T/<name>.tif`), a fresh file for each Edit in |
| Format | TIFF, Deflate compression (8), software "Adobe Lightroom 10.0 (Macintosh)" |
| Size | ~97–105 MB at 3648×5472 |
| Bit depth | 16 bits per channel, RGB, no alpha (48 bpp) |
| Color | ProPhoto RGB ICC profile (940 bytes) |
| Orientation | Rotation baked into the pixels (portrait), no Orientation tag |
| Metadata | IPTC present, 35 EXIF keys, ~175 XMP tags including ~40 `crs:` develop settings |
| Adjustments | Baked into the pixels (`crs:AlreadyApplied`) |
| Writability | File and folder writable by the user, nlink 1 |
| Separate copy | Yes. The cloud original is never handed over. |

The PRD expected JPEG or TIFF. In practice the handoff is always a **16-bit ProPhoto TIFF**, even for an 8-bit JPEG source.

## Return paths

| Case | Launch | Result |
| --- | --- | --- |
| Save (atomic: temp + rename, new inode) → Finish | cold | ✅ Imported as a new photo stacked with the original; magenta frame visible |
| Save (in place: same inode) → Finish | warm | ✅ Same as above |
| Save As to `~/Pictures` → Finish → "No edits made" → **Browse...** → pick file | warm | ✅ Imported and stacked with the original |
| Simulated failed write → Cancel | warm | ✅ Handoff hash unchanged; nothing imported; original untouched |

After Finish, Lightroom copied the saved file into the library's `originals/YYYY/YYYY-MM-DD/` as a new original, stacked with the source. It then deleted the handoff TIFF from `$TMPDIR`. Cancel also deleted it. The cloud original is never modified.

If nothing changed at the handoff path, Finish shows "No edits made" with **Cancel / Browse... / Try again**. That is where the Save As path comes back in.

## Findings and decisions

- **Minimum round trip (Edit in → Save As → Browse):** works.
- **Direct save:** works with both atomic rename and in-place rewrite, cold and warm. The design's atomic replace is fine.
- **Format matrix:** the Lightroom path only needs to read and write 16-bit ProPhoto RGB Deflate TIFF without alpha, preserving EXIF/IPTC/XMP. Output must stay RGB-only. The first spike writer added an alpha channel; this is fixed in the spike now.
- **Cold vs warm:** on a cold launch, Lightroom's open event arrives just *after* `didFinishLaunching` (~0.5 s). The app must handle the open whenever it arrives, and must not assume an open before launch finishes. The spike now treats the first open within 2 s of start as cold.
- **Originals:** the handoff is always a temporary copy, so "save over the handoff file" can never touch the cloud original. Lightroom adds the result as a new stacked photo, and the user can delete it.
- **Size:** 100 MB TIFFs per 20 MP photo matter for upload time to the server. Consider compressing them losslessly for upload, or reducing the bit depth on the client, before sending.

## Still to do

- Repeat with an edited camera RAW from the cloud library.
- Rerun one save after the alpha fix to confirm the output is 3-channel.
- Remove the test results from Lightroom: 3 stacked TIFFs in "2026-11-16 MVHS Home Show Grace Extra", and `~/Pictures/YL_04699-painite.tif`.
