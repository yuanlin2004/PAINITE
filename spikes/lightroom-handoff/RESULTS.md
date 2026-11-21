# Lightroom handoff spike results

Status: **app built and smoke-tested; Lightroom runs pending**

- Date: 2026-11-21
- macOS build: macOS 27.2.1 (26B131), Apple silicon
- Swift: Apple Swift 6.2
- Lightroom desktop build: 10.0 (com.adobe.lightroomCC, CFBundleVersion 10.0). Lightroom Classic not installed.
- Spike commit: eced50b

## Smoke test without Lightroom

`open -a` on a 6016×6016 Display P3 JPEG: the cold open was delivered through `application(_:open:)` before `didFinishLaunching` (logged `cold`), and a second open in the same process was logged `warm`. Both files writable, no `crs:` XMP tags. Build needed no code changes.

## Edit in dialog

Options Lightroom offers when sending to a third-party app (file type, bit depth, color space, resolution, other):

## What arrives

| Input | Launch | Lightroom setting | Path / location | Format | Bit depth | ICC profile | Adjustments baked in? | Separate copy? | File writable | Dir writable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW | cold | default | | | | | | | | |
| RAW | warm | default | | | | | | | | |
| JPEG | cold | default | | | | | | | | |
| JPEG | warm | default | | | | | | | | |

## Return paths

| Input | Launch | Save As → Browse | Save (atomic) → Finish | Save (in place) → Finish | Cancel keeps original | Failed write keeps handoff |
| --- | --- | --- | --- | --- | --- | --- |
| RAW | cold | | | | | |
| RAW | warm | | | | | |
| JPEG | cold | | | | | |
| JPEG | warm | | | | | |

## Findings and decisions

- Minimum round trip (Edit in → Save As → Browse):
- Direct save supported cases:
- Save method for the Mac app (atomic rename vs in place):
- Format matrix for the PRD:
- What happens to the handoff copy after Finish/Cancel:
