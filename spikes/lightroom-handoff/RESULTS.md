# Lightroom handoff spike results

Status: **not yet run**

- Date:
- macOS build:
- Lightroom desktop build:
- Spike commit:

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
