# PAINITE Mac app design

Mockups: `mockup-main-dark.jpg` (ready), `mockup-main-light.jpg` (Lightroom document, preview updating), `mockup-welcome.jpg`. The HTML source of the mockups is kept outside the repo.

## Principles

1. **The photo is the interface.** The canvas shows what was rendered, what is stale, and what is next. Status sits on the pixels, not in a header.
2. **One decision per glance.** The main question is "does it look better?" Model IDs, timings, upload messages and help text are not on the main screen.
3. **Nothing moves by itself.** Zoom and pan never move the preview area. The app can point to the area but never relocates it.
4. **Native first.** Use a unified toolbar, an inspector, standard switches, sliders and menus, and system materials. The one brand touch is the orange accent.

## Window layout

- One `NSWindow` with `.fullSizeContentView` and a unified `NSToolbar`. The custom 30 pt header bar is removed.
- The canvas fills the window behind the toolbar.
- On the right is a floating inspector: SwiftUI `.inspector`, 296 pt wide, inset 8 pt, 20 pt corner radius, sidebar material. Toggle it with ⌥⌘I.
- Minimum window size is 960×600. Below 1100 pt the inspector overlays the canvas.

## Toolbar

From left to right:

| Item | Content |
|---|---|
| Title | File name. The subtitle is `4032 × 3024 · TIFF 16-bit`, with `From Lightroom ·` in front for a Lightroom document. |
| Center group, over the canvas | **Compare** segmented control: Split (`rectangle.lefthalf.inset.filled`) or Side by Side (`rectangle.split.2x1`). |
| | **Select Area** (`viewfinder`), a toggle. |
| | **Zoom** pop-up: Fit, 50%, 100%, 200%. The −/+ buttons are removed; ⌘−/⌘+ and pinch do the same job. |
| Trailing | Inspector toggle (`sidebar.trailing`). |
| | Primary button: **Export…** (`square.and.arrow.up`), or **Save to Lightroom** (`arrow.uturn.backward`) for a Lightroom document. |

Export lives in the toolbar so it stays put when steps are added and when the inspector is hidden.

## Canvas and compare modes

- **Split (default).** One full-size image with a draggable divider. A 2 pt white line has a 32 pt round handle. Left of the divider is Before; right of it is After.
  - When a new preview area is chosen, the divider snaps to the centre of the area once. After that it moves only when you drag it.
- **Side by Side (⌥2).** Two panes with a 4 pt gutter and locked zoom and pan, as the code does today. Use it for global judgement at Fit.
- **Flip.** Hold `\` in either mode to show Before full-frame. A "Before" tag appears while it is held. This replaces a third "Single" mode.
- **Tags.** Floating capsules sit 14 pt inside the image corners. They are dark glass, 11.5 pt semibold white text.
  - Before tag: `Before`.
  - After tag: `After` plus a state segment (see below).
  - Tags fade to 40% opacity after 2 s without pointer movement.
- **Canvas background.** Dynamic: #161618 in dark, #E6E6E9 in light. The photo has a soft shadow and a 0.5 pt edge.
- **Faces.**
  - Rounded boxes, drawn on After only. Enhanced faces get a 1.5 pt white-over-green stroke. Skipped faces get a dashed orange stroke.
  - The boxes show when **Show faces** is checked in the Faces step (on by default, as in the current app). Holding ⌥ shows them temporarily when it is off.

## Preview area and render states

Outside the area on the After side, the original is shown under a 46% black scrim, so it reads as "not rendered" rather than "broken". The area has a 1 pt white outline, a 2 pt dark halo, and 16 pt L-shaped corner brackets. A label tab above it reads `Preview area 2048 × 1536`, using the size in source pixels.

| State | Area | After tag |
|---|---|---|
| Ready | White outline and brackets, still | `After │ Denoise 60 · Sharpen 25`. This is the recipe the pixels show. |
| Settings changed / Updating | Orange dash march (7/5, 1 s cycle). The old pixels stay visible, desaturated to 40% at 95% brightness. | `After │ spinner Updating · 40%` |
| Dragging a slider | Same as Updating, but no job is sent | `After │ Release to preview` |
| Failed | Red outline | `After │ ⚠ Couldn't render · Try Again` (clickable) |
| Area off-screen | An edge arrow at the window edge | A pill reading `Preview area is off-screen · Show (⌘G) · Preview Here (⇧⌘E)` |

**Area rules:**
- Select with ⌘E (drag, click to recentre, drag inside to move), shift-drag at any time, or ⇧⌘E to preview the visible area.
- A drag larger than the cap turns orange beyond the limit and is clipped from where the drag started.
- Turning on Upscale halves the cap. A one-line toast explains the change.
- Double-click inside the area zooms to 100% on it.

**Commits.** A slider sends its job on mouse-up. Keyboard and scroll changes send after 400 ms idle.

## Inspector

From top to bottom:

1. **Lightroom card.** Shown only for a Lightroom document. Accent-tinted, with the text: **Opened from Lightroom** — "Save to Lightroom replaces the edit copy. Then click Finish in Lightroom."
   - When writing in place isn't possible, it reads: "Lightroom's copy can't be replaced. Export, then use Browse in Lightroom." The card then gives a Copy Path button.
2. **Preset pop-up** (full width), plus a `…` menu.
   - Presets: **Clean Up** (Denoise 50), **High ISO** (Denoise 70, Sharpen 25), **Portrait** (Denoise 50, Faces), **Rescue Blur** (Deblur 50, Sharpen 20), **Print 2×** (Denoise 40, Sharpen 20, Upscale).
   - Then: Save Preset…, and any user presets.
   - Changing any control shows `Custom · from High ISO`.
   - The `…` menu holds: Reset to Preset, Save as Preset…, Copy Recipe ⇧⌘C, Paste Recipe ⇧⌘V.
3. **Steps list (free-form, reorderable).** A `STEPS` header with an **+ Add Step** menu (Denoise, Deblur, Sharpen, Faces, Upscale 2×), then one rounded card per step. Steps run top to bottom, in the order the user sets. A caption under the list reads "Steps run top to bottom · drag to reorder".

| Step | Symbol | Control when on |
|---|---|---|
| Denoise | `aqi.medium` | Slider 0–100 with a value |
| Deblur | `scope` | `Motion │ Focus` segmented control, plus a slider |
| Sharpen | `triangle.lefthalf.filled` | Slider |
| Faces | `face.smiling` | **Show faces** checkbox (on by default) that draws the face boxes on the photo, with the result inline: `3 enhanced · 1 skipped ⓘ` or `None found`. The ⓘ popover explains why faces were skipped. |
| Upscale 2× | `arrow.up.left.and.arrow.down.right` | The output size is shown inline, e.g. `8064 × 6048` |

   - Each card has a tinted 26 pt icon tile, a name, and a trailing switch. A switched-off step stays in its place in the list, dimmed and collapsed, so it can be compared without losing its position or settings.
   - **Reorder:** drag a card. A grip (`line.3.horizontal` / six dots) and a `…` menu appear on hover. While dragging, the card lifts with a shadow and an accent insertion line shows where it will land. ⌥⌘↑ / ⌥⌘↓ move the selected step for keyboard users. Reordering is undoable (⌘Z) and re-renders the preview.
   - **`…` menu per card:** Move Up, Move Down, Duplicate, Solo (also ⌥-click the switch), Remove (⌫).
   - The same step can appear more than once (e.g. Denoise before and after Upscale), matching the server pipeline. The server's limits still apply: at most 4 steps and one Upscale. When the list is full, Add Step's items are disabled with the tooltip "Up to 4 steps"; a second Upscale is disabled with "Only one Upscale per recipe".
   - Presets load a whole ordered step list. Any add, remove, reorder or slider change shows `Custom · from <preset>`.
4. **Footer** (pinned):
   - `Output 4032 × 3024 · TIFF 16-bit`
   - `Server ● spark1`. The dot is green, amber or red. Click it to open a popover with the host, upload progress and models.
   - If placeholder models are active, one line says so: `Test models in use`.

Removed from the inspector: model IDs (moved to Settings › Server), the recipe label, the preview-area paragraph, step numbers, the up/down chevrons (replaced by drag and the `…` menu), the format control, and "server N ms" (now in the tooltip on the After tag).

## Export and Lightroom

- **Export… (⌘⇧S)** opens the save panel.
  - Its accessory view holds the format (Same as Original / JPEG / TIFF), JPEG quality, and 8/16-bit.
  - The default filename is `<name>-painite.<ext>`.
- **Export sheet.** Progress runs in a sheet: Preparing → Rendering 62% → Saving, with Cancel. When it finishes, the sheet shows Show in Finder. A failure says "Nothing was written."
- **Unseen-settings guard.** If the preview is Updating or Failed, Export first asks: "The preview doesn't show these settings yet." The options are **Update Preview** (default) and **Export Anyway**.
- **Save to Lightroom (⌘S)** writes the file atomically in place.
  - On success, a banner reads: "Saved. Click Finish in Lightroom."
  - Closing without saving shows "Lightroom is still waiting for this photo" with Save / Don't Save / Cancel.

## Welcome, empty and offline

- **Welcome window** (720×480):
  - The whole window is a drop target with a dashed 16 pt-radius frame.
  - Inside the frame: the logo, **Drop a photo here**, "JPEG or TIFF, up to 16-bit. Or open one from Lightroom with Edit in.", and **Open Photo…**.
  - Below the frame: a **Recent** row of 4 thumbnails.
  - Server status is hidden when the server is healthy.
- **No server configured.** The drop frame becomes a solid panel with the text **Connect to your server** — "Paste the setup code from your PAINITE server to start." — and a **Connect to Server…** button.
  - Settings takes one setup-code field. The URL, token and CA fingerprint go under Advanced.
- **Offline in a document.** A canvas banner reads `Can't reach spark1 · Retrying in 5 s · Retry Now · Settings…`.
  - The app retries automatically at 2/5/15/30 s, and again on focus and on network change.
  - Export is disabled, with the reason in its tooltip.
- **Opening a photo.** The image appears at once. The inspector stays usable while the upload runs, shown as a thin progress line under the toolbar.

## Visual spec

| Token | Light | Dark |
|---|---|---|
| Accent (tint, switches, sliders) | #E0582F | #F58A3D |
| Accent fill (primary button, white text) | #EA6A33→#D4502A gradient | same |
| Accent text | #C94A22 | #F7A062 |
| Canvas | #E6E6E9 | #161618 |
| Inspector | sidebar material ≈ rgba(249,249,251,.84) | ≈ rgba(38,38,41,.80) |
| Group card | white 75% | white 4.5% |
| Separators | black 7% | white 7.5% |
| Scrim outside the area | black 46% | black 46% |
| Updating stroke | #F58A3D | #F58A3D |

- **Type.** SF Pro. 13 pt body and row names (medium). 11.5 pt captions and tags (semibold). 13 pt semibold title with an 11 pt subtitle. All numbers use `monospacedDigit()`.
- **Spacing.**
  - 4 pt grid. Inspector padding is 14 pt and the group gap is 14 pt.
  - Rows are 44 pt high. Slider rows are indented 36 pt to align with the names.
  - Toolbar capsules are 34 pt high with a 10 pt gap.
  - Corner radii: canvas tags 12, popups 10, cards 13, inspector 20.
- **Motion.**
  - The dash march runs in a 1 s cycle.
  - When a render lands, the area flashes white for 250 ms.
  - Tags fade after 2 s.
  - With Reduce Motion on, the dash march becomes a static dash and the flash becomes an instant change.

## Keyboard

| Key | Action |
|---|---|
| ⌥1 / ⌥2 | Split / Side by Side |
| `\` (hold) | Show Before |
| ⌘E | Select Area |
| ⇧⌘E | Preview visible area |
| ⌘G | Go to the preview area |
| ⌘0 / ⌘1 / ⌘+ / ⌘− | Fit / 100% / zoom in / zoom out |
| ⌘R | Retry or re-render |
| ⌘[ / ⌘] | Previous / next operation |
| ← / → | Adjust strength ±1; with ⇧, ±10 |
| ⌘Z | Undo any recipe change |
| ⇧⌘C / ⇧⌘V | Copy / Paste Recipe |
| ⌥⌘I | Toggle the inspector |
| ⌘S / ⇧⌘S | Save to Lightroom / Export… |
| Esc | Cancel selection or render |

## Decisions

| Decision | Chosen | Why | Rejected |
|---|---|---|---|
| Default compare | **Split**, with Side by Side as an option | Noise and sharpness are judged on the same pixels, at full width | Side by Side as the default: each pane is about 490 pt at Fit |
| Third mode | None. Hold `\` to flip. | Split dragged to an edge already gives a single view | A "Single" segment |
| Pipeline | **Free-form reorderable step cards** (Yuan's call) | Order matters to the result and experts need it; hover-only grips and `…` menus keep it quiet | A fixed-order list of switches |
| Default recipe | **Last used recipe** (Clean Up on first run) | Predictable, and needs no server work | "Auto": there is no estimator yet, and it hides what runs. Revisit when the server can estimate noise. |
| Presets | 5 built-in, plus Save Preset; edits show `Custom · from X` | Covers the beginner path the reviewers wanted from Auto | No presets |
| Names | **Denoise, Deblur (Motion/Focus), Sharpen, Faces, Upscale 2×** | Verbs say what the model does and match the CLI | Noise/Detail/Enlarge (Lightroom nouns suggest a classic slider), "Unblur", "Shake" |
| Pane labels | Before / After | Universal | Original / Enhanced |
| Stale pixels | Desaturated old result, orange dash march, % in tag | Obvious at a glance, calm, and keeps context | Yellow pulse; a grey "Out of date" in a header |
| Export location | Toolbar primary button | Fixed place, visible with the inspector hidden | The bottom of the inspector |
| Export format | The save panel's accessory view | This is where Mac users expect it | A segmented control in the inspector |
| Server status | A footer dot plus a banner when it fails | Quiet when it works, loud when it doesn't | A dot in the toolbar; "server N ms" |
| Preview visible area | Menu and ⇧⌘E only | Fewer toolbar buttons | A separate toolbar button |

## Build order

1. **Window shell** (biggest visible win): `.fullSizeContentView`, a unified toolbar, the `.inspector`, the accent colour asset, dynamic canvas colours, and removing the header bar.
2. **Canvas states:** the Before/After tags, the After state segment, the area label and brackets, the scrim, desaturated stale pixels, the dash march, and commit on release.
3. **Inspector rework:** reorderable step cards with drag, hover grip and `…` menu, the preset pop-up, Custom tracking, last-used recipe, undo, and the cuts listed above.
4. **Split mode** with the divider, plus hold `\`.
5. **Export guard,** format moved to the save panel accessory, and the Lightroom card and close sheet.
6. **Welcome drop zone,** setup-code onboarding, the offline banner and auto-retry.
7. **Polish:** the off-screen pointer (⌘G), cap feedback, face hover, solo, and the Copy/Paste Recipe and Keyboard Shortcuts sheet.
