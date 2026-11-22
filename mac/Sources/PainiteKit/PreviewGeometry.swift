// Which part of the photo to ask the server to preview.

import CoreGraphics
import Foundation

public enum PreviewGeometry {
    /// Largest preview side in source pixels. Keeps a preview to a few tiles so it
    /// returns in seconds; when zoomed out further, the preview covers the middle
    /// of the view and the rest of the "after" pane shows the original, dimmed.
    public static func maxSide(for operation: Operation) -> Int {
        maxSide(scale: operation.scale)
    }

    /// Same, for a pipeline whose output is `scale` times the input on each axis.
    public static func maxSide(scale: Int) -> Int {
        scale >= 2 ? 1024 : 2048
    }

    /// The region to request for a view showing `visible` (displayed source pixels)
    /// of an image of `imageSize`. Returns nil when nothing of the image is visible.
    public static func requestRegion(visible: CGRect, imageSize: CGSize, operation: Operation) -> Region? {
        requestRegion(visible: visible, imageSize: imageSize, scale: operation.scale)
    }

    public static func requestRegion(visible: CGRect, imageSize: CGSize, scale: Int) -> Region? {
        let bounds = CGRect(origin: .zero, size: imageSize)
        var r = visible.intersection(bounds)
        guard !r.isNull, r.width >= 1, r.height >= 1 else { return nil }
        let cap = CGFloat(maxSide(scale: scale))
        if r.width > cap {
            r.origin.x = r.midX - cap / 2
            r.size.width = cap
        }
        if r.height > cap {
            r.origin.y = r.midY - cap / 2
            r.size.height = cap
        }
        let x0 = max(0, Int(r.minX.rounded(.down)))
        let y0 = max(0, Int(r.minY.rounded(.down)))
        let x1 = min(Int(imageSize.width), Int(r.maxX.rounded(.up)))
        let y1 = min(Int(imageSize.height), Int(r.maxY.rounded(.up)))
        guard x1 > x0, y1 > y0 else { return nil }
        return Region(x: x0, y: y0, w: x1 - x0, h: y1 - y0)
    }

    /// Whether an already rendered region still covers what a new request would ask for.
    public static func covers(_ rendered: Region, _ wanted: Region) -> Bool {
        rendered.x <= wanted.x && rendered.y <= wanted.y &&
            rendered.x + rendered.w >= wanted.x + wanted.w && rendered.y + rendered.h >= wanted.y + wanted.h
    }
}

/// Orientation-aware mapping and tile snapping, mirroring
/// server/painite_server/imaging/geometry.py, so the app can show the exact
/// box the server will render before it answers.
public enum TileGrid {
    /// Grows a region in displayed pixels outward to whole tiles of the full
    /// image's grid, which is anchored at the stored (unrotated) pixel origin.
    /// `storedWidth`/`storedHeight` are the file's pixel dimensions before orientation.
    public static func snap(_ r: Region, storedWidth w: Int, storedHeight h: Int, orientation o: Int, tile: Int) -> Region {
        guard tile > 0 else { return r }
        let o = (1...8).contains(o) ? o : 1
        let stored = map(r, displayToStored, w, h, o)
        let x0 = (stored.x / tile) * tile
        let y0 = (stored.y / tile) * tile
        let x1 = min(((stored.x + stored.w + tile - 1) / tile) * tile, w)
        let y1 = min(((stored.y + stored.h + tile - 1) / tile) * tile, h)
        let snapped = Region(x: x0, y: y0, w: x1 - x0, h: y1 - y0)
        return map(snapped, storedToDisplay, w, h, o)
    }

    static func displayToStored(_ dx: Int, _ dy: Int, _ w: Int, _ h: Int, _ o: Int) -> (Int, Int) {
        switch o {
        case 2: (w - 1 - dx, dy)
        case 3: (w - 1 - dx, h - 1 - dy)
        case 4: (dx, h - 1 - dy)
        case 5: (dy, dx)
        case 6: (dy, h - 1 - dx)
        case 7: (w - 1 - dy, h - 1 - dx)
        case 8: (w - 1 - dy, dx)
        default: (dx, dy)
        }
    }

    static func storedToDisplay(_ sx: Int, _ sy: Int, _ w: Int, _ h: Int, _ o: Int) -> (Int, Int) {
        switch o {
        case 2: (w - 1 - sx, sy)
        case 3: (w - 1 - sx, h - 1 - sy)
        case 4: (sx, h - 1 - sy)
        case 5: (sy, sx)
        case 6: (h - 1 - sy, sx)
        case 7: (h - 1 - sy, w - 1 - sx)
        case 8: (sy, w - 1 - sx)
        default: (sx, sy)
        }
    }

    static func map(_ r: Region, _ fn: (Int, Int, Int, Int, Int) -> (Int, Int), _ w: Int, _ h: Int, _ o: Int) -> Region {
        let (x0, y0) = fn(r.x, r.y, w, h, o)
        let (x1, y1) = fn(r.x + r.w - 1, r.y + r.h - 1, w, h, o)
        return Region(x: min(x0, x1), y: min(y0, y1), w: abs(x1 - x0) + 1, h: abs(y1 - y0) + 1)
    }
}
