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

    /// Tiles per side of one preview: 2 (4 tiles) on a slow server (Apple GPU or
    /// CPU) or for an upscale, else 4. Every tile costs the same, so this bounds
    /// a preview's time.
    public static func maxTiles(scale: Int, slowServer: Bool) -> Int {
        slowServer || scale >= 2 ? 2 : 4
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
    /// Grows a region in displayed pixels outward to the full image's tile grid,
    /// which is anchored at the stored (unrotated) pixel origin, inset by the
    /// server's blend `band`: edges land on k*tile + band (left/top) or
    /// k*tile - band (right/bottom), or on the image edge, so the preview needs
    /// only the tiles under it. With `maxTiles`, keeps at most that many tiles per
    /// side, centered on the request. `storedWidth`/`storedHeight` are the file's
    /// pixel dimensions before orientation.
    public static func snap(_ r: Region, storedWidth w: Int, storedHeight h: Int, orientation o: Int, tile: Int,
                            band: Int = 0, maxTiles: Int? = nil) -> Region {
        guard tile > 0 else { return r }
        let o = (1...8).contains(o) ? o : 1
        let stored = map(r, displayToStored, w, h, o)
        let (x0, x1) = snapAxis(stored.x, stored.x + stored.w, tile: tile, size: w, band: band, maxTiles: maxTiles)
        let (y0, y1) = snapAxis(stored.y, stored.y + stored.h, tile: tile, size: h, band: band, maxTiles: maxTiles)
        let snapped = Region(x: x0, y: y0, w: x1 - x0, h: y1 - y0)
        return map(snapped, storedToDisplay, w, h, o)
    }

    /// Mirrors server/painite_server/imaging/geometry.py `_snap_axis`, plus the tile cap.
    static func snapAxis(_ lo: Int, _ hi: Int, tile: Int, size: Int, band: Int, maxTiles: Int?) -> (Int, Int) {
        let count = (size + tile - 1) / tile
        var k0 = max(floorDiv(lo - band, tile), 0)
        var k1 = min((hi + band + tile - 1) / tile, count)
        if let m = maxTiles, m > 0, k1 - k0 > m {
            let center = Double(lo + hi) / 2 / Double(tile)
            k0 = min(max(Int((center - Double(m) / 2).rounded()), 0), count - m)
            k1 = k0 + m
        }
        let a = k0 == 0 ? 0 : k0 * tile + band
        let b = k1 * tile >= size ? size : k1 * tile - band
        return (a, b)
    }

    static func floorDiv(_ a: Int, _ b: Int) -> Int {
        a >= 0 ? a / b : -((-a + b - 1) / b)
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
