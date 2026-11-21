// Which part of the photo to ask the server to preview.

import CoreGraphics
import Foundation

public enum PreviewGeometry {
    /// Largest preview side in source pixels. Keeps a preview to a few tiles so it
    /// returns in seconds; when zoomed out further, the preview covers the middle
    /// of the view and the rest of the "after" pane shows the original, dimmed.
    public static func maxSide(for operation: Operation) -> Int {
        operation.scale == 2 ? 1024 : 2048
    }

    /// The region to request for a view showing `visible` (displayed source pixels)
    /// of an image of `imageSize`. Returns nil when nothing of the image is visible.
    public static func requestRegion(visible: CGRect, imageSize: CGSize, operation: Operation) -> Region? {
        let bounds = CGRect(origin: .zero, size: imageSize)
        var r = visible.intersection(bounds)
        guard !r.isNull, r.width >= 1, r.height >= 1 else { return nil }
        let cap = CGFloat(maxSide(for: operation))
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
