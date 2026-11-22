import AppKit
import PainiteKit
import QuartzCore
import SwiftUI

/// Before and after, always at the same zoom and position.
///
/// Two modes: Split (one image with a draggable divider: Before on the left,
/// After on the right) and Side by Side (two panes). Both panes show the same
/// displayed-source-pixel rectangle. The after pane shows the original under a
/// scrim, with the server's preview on top of the region it covers. Upscaled
/// previews are drawn into the same source region, so both sides keep the same
/// framing (PRD). Layers are color matched by Core Animation.
final class CompareView: NSView {
    enum Mode: String, CaseIterable {
        case split, sideBySide
    }

    /// Visible rectangle in displayed source pixels, and zoom in percent (100 = one image pixel per screen pixel).
    var onViewportChange: ((CGRect, Double) -> Void)?
    /// A new preview area picked by dragging, in displayed source pixels.
    var onSelectArea: ((CGRect) -> Void)?
    /// Called when Esc ends Select Area without a selection.
    var onCancelSelect: (() -> Void)?

    /// When on, a drag picks the preview area instead of panning. Shift-drag
    /// does the same at any time.
    var selectMode = false {
        didSet {
            guard oldValue != selectMode else { return }
            window?.invalidateCursorRects(for: self)
        }
    }

    var mode: Mode = .split {
        didSet {
            guard oldValue != mode else { return }
            needsLayout = true
            window?.invalidateCursorRects(for: self)
        }
    }

    private var original: CGImage?
    private var previewImage: CGImage?
    private var previewRegion: CGRect = .zero
    private var imageSize: CGSize = .zero
    private var box: CGRect?
    private var boxLabel = ""
    /// The picked area (before tile snapping), which a drag inside the box moves.
    private var selection: CGRect?
    private var drag: Drag?
    /// Split divider position as a fraction of the view's width.
    private var dividerFraction: CGFloat = 0.5
    /// True while `\` is held: show Before everywhere.
    private var peekingBefore = false {
        didSet { if oldValue != peekingBefore { applyTransform(notify: false) } }
    }

    private enum Drag {
        case pan
        case divider
        /// Drawing a new area from this image point.
        case draw(from: CGPoint, to: CGPoint)
        /// Moving the selection by the distance from this image point.
        case move(from: CGPoint, to: CGPoint, original: CGRect)
    }

    /// With no steps the enhanced side is just the original, shown without the scrim.
    var enhancedIsOriginal = false {
        didSet { if oldValue != enhancedIsOriginal { applyTransform(notify: false) } }
    }
    private var boxState: BoxState?

    enum BoxState {
        /// Settings changed and the preview is waiting to be sent or rendering: marching orange dashes.
        case pending, rendering
        /// The preview inside the box is current.
        case ready
        case failed
    }

    /// Points per image pixel.
    private var scale: CGFloat = 1
    /// Image pixel at the center of each pane.
    private var center: CGPoint = .zero
    private var fitMode = true

    private let leftPane = CALayer()
    private let rightPane = CALayer()
    /// In Split mode, clips the after pane to the right of the divider.
    private let rightMask = CALayer()
    private let gutter = CALayer()
    private let dividerLine = CALayer()
    private let dividerHandle = CALayer()
    private let leftImage = CALayer()
    private let rightBase = CALayer()
    private let rightPreview = CALayer()
    /// Box, label and drag band, above the panes: one per pane (only the first in Split mode).
    private let overlays = [CALayer(), CALayer()]
    private let outlines = [CAShapeLayer(), CAShapeLayer()]
    private let brackets = [CAShapeLayer(), CAShapeLayer()]
    private let labels = [CATextLayer(), CATextLayer()]
    private let bands = [CALayer(), CALayer()]
    /// Face outlines (from a face step), on the after side only.
    private var faces: [FaceInfo] = []
    private var faceLayers: [CAShapeLayer] = []

    static let accent = NSColor(srgbRed: 0xF5 / 255, green: 0x8A / 255, blue: 0x3D / 255, alpha: 1)

    override init(frame: NSRect) {
        super.init(frame: frame)
        let root = CALayer()
        root.isGeometryFlipped = true
        layer = root
        wantsLayer = true
        for pane in [leftPane, rightPane] {
            pane.masksToBounds = true
            root.addSublayer(pane)
        }
        rightMask.backgroundColor = NSColor.black.cgColor
        root.addSublayer(gutter)
        for l in [leftImage, rightBase, rightPreview] {
            l.anchorPoint = .zero
            l.minificationFilter = .trilinear
            l.contentsGravity = .resize
        }
        leftPane.addSublayer(leftImage)
        rightPane.addSublayer(rightBase)
        rightPane.addSublayer(rightPreview)

        for i in 0..<2 {
            let overlay = overlays[i]
            overlay.masksToBounds = true
            root.addSublayer(overlay)

            let outline = outlines[i]
            outline.fillColor = nil
            outline.lineWidth = 1.5
            outline.isHidden = true
            outline.shadowColor = NSColor.black.cgColor
            outline.shadowOpacity = 0.5
            outline.shadowRadius = 1
            outline.shadowOffset = .zero
            overlay.addSublayer(outline)

            let br = brackets[i]
            br.fillColor = nil
            br.lineWidth = 3
            br.lineCap = .square
            br.isHidden = true
            overlay.addSublayer(br)

            let label = labels[i]
            label.fontSize = 11
            label.font = NSFont.systemFont(ofSize: 11, weight: .semibold)
            label.foregroundColor = NSColor.white.cgColor
            label.backgroundColor = NSColor(white: 0, alpha: 0.55).cgColor
            label.cornerRadius = 9
            label.alignmentMode = .center
            label.isHidden = true
            overlay.addSublayer(label)

            let band = bands[i]
            band.borderColor = CompareView.accent.cgColor
            band.borderWidth = 1.5
            band.backgroundColor = CompareView.accent.withAlphaComponent(0.12).cgColor
            band.isHidden = true
            overlay.addSublayer(band)
        }

        dividerLine.backgroundColor = NSColor.white.cgColor
        dividerLine.shadowColor = NSColor.black.cgColor
        dividerLine.shadowOpacity = 0.35
        dividerLine.shadowRadius = 2
        dividerLine.shadowOffset = .zero
        root.addSublayer(dividerLine)
        dividerHandle.backgroundColor = NSColor.white.cgColor
        dividerHandle.cornerRadius = 16
        dividerHandle.shadowColor = NSColor.black.cgColor
        dividerHandle.shadowOpacity = 0.3
        dividerHandle.shadowRadius = 4
        dividerHandle.shadowOffset = CGSize(width: 0, height: 1)
        dividerHandle.contentsGravity = .center
        root.addSublayer(dividerHandle)
        updateColors()
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }
    override var wantsUpdateLayer: Bool { true }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        updateColors()
    }

    private func updateColors() {
        let dark = effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        layer?.backgroundColor = (dark ? NSColor(srgbRed: 0.11, green: 0.11, blue: 0.118, alpha: 1)
                                       : NSColor(srgbRed: 0.91, green: 0.91, blue: 0.918, alpha: 1)).cgColor
        gutter.backgroundColor = (dark ? NSColor(white: 0.07, alpha: 1) : NSColor(white: 0.8, alpha: 1)).cgColor
        let config = NSImage.SymbolConfiguration(pointSize: 12, weight: .semibold)
        if let img = NSImage(systemSymbolName: "chevron.left.chevron.right", accessibilityDescription: "Divider")?
            .withSymbolConfiguration(config) {
            let tinted = NSImage(size: img.size, flipped: false) { rect in
                img.draw(in: rect)
                NSColor(white: 0.15, alpha: 1).set()
                rect.fill(using: .sourceAtop)
                return true
            }
            dividerHandle.contents = tinted
            dividerHandle.contentsScale = window?.backingScaleFactor ?? 2
        }
    }

    // MARK: Content

    func setOriginal(_ image: CGImage?, size: CGSize) {
        guard image !== original else { return }
        original = image
        imageSize = size
        leftImage.contents = image
        rightBase.contents = image
        fitMode = true
        needsLayout = true
    }

    /// Outline faces: enhanced ones in green, faces skipped for their size dashed in orange.
    func setFaces(_ list: [FaceInfo]) {
        guard list != faces else { return }
        faces = list
        faceLayers.forEach { $0.removeFromSuperlayer() }
        faceLayers = []
        for face in list {
            let l = CAShapeLayer()
            l.fillColor = nil
            l.lineWidth = 1.5
            l.shadowColor = NSColor.black.cgColor
            l.shadowOpacity = 0.5
            l.shadowRadius = 1
            l.shadowOffset = .zero
            if face.status == .enhanced {
                l.strokeColor = NSColor.systemGreen.cgColor
            } else {
                l.strokeColor = NSColor.systemOrange.cgColor
                l.lineDashPattern = [5, 4]
            }
            rightPane.addSublayer(l)
            faceLayers.append(l)
        }
        applyTransform(notify: false)
    }

    func setPreview(_ image: CGImage?, region: Region?) {
        let rect = region.map { CGRect(x: $0.x, y: $0.y, width: $0.w, height: $0.h) } ?? .zero
        guard image !== previewImage || rect != previewRegion else { return }
        previewImage = image
        previewRegion = rect
        rightPreview.contents = image
        applyTransform(notify: false)
    }

    /// The region the preview covers or is about to cover, in displayed source pixels,
    /// with its label (e.g. "Preview area  2048 × 1536"). Ready: a quiet white
    /// outline with corner brackets. Pending or rendering: marching orange dashes,
    /// and the old result inside is dimmed.
    func setPreviewBox(_ region: Region?, state: BoxState, label: String) {
        box = region.map { CGRect(x: $0.x, y: $0.y, width: $0.w, height: $0.h) }
        if label != boxLabel {
            boxLabel = label
            for l in labels { l.string = label }
        }
        if state != boxState {
            withoutAnimation {
                for (o, br) in zip(outlines, brackets) {
                    o.removeAnimation(forKey: "march")
                    switch state {
                    case .pending, .rendering:
                        o.strokeColor = CompareView.accent.cgColor
                        o.lineDashPattern = [6, 4]
                        let march = CABasicAnimation(keyPath: "lineDashPhase")
                        march.fromValue = 0
                        march.toValue = -10
                        march.duration = 0.5
                        march.repeatCount = .infinity
                        o.add(march, forKey: "march")
                        br.strokeColor = CompareView.accent.cgColor
                    case .ready:
                        o.strokeColor = NSColor(white: 1, alpha: 0.85).cgColor
                        o.lineDashPattern = nil
                        br.strokeColor = NSColor.white.cgColor
                    case .failed:
                        o.strokeColor = NSColor.systemOrange.cgColor
                        o.lineDashPattern = nil
                        br.strokeColor = NSColor.systemOrange.cgColor
                    }
                }
            }
        }
        boxState = state
        applyTransform(notify: false)
    }

    /// The picked area, so a drag inside the box can move it.
    func setSelection(_ rect: CGRect?) {
        selection = rect
    }

    /// Puts the Split divider through the middle of a newly chosen area (once; after
    /// that it only moves when dragged).
    func centerDivider(on rect: CGRect) {
        guard mode == .split, bounds.width > 0, scale > 0 else { return }
        let o = origin(in: paneSize)
        let x = o.x + rect.midX * scale
        dividerFraction = min(max(x / bounds.width, 0.05), 0.95)
        layoutPanes()
        applyTransform(notify: false)
    }

    // MARK: Geometry

    private var backingScale: CGFloat { window?.backingScaleFactor ?? 2 }
    private var paneSize: CGSize { leftPane.bounds.size }
    private var dividerX: CGFloat { (bounds.width * dividerFraction).rounded() }

    var zoomPercent: Double { Double(scale * backingScale * 100) }

    private var fitScale: CGFloat {
        guard imageSize.width > 0, imageSize.height > 0, paneSize.width > 0 else { return 1 }
        // Leave a margin around the photo at Fit.
        let w = max(paneSize.width - 32, 1)
        let h = max(paneSize.height - 32, 1)
        return min(w / imageSize.width, h / imageSize.height)
    }

    override func layout() {
        super.layout()
        layoutPanes()
        if fitMode { fit() } else { applyTransform() }
    }

    private func layoutPanes() {
        withoutAnimation {
            let w = bounds.width
            let h = bounds.height
            switch mode {
            case .split:
                leftPane.frame = bounds
                rightPane.frame = bounds
                let x = dividerX
                rightMask.frame = CGRect(x: x, y: 0, width: max(0, w - x), height: h)
                rightPane.mask = rightMask
                gutter.isHidden = true
                overlays[0].frame = bounds
                overlays[1].isHidden = true
                dividerLine.isHidden = false
                dividerHandle.isHidden = false
                dividerLine.frame = CGRect(x: x - 1, y: 0, width: 2, height: h)
                dividerHandle.frame = CGRect(x: x - 16, y: h / 2 - 16, width: 32, height: 32)
            case .sideBySide:
                let g: CGFloat = 4
                let half = floor((w - g) / 2)
                rightPane.mask = nil
                leftPane.frame = CGRect(x: 0, y: 0, width: half, height: h)
                gutter.isHidden = false
                gutter.frame = CGRect(x: half, y: 0, width: g, height: h)
                rightPane.frame = CGRect(x: half + g, y: 0, width: w - half - g, height: h)
                overlays[0].frame = leftPane.frame
                overlays[1].frame = rightPane.frame
                overlays[1].isHidden = false
                dividerLine.isHidden = true
                dividerHandle.isHidden = true
            }
        }
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        for l in [leftImage, rightBase, rightPreview] { l.contentsScale = 1 }
        updateColors()
        applyTransform()
    }

    private func origin(in pane: CGSize) -> CGPoint {
        let bs = backingScale
        // Pixel-align so 100% shows each image pixel on exactly one screen pixel.
        let x = pane.width / 2 - center.x * scale
        let y = pane.height / 2 - center.y * scale
        return CGPoint(x: (x * bs).rounded() / bs, y: (y * bs).rounded() / bs)
    }

    private func applyTransform(notify: Bool = true) {
        guard imageSize.width > 0 else { return }
        let o = origin(in: paneSize)
        let pixelZoom = scale * backingScale
        let stale = boxState != .ready && boxState != nil
        withoutAnimation {
            for l in [leftImage, rightBase] {
                l.bounds = CGRect(origin: .zero, size: imageSize)
                l.position = o
                l.transform = CATransform3DMakeScale(scale, scale, 1)
                l.magnificationFilter = pixelZoom > 1.01 ? .nearest : .linear
            }
            rightPane.isHidden = peekingBefore && mode == .split
            rightBase.opacity = Float(enhancedIsOriginal || peekingBefore ? 1 : 0.5)
            if let previewImage, previewRegion.width > 0, !peekingBefore {
                let pw = CGFloat(previewImage.width)
                let ph = CGFloat(previewImage.height)
                let s = scale * previewRegion.width / pw
                rightPreview.isHidden = false
                rightPreview.opacity = stale ? 0.55 : 1
                rightPreview.bounds = CGRect(x: 0, y: 0, width: pw, height: ph)
                rightPreview.position = CGPoint(x: o.x + previewRegion.minX * scale, y: o.y + previewRegion.minY * scale)
                rightPreview.transform = CATransform3DMakeScale(s, s, 1)
                rightPreview.magnificationFilter = s * backingScale > 1.01 ? .nearest : .linear
            } else {
                rightPreview.isHidden = true
            }
            for i in 0..<2 {
                let outline = outlines[i], br = brackets[i], label = labels[i]
                if let box, box.width > 0, !peekingBefore {
                    let frame = paneRect(box, origin: o)
                    outline.isHidden = false
                    outline.frame = frame
                    outline.path = CGPath(rect: CGRect(origin: .zero, size: frame.size), transform: nil)
                    br.isHidden = false
                    br.frame = frame
                    br.path = bracketPath(frame.size)
                    let text = boxLabel as NSString
                    let tw = ceil(text.size(withAttributes: [.font: NSFont.systemFont(ofSize: 11, weight: .semibold)]).width) + 20
                    label.isHidden = boxLabel.isEmpty || frame.width < 60
                    let above = frame.minY - 26
                    label.frame = CGRect(x: frame.minX, y: above >= 4 ? above : frame.minY + 6, width: tw, height: 20)
                    label.contentsScale = backingScale
                } else {
                    outline.isHidden = true
                    br.isHidden = true
                    label.isHidden = true
                }
            }
            for (i, face) in faces.enumerated() where i < faceLayers.count {
                let r = CGRect(x: face.box.x, y: face.box.y, width: face.box.w, height: face.box.h)
                let frame = paneRect(r, origin: o)
                let l = faceLayers[i]
                l.isHidden = peekingBefore
                l.frame = frame
                let radius = min(6, frame.width / 4, frame.height / 4)
                l.path = CGPath(roundedRect: CGRect(origin: .zero, size: frame.size),
                                cornerWidth: radius, cornerHeight: radius, transform: nil)
            }
            if let band = bandRect, band.width > 0 {
                for l in bands {
                    l.isHidden = false
                    l.frame = paneRect(band, origin: o)
                }
            } else {
                for l in bands { l.isHidden = true }
            }
        }
        if notify { reportViewport() }
    }

    /// Four 10 pt corner brackets just outside the box.
    private func bracketPath(_ size: CGSize) -> CGPath {
        let p = CGMutablePath()
        let a: CGFloat = min(10, size.width / 3, size.height / 3)
        let w = size.width, h = size.height
        p.move(to: CGPoint(x: 0, y: a)); p.addLine(to: .zero); p.addLine(to: CGPoint(x: a, y: 0))
        p.move(to: CGPoint(x: w - a, y: 0)); p.addLine(to: CGPoint(x: w, y: 0)); p.addLine(to: CGPoint(x: w, y: a))
        p.move(to: CGPoint(x: w, y: h - a)); p.addLine(to: CGPoint(x: w, y: h)); p.addLine(to: CGPoint(x: w - a, y: h))
        p.move(to: CGPoint(x: a, y: h)); p.addLine(to: CGPoint(x: 0, y: h)); p.addLine(to: CGPoint(x: 0, y: h - a))
        return p
    }

    private func paneRect(_ r: CGRect, origin o: CGPoint) -> CGRect {
        CGRect(x: o.x + r.minX * scale, y: o.y + r.minY * scale, width: r.width * scale, height: r.height * scale)
    }

    /// The area being drawn or moved, clipped to the image.
    private var bandRect: CGRect? {
        let bounds = CGRect(origin: .zero, size: imageSize)
        switch drag {
        case .draw(let a, let b):
            return CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(a.x - b.x), height: abs(a.y - b.y))
                .intersection(bounds)
        case .move(let a, let b, let r):
            var moved = r.offsetBy(dx: b.x - a.x, dy: b.y - a.y)
            moved.origin.x = min(max(moved.minX, 0), max(0, imageSize.width - moved.width))
            moved.origin.y = min(max(moved.minY, 0), max(0, imageSize.height - moved.height))
            return moved
        default:
            return nil
        }
    }

    /// The image point under an event, in displayed source pixels.
    private func imagePoint(_ event: NSEvent) -> CGPoint {
        let p = paneLocation(event)
        let o = origin(in: paneSize)
        return CGPoint(x: (p.x - o.x) / scale, y: (p.y - o.y) / scale)
    }

    private func reportViewport() {
        guard scale > 0, paneSize.width > 0 else { return }
        let o = origin(in: paneSize)
        let visible = CGRect(x: -o.x / scale, y: -o.y / scale, width: paneSize.width / scale, height: paneSize.height / scale)
        onViewportChange?(visible, zoomPercent)
    }

    private func clampCenter() {
        center.x = min(max(center.x, 0), imageSize.width)
        center.y = min(max(center.y, 0), imageSize.height)
    }

    private func withoutAnimation(_ body: () -> Void) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        body()
        CATransaction.commit()
    }

    // MARK: Zoom commands

    func fit() {
        fitMode = true
        scale = fitScale
        center = CGPoint(x: imageSize.width / 2, y: imageSize.height / 2)
        applyTransform()
    }

    func actualSize() { setScale(1 / backingScale, anchor: nil) }
    func zoomIn() { setScale(scale * 2, anchor: nil) }
    func zoomOut() { setScale(scale / 2, anchor: nil) }
    /// Zoom to a percentage (100 = one image pixel per screen pixel).
    func zoom(toPercent p: Double) { setScale(CGFloat(p / 100) / backingScale, anchor: nil) }

    /// Zooms keeping the image point under `anchor` (pane coordinates) fixed.
    private func setScale(_ newScale: CGFloat, anchor: CGPoint?) {
        guard imageSize.width > 0 else { return }
        let minScale = min(fitScale, 0.05 / backingScale)
        let maxScale = 32 / backingScale
        let s = min(max(newScale, minScale), maxScale)
        let pane = paneSize
        let a = anchor ?? CGPoint(x: pane.width / 2, y: pane.height / 2)
        let o = origin(in: pane)
        let imagePoint = CGPoint(x: (a.x - o.x) / scale, y: (a.y - o.y) / scale)
        scale = s
        center = CGPoint(x: (pane.width / 2 - (a.x - imagePoint.x * s)) / s,
                         y: (pane.height / 2 - (a.y - imagePoint.y * s)) / s)
        fitMode = abs(s - fitScale) < 0.0001
        clampCenter()
        applyTransform()
    }

    private func pan(dx: CGFloat, dy: CGFloat) {
        guard scale > 0 else { return }
        center.x -= dx / scale
        center.y -= dy / scale
        fitMode = false
        clampCenter()
        applyTransform()
    }

    // MARK: Events

    /// The event's location in the coordinates of whichever pane it falls in.
    private func paneLocation(_ event: NSEvent) -> CGPoint {
        let p = convert(event.locationInWindow, from: nil)
        let pane = p.x >= rightPane.frame.minX ? rightPane.frame : leftPane.frame
        return CGPoint(x: p.x - pane.minX, y: p.y - pane.minY)
    }

    private func nearDivider(_ event: NSEvent) -> Bool {
        guard mode == .split else { return false }
        let p = convert(event.locationInWindow, from: nil)
        if abs(p.x - dividerX) <= 6 { return true }
        return dividerHandle.frame.insetBy(dx: -4, dy: -4).contains(p)
    }

    override func scrollWheel(with event: NSEvent) {
        if event.modifierFlags.contains(.command) || !event.hasPreciseScrollingDeltas {
            let factor = exp(event.scrollingDeltaY * (event.hasPreciseScrollingDeltas ? 0.01 : 0.1))
            setScale(scale * factor, anchor: paneLocation(event))
        } else {
            pan(dx: event.scrollingDeltaX, dy: event.scrollingDeltaY)
        }
    }

    override func magnify(with event: NSEvent) {
        setScale(scale * (1 + event.magnification), anchor: paneLocation(event))
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        guard imageSize.width > 0, scale > 0 else { return }
        if selectMode || event.modifierFlags.contains(.shift) {
            let p = imagePoint(event)
            if let sel = selection, let box, box.contains(p) {
                drag = .move(from: p, to: p, original: sel)
            } else {
                drag = .draw(from: p, to: p)
            }
            return
        }
        if nearDivider(event) {
            drag = .divider
            return
        }
        drag = .pan
        if event.clickCount == 2 {
            if let box, box.contains(imagePoint(event)), fitMode {
                // Double-click inside the preview area: look at it at 100%.
                center = CGPoint(x: box.midX, y: box.midY)
                setScale(1 / backingScale, anchor: nil)
            } else if fitMode {
                setScale(1 / backingScale, anchor: paneLocation(event))
            } else {
                fit()
            }
        }
    }

    override func mouseDragged(with event: NSEvent) {
        switch drag {
        case .divider:
            let p = convert(event.locationInWindow, from: nil)
            guard bounds.width > 0 else { return }
            dividerFraction = min(max(p.x / bounds.width, 0), 1)
            layoutPanes()
        case .draw(let a, _):
            drag = .draw(from: a, to: imagePoint(event))
            applyTransform(notify: false)
        case .move(let a, _, let r):
            drag = .move(from: a, to: imagePoint(event), original: r)
            applyTransform(notify: false)
        default:
            pan(dx: event.deltaX, dy: event.deltaY)
        }
    }

    override func mouseUp(with event: NSEvent) {
        defer {
            drag = nil
            applyTransform(notify: false)
            window?.invalidateCursorRects(for: self)
        }
        switch drag {
        case .draw(let a, let b):
            // A click (or a tiny drag) centers the current box on that point.
            if abs(a.x - b.x) * scale < 6 || abs(a.y - b.y) * scale < 6 {
                let size = selection?.size ?? box?.size ?? .zero
                guard size.width > 0 else { return }
                let r = CGRect(x: a.x - size.width / 2, y: a.y - size.height / 2, width: size.width, height: size.height)
                onSelectArea?(clampInside(r))
            } else if let band = bandRect {
                onSelectArea?(band)
            }
        case .move:
            if let band = bandRect { onSelectArea?(band) }
        default:
            break
        }
    }

    private func clampInside(_ r: CGRect) -> CGRect {
        var r = r
        r.origin.x = min(max(r.minX, 0), max(0, imageSize.width - r.width))
        r.origin.y = min(max(r.minY, 0), max(0, imageSize.height - r.height))
        return r
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53, selectMode || drag != nil { // Esc
            drag = nil
            applyTransform(notify: false)
            onCancelSelect?()
        } else if event.charactersIgnoringModifiers == "\\", event.modifierFlags.intersection(.deviceIndependentFlagsMask).isEmpty {
            peekingBefore = true
        } else {
            super.keyDown(with: event)
        }
    }

    override func keyUp(with event: NSEvent) {
        if event.charactersIgnoringModifiers == "\\" {
            peekingBefore = false
        } else {
            super.keyUp(with: event)
        }
    }

    override func flagsChanged(with event: NSEvent) {
        super.flagsChanged(with: event)
        window?.invalidateCursorRects(for: self)
    }

    override func resetCursorRects() {
        let selecting = selectMode || NSEvent.modifierFlags.contains(.shift)
        addCursorRect(bounds, cursor: selecting ? .crosshair : .openHand)
        if mode == .split, !selecting {
            addCursorRect(CGRect(x: dividerX - 6, y: 0, width: 12, height: bounds.height), cursor: .resizeLeftRight)
        }
    }
}

/// SwiftUI wrapper.
struct CompareViewRepresentable: NSViewRepresentable {
    @ObservedObject var model: DocumentModel

    func makeNSView(context: Context) -> CompareView {
        let view = CompareView(frame: .zero)
        let model = model
        view.onViewportChange = { [weak model] visible, zoom in
            model?.viewportChanged(visible: visible, zoomPercent: zoom)
        }
        view.onSelectArea = { [weak model] rect in model?.selectArea(rect) }
        view.onCancelSelect = { [weak model] in model?.isSelectingArea = false }
        model.viewport = view
        return view
    }

    func updateNSView(_ view: CompareView, context: Context) {
        view.setOriginal(model.image, size: model.imageSize)
        view.mode = model.compareMode
        view.selectMode = model.isSelectingArea
        view.setSelection(model.selection)
        let noSteps = model.signature.items.isEmpty
        view.enhancedIsOriginal = noSteps
        view.setPreview(noSteps ? nil : model.preview?.image, region: noSteps ? nil : model.preview?.region)
        view.setFaces(noSteps || !model.showFaces || !model.hasFaceStep ? [] : model.preview?.faces ?? [])
    }
}
