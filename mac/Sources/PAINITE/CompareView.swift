import AppKit
import PainiteKit
import QuartzCore
import SwiftUI

/// Before and after, side by side, always at the same zoom and position.
///
/// Both panes show the same displayed-source-pixel rectangle. The after pane
/// shows the original dimmed, with the server's preview on top of the region
/// it covers. Upscaled previews are drawn into the same source region, so both
/// sides keep the same framing (PRD). Layers are color matched by Core Animation.
final class CompareView: NSView {
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

    private var original: CGImage?
    private var previewImage: CGImage?
    private var previewRegion: CGRect = .zero
    private var imageSize: CGSize = .zero
    private var box: CGRect?
    /// The picked area (before tile snapping), which a drag inside the box moves.
    private var selection: CGRect?
    private var drag: Drag?

    private enum Drag {
        case pan
        /// Drawing a new area from this image point.
        case draw(from: CGPoint, to: CGPoint)
        /// Moving the selection by the distance from this image point.
        case move(from: CGPoint, to: CGPoint, original: CGRect)
    }
    /// With no steps the enhanced side is just the original, shown undimmed.
    var enhancedIsOriginal = false {
        didSet { if oldValue != enhancedIsOriginal { applyTransform(notify: false) } }
    }
    private var boxState: BoxState = .pending

    enum BoxState {
        /// Waiting to send, or the server is rendering it: the outline pulses.
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
    private let divider = CALayer()
    private let leftImage = CALayer()
    private let rightBase = CALayer()
    private let rightPreview = CALayer()
    private let outline = CALayer()
    /// The same box on the original side.
    private let outlineLeft = CALayer()
    /// The area being drawn or moved, in both panes.
    private let bands = [CALayer(), CALayer()]
    /// Face outlines (from a face step), in both panes.
    private var faces: [FaceInfo] = []
    private var faceLayers: [CAShapeLayer] = []

    override init(frame: NSRect) {
        super.init(frame: frame)
        let root = CALayer()
        root.isGeometryFlipped = true
        root.backgroundColor = NSColor(white: 0.12, alpha: 1).cgColor
        layer = root
        wantsLayer = true
        for pane in [leftPane, rightPane] {
            pane.masksToBounds = true
            root.addSublayer(pane)
        }
        divider.backgroundColor = NSColor(white: 0.35, alpha: 1).cgColor
        root.addSublayer(divider)
        for l in [leftImage, rightBase, rightPreview] {
            l.anchorPoint = .zero
            l.minificationFilter = .trilinear
            l.contentsGravity = .resize
        }
        leftPane.addSublayer(leftImage)
        rightPane.addSublayer(rightBase)
        rightPane.addSublayer(rightPreview)
        outline.borderColor = NSColor(white: 1, alpha: 0.8).cgColor
        outline.borderWidth = 1.5
        outline.isHidden = true
        rightPane.addSublayer(outline)
        outlineLeft.borderWidth = 1.5
        outlineLeft.isHidden = true
        leftPane.addSublayer(outlineLeft)
        for (band, pane) in zip(bands, [leftPane, rightPane]) {
            band.borderColor = NSColor.controlAccentColor.cgColor
            band.borderWidth = 1.5
            band.backgroundColor = NSColor.controlAccentColor.withAlphaComponent(0.12).cgColor
            band.isHidden = true
            pane.addSublayer(band)
        }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }
    override var wantsUpdateLayer: Bool { true }

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
            for pane in [leftPane, rightPane] {
                let l = CAShapeLayer()
                l.fillColor = nil
                l.lineWidth = 1.5
                if face.status == .enhanced {
                    l.strokeColor = NSColor.systemGreen.cgColor
                } else {
                    l.strokeColor = NSColor.systemOrange.cgColor
                    l.lineDashPattern = [5, 4]
                }
                pane.insertSublayer(l, below: outline.superlayer === pane ? outline : outlineLeft)
                faceLayers.append(l)
            }
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

    /// The region the preview covers or is about to cover, in displayed source pixels.
    /// Moves with every zoom and pan; pulses while the server is working on it.
    func setPreviewBox(_ region: Region?, state: BoxState) {
        box = region.map { CGRect(x: $0.x, y: $0.y, width: $0.w, height: $0.h) }
        // Pending and rendering look the same, so the pulse doesn't restart when the request goes out.
        let style: (BoxState) -> Int = { $0 == .ready ? 1 : $0 == .failed ? 2 : 0 }
        if style(state) != style(boxState) || (style(state) == 0 && outline.animation(forKey: "pulse") == nil) {
            withoutAnimation {
                for o in [outline, outlineLeft] {
                    o.removeAnimation(forKey: "pulse")
                    o.opacity = 1
                    switch state {
                    case .pending, .rendering:
                        o.borderColor = NSColor.systemYellow.withAlphaComponent(0.9).cgColor
                        let pulse = CABasicAnimation(keyPath: "opacity")
                        pulse.fromValue = 1
                        pulse.toValue = 0.15
                        pulse.duration = 0.6
                        pulse.autoreverses = true
                        pulse.repeatCount = .infinity
                        pulse.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                        o.add(pulse, forKey: "pulse")
                    case .ready:
                        o.borderColor = NSColor(white: 1, alpha: 0.8).cgColor
                    case .failed:
                        o.borderColor = NSColor.systemOrange.cgColor
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

    // MARK: Geometry

    private var backingScale: CGFloat { window?.backingScaleFactor ?? 2 }
    private var paneSize: CGSize { leftPane.bounds.size }

    var zoomPercent: Double { Double(scale * backingScale * 100) }

    private var fitScale: CGFloat {
        guard imageSize.width > 0, imageSize.height > 0, paneSize.width > 0 else { return 1 }
        return min(paneSize.width / imageSize.width, paneSize.height / imageSize.height)
    }

    override func layout() {
        super.layout()
        withoutAnimation {
            let w = bounds.width
            let half = floor((w - 1) / 2)
            leftPane.frame = CGRect(x: 0, y: 0, width: half, height: bounds.height)
            divider.frame = CGRect(x: half, y: 0, width: 1, height: bounds.height)
            rightPane.frame = CGRect(x: half + 1, y: 0, width: w - half - 1, height: bounds.height)
        }
        if fitMode { fit() } else { applyTransform() }
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        for l in [leftImage, rightBase, rightPreview] { l.contentsScale = 1 }
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
        withoutAnimation {
            for l in [leftImage, rightBase] {
                l.bounds = CGRect(origin: .zero, size: imageSize)
                l.position = o
                l.transform = CATransform3DMakeScale(scale, scale, 1)
                l.magnificationFilter = pixelZoom > 1.01 ? .nearest : .linear
            }
            rightBase.opacity = Float(enhancedIsOriginal ? 1 : 0.35)
            if let previewImage, previewRegion.width > 0 {
                let pw = CGFloat(previewImage.width)
                let ph = CGFloat(previewImage.height)
                let s = scale * previewRegion.width / pw
                rightPreview.isHidden = false
                rightPreview.bounds = CGRect(x: 0, y: 0, width: pw, height: ph)
                rightPreview.position = CGPoint(x: o.x + previewRegion.minX * scale, y: o.y + previewRegion.minY * scale)
                rightPreview.transform = CATransform3DMakeScale(s, s, 1)
                rightPreview.magnificationFilter = s * backingScale > 1.01 ? .nearest : .linear
            } else {
                rightPreview.isHidden = true
            }
            if let box, box.width > 0 {
                let frame = paneRect(box, origin: o).insetBy(dx: -1.5, dy: -1.5)
                for l in [outline, outlineLeft] {
                    l.isHidden = false
                    l.frame = frame
                }
            } else {
                outline.isHidden = true
                outlineLeft.isHidden = true
            }
            for (i, face) in faces.enumerated() {
                let r = CGRect(x: face.box.x, y: face.box.y, width: face.box.w, height: face.box.h)
                let frame = paneRect(r, origin: o)
                for l in faceLayers[(2 * i)..<(2 * i + 2)] {
                    l.frame = frame
                    l.path = CGPath(rect: CGRect(origin: .zero, size: frame.size), transform: nil)
                }
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
        drag = .pan
        if event.clickCount == 2 {
            if fitMode {
                setScale(1 / backingScale, anchor: paneLocation(event))
            } else {
                fit()
            }
        }
    }

    override func mouseDragged(with event: NSEvent) {
        switch drag {
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
        } else {
            super.keyDown(with: event)
        }
    }

    override func flagsChanged(with event: NSEvent) {
        super.flagsChanged(with: event)
        window?.invalidateCursorRects(for: self)
    }

    override func resetCursorRects() {
        let selecting = selectMode || NSEvent.modifierFlags.contains(.shift)
        addCursorRect(bounds, cursor: selecting ? .crosshair : .openHand)
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
        view.selectMode = model.isSelectingArea
        view.setSelection(model.selection)
        let noSteps = model.signature.items.isEmpty
        view.enhancedIsOriginal = noSteps
        view.setPreview(noSteps ? nil : model.preview?.image, region: noSteps ? nil : model.preview?.region)
        view.setFaces(noSteps || !model.showFaces ? [] : model.preview?.faces ?? [])
    }
}
