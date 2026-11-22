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

    private var original: CGImage?
    private var previewImage: CGImage?
    private var previewRegion: CGRect = .zero
    private var imageSize: CGSize = .zero
    private var box: CGRect?
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
                outline.removeAnimation(forKey: "pulse")
                outline.opacity = 1
                switch state {
                case .pending, .rendering:
                    outline.borderColor = NSColor.systemYellow.withAlphaComponent(0.9).cgColor
                    let pulse = CABasicAnimation(keyPath: "opacity")
                    pulse.fromValue = 1
                    pulse.toValue = 0.15
                    pulse.duration = 0.6
                    pulse.autoreverses = true
                    pulse.repeatCount = .infinity
                    pulse.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                    outline.add(pulse, forKey: "pulse")
                case .ready:
                    outline.borderColor = NSColor(white: 1, alpha: 0.8).cgColor
                case .failed:
                    outline.borderColor = NSColor.systemOrange.cgColor
                }
            }
        }
        boxState = state
        applyTransform(notify: false)
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
            rightBase.opacity = 0.35
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
                outline.isHidden = false
                outline.frame = CGRect(x: o.x + box.minX * scale, y: o.y + box.minY * scale,
                                       width: box.width * scale, height: box.height * scale)
                    .insetBy(dx: -1.5, dy: -1.5)
            } else {
                outline.isHidden = true
            }
        }
        if notify { reportViewport() }
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
        if event.clickCount == 2 {
            if fitMode {
                setScale(1 / backingScale, anchor: paneLocation(event))
            } else {
                fit()
            }
        }
    }

    override func mouseDragged(with event: NSEvent) {
        pan(dx: event.deltaX, dy: event.deltaY)
    }

    override func resetCursorRects() {
        addCursorRect(bounds, cursor: .openHand)
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
        model.viewport = view
        return view
    }

    func updateNSView(_ view: CompareView, context: Context) {
        view.setOriginal(model.image, size: model.imageSize)
        view.setPreview(model.preview?.image, region: model.preview?.region)
    }
}
