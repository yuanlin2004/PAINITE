import AppKit
import PainiteKit
import SwiftUI
import UniformTypeIdentifiers

/// A window for one photo. It sits in the responder chain, so it receives the
/// File and View menu actions.
final class DocumentWindowController: NSWindowController, NSWindowDelegate, NSMenuItemValidation {
    let model: DocumentModel
    var onClose: ((DocumentWindowController) -> Void)?

    init(url: URL) {
        model = DocumentModel(url: url)
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 800),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered, defer: false)
        window.title = url.lastPathComponent
        window.representedURL = url
        window.tabbingMode = .disallowed
        window.setFrameAutosaveName("PAINITEDocument")
        super.init(window: window)
        window.delegate = self
        let root = DocumentView(model: model, onSaveAs: { [weak self] in self?.saveDocumentAs(nil) })
        window.contentViewController = NSHostingController(rootView: root)
        window.setContentSize(NSSize(width: 1280, height: 800))
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    // MARK: Menu actions

    @objc func saveDocument(_ sender: Any?) {
        if model.canSaveInPlace {
            model.saveInPlace()
        } else {
            saveDocumentAs(sender)
        }
    }

    @objc func saveDocumentAs(_ sender: Any?) {
        guard let window, !model.export.isRunning else { return }
        let panel = NSSavePanel()
        let format = model.handoff.isLightroom ? (model.sourceFormat ?? .tiff) : model.exportFormat
        panel.allowedContentTypes = [format == .jpeg ? .jpeg : .tiff]
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = Handoff.suggestedName(for: model.url, format: format)
        if model.handoff.isLightroom {
            // The handoff copy lives in a temporary folder that Lightroom deletes.
            panel.directoryURL = FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask).first
        } else {
            panel.directoryURL = model.url.deletingLastPathComponent()
        }
        panel.beginSheetModal(for: window) { [weak self] response in
            guard response == .OK, let url = panel.url, let self else { return }
            model.saveAs(to: url, format: format)
        }
    }

    @objc func zoomIn(_ sender: Any?) { model.viewport?.zoomIn() }
    @objc func zoomOut(_ sender: Any?) { model.viewport?.zoomOut() }
    @objc func zoomToFit(_ sender: Any?) { model.viewport?.fit() }
    @objc func actualSize(_ sender: Any?) { model.viewport?.actualSize() }
    @objc func selectPreviewArea(_ sender: Any?) { model.isSelectingArea.toggle() }
    @objc func previewAreaInView(_ sender: Any?) { model.previewVisibleArea() }

    func validateMenuItem(_ item: NSMenuItem) -> Bool {
        switch item.action {
        case #selector(saveDocument(_:)):
            item.title = model.canSaveInPlace ? "Save to Lightroom Copy" : "Save…"
            return model.image != nil && model.loadError == nil && !model.export.isRunning
        case #selector(saveDocumentAs(_:)):
            return model.image != nil && model.loadError == nil && !model.export.isRunning
        default:
            return true
        }
    }

    // MARK: Window

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        guard model.export.isRunning else { return true }
        let alert = NSAlert()
        alert.messageText = "Cancel the export and close?"
        alert.informativeText = "The export of \(model.title) is still running. Nothing is written if you cancel."
        alert.addButton(withTitle: "Keep Exporting")
        alert.addButton(withTitle: "Cancel Export and Close")
        alert.beginSheetModal(for: sender) { [weak self] response in
            if response == .alertSecondButtonReturn {
                self?.model.cancelExport()
                sender.close()
            }
        }
        return false
    }

    func windowWillClose(_ notification: Notification) {
        model.close()
        onClose?(self)
    }
}
