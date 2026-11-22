import AppKit
import Combine
import PainiteKit
import SwiftUI
import UniformTypeIdentifiers

/// A window for one photo. It sits in the responder chain, so it receives the
/// File and View menu actions.
final class DocumentWindowController: NSWindowController, NSWindowDelegate, NSMenuItemValidation, NSToolbarDelegate {
    let model: DocumentModel
    var onClose: ((DocumentWindowController) -> Void)?
    private var subscriptions: Set<AnyCancellable> = []

    init(url: URL) {
        model = DocumentModel(url: url)
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1440, height: 900),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered, defer: false)
        window.title = url.lastPathComponent
        window.representedURL = url
        window.tabbingMode = .disallowed
        window.setFrameAutosaveName("PAINITEDocument")
        super.init(window: window)
        window.delegate = self
        let root = DocumentView(model: model)
        window.contentViewController = NSHostingController(rootView: root)
        window.setContentSize(NSSize(width: 1440, height: 900))

        let toolbar = NSToolbar(identifier: "PAINITEDocument")
        toolbar.delegate = self
        toolbar.displayMode = .iconOnly
        toolbar.allowsUserCustomization = false
        window.toolbar = toolbar
        window.toolbarStyle = .unified
        window.titlebarSeparatorStyle = .none

        // "4032 × 3024 · TIFF 16-bit", with "From Lightroom · " in front for a Lightroom photo.
        model.$imageSize.combineLatest(model.$sourceFormat, model.$bitsPerChannel)
            .receive(on: RunLoop.main)
            .sink { [weak self] size, format, bits in
                guard let self, size.width > 0 else { return }
                var parts: [String] = []
                if model.handoff.isLightroom { parts.append("From Lightroom") }
                parts.append("\(Int(size.width)) × \(Int(size.height))")
                if let format { parts.append("\(format == .jpeg ? "JPEG" : "TIFF") \(bits)-bit") }
                self.window?.subtitle = parts.joined(separator: " · ")
            }
            .store(in: &subscriptions)
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    // MARK: Toolbar

    private enum Item {
        static let compare = NSToolbarItem.Identifier("compare")
        static let selectArea = NSToolbarItem.Identifier("selectArea")
        static let zoom = NSToolbarItem.Identifier("zoom")
        static let inspector = NSToolbarItem.Identifier("inspector")
        static let export = NSToolbarItem.Identifier("export")
    }

    func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        [.flexibleSpace, Item.compare, Item.selectArea, Item.zoom, .flexibleSpace, Item.inspector, Item.export]
    }

    func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        toolbarDefaultItemIdentifiers(toolbar)
    }

    func toolbar(_ toolbar: NSToolbar, itemForItemIdentifier id: NSToolbarItem.Identifier,
                 willBeInsertedIntoToolbar flag: Bool) -> NSToolbarItem? {
        let item = NSToolbarItem(itemIdentifier: id)
        let view: AnyView
        switch id {
        case Item.compare:
            item.label = "Compare"
            view = AnyView(CompareModeControl(model: model))
        case Item.selectArea:
            item.label = "Select Area"
            view = AnyView(SelectAreaButton(model: model))
        case Item.zoom:
            item.label = "Zoom"
            view = AnyView(ZoomMenu(model: model))
        case Item.inspector:
            item.label = "Inspector"
            view = AnyView(InspectorToggle(model: model))
        case Item.export:
            item.label = "Export"
            view = AnyView(ExportButton(model: model, export: { [weak self] in self?.exportAction() }))
        default:
            return nil
        }
        let host = NSHostingView(rootView: view.tint(.painite))
        host.frame.size = host.fittingSize
        item.view = host
        return item
    }

    // MARK: Menu actions

    /// The toolbar's primary button: Save to Lightroom for a Lightroom photo, else Export….
    func exportAction() {
        if model.canSaveInPlace { saveDocument(nil) } else { saveDocumentAs(nil) }
    }

    @objc func saveDocument(_ sender: Any?) {
        if model.canSaveInPlace {
            confirmUnseenSettings { [weak self] in self?.model.saveInPlace() }
        } else {
            saveDocumentAs(sender)
        }
    }

    @objc func saveDocumentAs(_ sender: Any?) {
        confirmUnseenSettings { [weak self] in self?.runSavePanel() }
    }

    /// Exporting renders the current settings; ask first if the preview doesn't show them yet.
    private func confirmUnseenSettings(_ go: @escaping () -> Void) {
        guard let window, !model.export.isRunning else { return }
        switch model.afterStatus {
        case .updating, .failed:
            let alert = NSAlert()
            alert.messageText = "The preview doesn't show these settings yet."
            alert.informativeText = "The export uses the current steps, which you haven't seen on the photo."
            alert.addButton(withTitle: "Update Preview")
            alert.addButton(withTitle: "Export Anyway")
            alert.beginSheetModal(for: window) { [weak self] response in
                if response == .alertSecondButtonReturn {
                    go()
                } else {
                    self?.model.retryPreview()
                }
            }
        default:
            go()
        }
    }

    private func runSavePanel() {
        guard let window, !model.export.isRunning else { return }
        let panel = NSSavePanel()
        let lightroom = model.handoff.isLightroom
        var format = lightroom ? (model.sourceFormat ?? .tiff) : model.exportFormat
        panel.allowedContentTypes = [format == .jpeg ? .jpeg : .tiff]
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = Handoff.suggestedName(for: model.url, format: format)
        if lightroom {
            // The handoff copy lives in a temporary folder that Lightroom deletes.
            panel.directoryURL = FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask).first
        } else {
            panel.directoryURL = model.url.deletingLastPathComponent()
            // The format lives in the save panel, not the inspector.
            let popup = NSPopUpButton(frame: .zero, pullsDown: false)
            popup.addItems(withTitles: ["TIFF", "JPEG"])
            popup.selectItem(at: format == .jpeg ? 1 : 0)
            let label = NSTextField(labelWithString: "Format:")
            let stack = NSStackView(views: [label, popup])
            stack.edgeInsets = NSEdgeInsets(top: 8, left: 12, bottom: 8, right: 12)
            panel.accessoryView = stack
            let handler = FormatPopupHandler(panel: panel, model: model)
            popup.target = handler
            popup.action = #selector(FormatPopupHandler.changed(_:))
            objc_setAssociatedObject(panel, &FormatPopupHandler.key, handler, .OBJC_ASSOCIATION_RETAIN)
        }
        panel.beginSheetModal(for: window) { [weak self] response in
            guard response == .OK, let url = panel.url, let self else { return }
            if !lightroom { format = model.exportFormat }
            model.saveAs(to: url, format: format)
        }
    }

    @objc func zoomIn(_ sender: Any?) { model.viewport?.zoomIn() }
    @objc func zoomOut(_ sender: Any?) { model.viewport?.zoomOut() }
    @objc func zoomToFit(_ sender: Any?) { model.viewport?.fit() }
    @objc func actualSize(_ sender: Any?) { model.viewport?.actualSize() }
    @objc func selectPreviewArea(_ sender: Any?) { model.isSelectingArea.toggle() }
    @objc func previewAreaInView(_ sender: Any?) { model.previewVisibleArea() }
    @objc func showSplit(_ sender: Any?) { model.compareMode = .split }
    @objc func showSideBySide(_ sender: Any?) { model.compareMode = .sideBySide }
    @objc func toggleInspector(_ sender: Any?) { model.inspectorVisible.toggle() }
    @objc func regeneratePreview(_ sender: Any?) { model.regeneratePreview() }

    func validateMenuItem(_ item: NSMenuItem) -> Bool {
        switch item.action {
        case #selector(saveDocument(_:)):
            item.title = model.canSaveInPlace ? "Save to Lightroom" : "Export…"
            return model.image != nil && model.loadError == nil && !model.export.isRunning
        case #selector(saveDocumentAs(_:)):
            // For a Lightroom photo, Export a Copy; otherwise Export… (⌘S) already does this.
            item.title = "Export a Copy…"
            item.isHidden = !model.canSaveInPlace
            return model.image != nil && model.loadError == nil && !model.export.isRunning
        case #selector(showSplit(_:)):
            item.state = model.compareMode == .split ? .on : .off
            return true
        case #selector(showSideBySide(_:)):
            item.state = model.compareMode == .sideBySide ? .on : .off
            return true
        case #selector(toggleInspector(_:)):
            item.title = model.inspectorVisible ? "Hide Inspector" : "Show Inspector"
            return true
        default:
            return true
        }
    }

    // MARK: Window

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if model.export.isRunning {
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
        if model.canSaveInPlace, !model.savedToLightroom, !model.closeWithoutSaving {
            let alert = NSAlert()
            alert.messageText = "Lightroom is still waiting for this photo."
            alert.informativeText = "Save to Lightroom first, then click Finish in Lightroom. If you close without saving, Lightroom keeps the unenhanced copy."
            alert.addButton(withTitle: "Save to Lightroom")
            alert.addButton(withTitle: "Cancel")
            alert.addButton(withTitle: "Close Without Saving")
            alert.beginSheetModal(for: sender) { [weak self] response in
                switch response {
                case .alertFirstButtonReturn: self?.saveDocument(nil)
                case .alertThirdButtonReturn:
                    self?.model.closeWithoutSaving = true
                    sender.close()
                default: break
                }
            }
            return false
        }
        return true
    }

    func windowWillClose(_ notification: Notification) {
        model.close()
        onClose?(self)
    }
}

/// Keeps the save panel's format pop-up in sync with the document.
@MainActor
private final class FormatPopupHandler: NSObject {
    static var key = 0
    weak var panel: NSSavePanel?
    let model: DocumentModel

    init(panel: NSSavePanel, model: DocumentModel) {
        self.panel = panel
        self.model = model
    }

    @objc func changed(_ sender: NSPopUpButton) {
        let format: ImageFormat = sender.indexOfSelectedItem == 1 ? .jpeg : .tiff
        model.exportFormat = format
        panel?.allowedContentTypes = [format == .jpeg ? .jpeg : .tiff]
        if let name = panel?.nameFieldStringValue {
            let base = (name as NSString).deletingPathExtension
            panel?.nameFieldStringValue = base + "." + format.fileExtension
        }
    }
}

// MARK: Toolbar controls

struct CompareModeControl: View {
    @ObservedObject var model: DocumentModel

    var body: some View {
        Picker("Compare", selection: $model.compareMode) {
            Image(systemName: "rectangle.lefthalf.inset.filled").tag(CompareView.Mode.split)
                .help("Split (⌥⌘1)")
            Image(systemName: "rectangle.split.2x1").tag(CompareView.Mode.sideBySide)
                .help("Side by Side (⌥⌘2)")
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .fixedSize()
        .help("Split or Side by Side. Hold \\ to see Before.")
    }
}

struct SelectAreaButton: View {
    @ObservedObject var model: DocumentModel

    var body: some View {
        Toggle(isOn: $model.isSelectingArea) {
            Image(systemName: "viewfinder")
        }
        .toggleStyle(.button)
        .help("Select the preview area: drag or click on the photo (⌘E). Shift-drag works at any time. ⇧⌘E previews the area in view.")
    }
}

struct ZoomMenu: View {
    @ObservedObject var model: DocumentModel

    var body: some View {
        Menu {
            Button("Zoom to Fit") { model.viewport?.fit() }
            Divider()
            ForEach([50, 100, 200, 400], id: \.self) { p in
                Button("\(p)%") { model.viewport?.zoom(toPercent: Double(p)) }
            }
            Divider()
            Button("Preview Area in View") { model.previewVisibleArea() }
        } label: {
            Text(model.zoomPercent > 0 ? "\(Int(model.zoomPercent.rounded()))%" : "Fit")
                .monospacedDigit()
                .frame(minWidth: 40)
        }
        .fixedSize()
        .help("Zoom (⌘0 Fit, ⌘1 Actual Size, ⌘+ / ⌘−)")
    }
}

struct InspectorToggle: View {
    @ObservedObject var model: DocumentModel

    var body: some View {
        Button { model.inspectorVisible.toggle() } label: {
            Image(systemName: "sidebar.trailing")
        }
        .help(model.inspectorVisible ? "Hide Inspector (⌥⌘I)" : "Show Inspector (⌥⌘I)")
    }
}

struct ExportButton: View {
    @ObservedObject var model: DocumentModel
    var export: () -> Void

    var body: some View {
        Button(action: export) {
            if model.canSaveInPlace {
                Label("Save to Lightroom", systemImage: "arrow.uturn.backward")
            } else {
                Label("Export…", systemImage: "square.and.arrow.up")
            }
        }
        .labelStyle(.titleAndIcon)
        .buttonStyle(.borderedProminent)
        .disabled(model.image == nil || model.loadError != nil || model.export.isRunning)
        .help(model.canSaveInPlace ? "Replace Lightroom's edit copy, then click Finish in Lightroom (⌘S)" : "Render the full photo and save it (⇧⌘S)")
    }
}
