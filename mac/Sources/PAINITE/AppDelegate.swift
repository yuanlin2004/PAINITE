import AppKit
import PainiteKit
import SwiftUI
import UniformTypeIdentifiers

/// AppKit lifecycle, so file opens from Finder and Lightroom's Edit in are
/// handled the same way whether they arrive before or after launch finishes
/// (spikes/lightroom-handoff/RESULTS.md: on a cold launch Lightroom's open
/// event arrives about 0.5 s after didFinishLaunching).
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    static weak var shared: AppDelegate?

    private var documents: [DocumentWindowController] = []
    private var welcome: NSWindow?
    private var settings: NSWindow?

    override init() {
        super.init()
        AppDelegate.shared = self
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.mainMenu = MainMenu.build()
        _ = ServerStore.shared
        // Give a launch-time open event a moment to arrive before showing the welcome window.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            guard let self, documents.isEmpty else { return }
            showWelcome()
        }
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        openDocuments(urls)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag { showWelcome() }
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let busy = documents.filter { $0.model.export.isRunning }
        guard !busy.isEmpty else { return .terminateNow }
        let alert = NSAlert()
        alert.messageText = "Quit and cancel \(busy.count == 1 ? "the running export" : "\(busy.count) running exports")?"
        alert.informativeText = "Nothing is written for a canceled export."
        alert.addButton(withTitle: "Keep Exporting")
        alert.addButton(withTitle: "Quit")
        return alert.runModal() == .alertSecondButtonReturn ? .terminateNow : .terminateCancel
    }

    // MARK: Documents

    static let supportedTypes: [UTType] = [.jpeg, .tiff]

    @objc func openDocument(_ sender: Any?) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = AppDelegate.supportedTypes
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK {
            openDocuments(panel.urls)
        }
    }

    func openDocuments(_ urls: [URL]) {
        for url in urls {
            if let existing = documents.first(where: { SafeWrite.sameFile($0.model.url, url) }) {
                existing.showWindow(nil)
                continue
            }
            let type = UTType(filenameExtension: url.pathExtension)
            if let type, !AppDelegate.supportedTypes.contains(where: { type.conforms(to: $0) }) {
                let alert = NSAlert()
                alert.messageText = "PAINITE can't enhance \(url.lastPathComponent)."
                alert.informativeText = "It works with JPEG and TIFF photos."
                alert.runModal()
                continue
            }
            let doc = DocumentWindowController(url: url)
            doc.onClose = { [weak self] closed in
                self?.documents.removeAll { $0 === closed }
            }
            documents.append(doc)
            if documents.count == 1 { doc.window?.center() } else { doc.window?.cascadeTopLeft(from: .zero) }
            doc.showWindow(nil)
            NSDocumentController.shared.noteNewRecentDocumentURL(url)
        }
        if !documents.isEmpty {
            welcome?.orderOut(nil)
            NSApp.activate()
        }
    }

    // MARK: Windows

    func showWelcome() {
        if welcome == nil {
            let view = WelcomeView(open: { [weak self] in self?.openDocument(nil) },
                                   openURLs: { [weak self] urls in self?.openDocuments(urls) })
            let w = NSWindow(contentViewController: NSHostingController(rootView: view))
            w.title = "PAINITE"
            w.styleMask = [.titled, .closable, .miniaturizable]
            w.isReleasedWhenClosed = false
            w.center()
            welcome = w
        }
        welcome?.makeKeyAndOrderFront(nil)
        NSApp.activate()
    }

    @objc func showSettings(_ sender: Any?) {
        if settings == nil {
            let w = NSWindow(contentViewController: NSHostingController(rootView: SettingsView()))
            w.title = "PAINITE Settings"
            w.styleMask = [.titled, .closable]
            w.isReleasedWhenClosed = false
            w.center()
            settings = w
        }
        settings?.makeKeyAndOrderFront(nil)
        NSApp.activate()
    }

    @objc func showWelcomeWindow(_ sender: Any?) { showWelcome() }
}

enum MainMenu {
    static func build() -> NSMenu {
        let main = NSMenu()

        let app = NSMenu(title: "PAINITE")
        app.addItem(withTitle: "About PAINITE", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        app.addItem(.separator())
        app.addItem(withTitle: "Settings…", action: #selector(AppDelegate.showSettings(_:)), keyEquivalent: ",")
        app.addItem(.separator())
        app.addItem(withTitle: "Hide PAINITE", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        let hideOthers = app.addItem(withTitle: "Hide Others", action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        app.addItem(withTitle: "Show All", action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        app.addItem(.separator())
        app.addItem(withTitle: "Quit PAINITE", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        add(app, to: main)

        let file = NSMenu(title: "File")
        file.addItem(withTitle: "Open…", action: #selector(AppDelegate.openDocument(_:)), keyEquivalent: "o")
        file.addItem(withTitle: "Welcome Window", action: #selector(AppDelegate.showWelcomeWindow(_:)), keyEquivalent: "")
        file.addItem(.separator())
        file.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        file.addItem(withTitle: "Save", action: #selector(DocumentWindowController.saveDocument(_:)), keyEquivalent: "s")
        let saveAs = file.addItem(withTitle: "Save As…", action: #selector(DocumentWindowController.saveDocumentAs(_:)), keyEquivalent: "s")
        saveAs.keyEquivalentModifierMask = [.command, .shift]
        add(file, to: main)

        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        let redo = edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "z")
        redo.keyEquivalentModifierMask = [.command, .shift]
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        add(edit, to: main)

        let view = NSMenu(title: "View")
        view.addItem(withTitle: "Zoom In", action: #selector(DocumentWindowController.zoomIn(_:)), keyEquivalent: "+")
        view.addItem(withTitle: "Zoom Out", action: #selector(DocumentWindowController.zoomOut(_:)), keyEquivalent: "-")
        view.addItem(withTitle: "Zoom to Fit", action: #selector(DocumentWindowController.zoomToFit(_:)), keyEquivalent: "0")
        view.addItem(withTitle: "Actual Size", action: #selector(DocumentWindowController.actualSize(_:)), keyEquivalent: "1")
        view.addItem(.separator())
        view.addItem(withTitle: "Select Preview Area", action: #selector(DocumentWindowController.selectPreviewArea(_:)), keyEquivalent: "e")
        let visible = view.addItem(withTitle: "Preview Area in View", action: #selector(DocumentWindowController.previewAreaInView(_:)), keyEquivalent: "e")
        visible.keyEquivalentModifierMask = [.command, .shift]
        add(view, to: main)

        let window = NSMenu(title: "Window")
        window.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        window.addItem(withTitle: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        add(window, to: main)
        NSApp.windowsMenu = window

        return main
    }

    private static func add(_ menu: NSMenu, to main: NSMenu) {
        let item = NSMenuItem(title: menu.title, action: nil, keyEquivalent: "")
        item.submenu = menu
        main.addItem(item)
    }
}
