// PAINITE Lightroom handoff spike.
//
// A throwaway AppKit app that registers as an editor for JPEG/TIFF/PSD/PNG and
// logs exactly what Adobe Lightroom desktop hands over through "Edit in", on
// both cold launch and while already running. It can also perform the three
// return paths the PRD asks us to validate:
//
//   * Save (atomic)    - write a marked copy to a temp file beside the handoff
//                        file and rename(2) it over the handoff path.
//   * Save (in place)  - truncate and rewrite the handoff file (same inode).
//   * Save As...       - write a marked copy elsewhere, for Lightroom "Browse".
//   * Failed write     - write a temp file, then abort before replacing; the
//                        handoff file's hash must be unchanged.
//
// "Marked" means the pixels get an obvious magenta frame so a successful
// return is visible in Lightroom at a glance.
//
// Every event is appended as one JSON object per line to
//   ~/Library/Logs/PainiteHandoffSpike/handoff.jsonl
// and shown in the window.
//
// Not product code: no server calls, no enhancement, minimal UI.

import AppKit
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

// MARK: - Logging

let logDir = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Logs/PainiteHandoffSpike", isDirectory: true)
let logURL = logDir.appendingPathComponent("handoff.jsonl")
let processStart = Date()
let iso: ISO8601DateFormatter = {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return f
}()

func jsonSafe(_ v: Any) -> Any {
    switch v {
    case let s as String: return s
    case let n as NSNumber: return n
    case let b as Bool: return b
    case let d as Date: return iso.string(from: d)
    case let u as URL: return u.path
    case let a as [Any]: return a.map(jsonSafe)
    case let d as [String: Any]: return d.mapValues(jsonSafe)
    case let d as NSDictionary:
        var out: [String: Any] = [:]
        for (k, val) in d { out["\(k)"] = jsonSafe(val) }
        return out
    case let data as Data: return "<\(data.count) bytes>"
    default: return String(describing: v)
    }
}

var uiLog: ((String) -> Void)?

func log(_ event: String, _ fields: [String: Any] = [:]) {
    var rec: [String: Any] = [
        "ts": iso.string(from: Date()),
        "event": event,
        "pid": ProcessInfo.processInfo.processIdentifier,
        "uptime_s": Date().timeIntervalSince(processStart),
    ]
    for (k, v) in fields { rec[k] = jsonSafe(v) }
    guard let data = try? JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys]),
          var line = String(data: data, encoding: .utf8) else { return }
    line += "\n"
    try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
    if let h = try? FileHandle(forWritingTo: logURL) {
        h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); try? h.close()
    } else {
        try? line.write(to: logURL, atomically: false, encoding: .utf8)
    }
    let pretty = (try? JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys, .prettyPrinted]))
        .flatMap { String(data: $0, encoding: .utf8) } ?? line
    DispatchQueue.main.async { uiLog?(pretty + "\n") }
}

// MARK: - Inspection

func sha256(_ url: URL) -> String? {
    guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func xattrs(_ path: String) -> [String] {
    let size = listxattr(path, nil, 0, 0)
    guard size > 0 else { return [] }
    var buf = [CChar](repeating: 0, count: size)
    listxattr(path, &buf, size, 0)
    return buf.split(separator: 0).map { String(cString: Array($0) + [0]) }
}

func fileFacts(_ url: URL) -> [String: Any] {
    let path = url.path
    var out: [String: Any] = ["path": path]
    var st = stat()
    if stat(path, &st) == 0 {
        out["inode"] = NSNumber(value: st.st_ino)
        out["size"] = NSNumber(value: st.st_size)
        out["mtime"] = Date(timeIntervalSince1970: TimeInterval(st.st_mtimespec.tv_sec)
            + TimeInterval(st.st_mtimespec.tv_nsec) / 1e9)
        out["mode_octal"] = String(st.st_mode & 0o7777, radix: 8)
        out["uid"] = NSNumber(value: st.st_uid)
        out["my_uid"] = NSNumber(value: getuid())
        out["nlink"] = NSNumber(value: st.st_nlink)
    } else {
        out["stat_error"] = String(cString: strerror(errno))
    }
    out["file_writable"] = access(path, W_OK) == 0
    out["dir_writable"] = access(url.deletingLastPathComponent().path, W_OK) == 0
    out["xattrs"] = xattrs(path)
    if let vals = try? url.resourceValues(forKeys: [.contentTypeKey, .volumeNameKey, .isUbiquitousItemKey]) {
        out["uttype"] = vals.contentType?.identifier ?? "?"
        out["volume"] = vals.volumeName ?? "?"
        out["ubiquitous"] = vals.isUbiquitousItem ?? false
    }
    if let listing = try? FileManager.default.contentsOfDirectory(atPath: url.deletingLastPathComponent().path) {
        out["dir_listing"] = Array(listing.sorted().prefix(30))
    }
    out["sha256"] = sha256(url) ?? "unreadable"
    return out
}

func imageFacts(_ url: URL) -> [String: Any] {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        return ["image_error": "CGImageSourceCreateWithURL failed"]
    }
    var out: [String: Any] = [
        "container_type": (CGImageSourceGetType(src) as String?) ?? "?",
        "frame_count": CGImageSourceGetCount(src),
    ]
    if let p = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [String: Any] {
        let keep = ["PixelWidth", "PixelHeight", "Depth", "ColorModel", "ProfileName",
                    "Orientation", "HasAlpha", "IsFloat", "DPIWidth", "DPIHeight"]
        for k in keep { if let v = p[k] { out[k] = v } }
        out["metadata_dicts"] = p.keys.filter { $0.hasPrefix("{") }.sorted()
        if let tiff = p["{TIFF}"] as? [String: Any] {
            out["tiff_software"] = tiff["Software"] ?? NSNull()
            out["tiff_compression"] = tiff["Compression"] ?? NSNull()
        }
        if let exif = p["{Exif}"] as? [String: Any] {
            out["exif_colorspace"] = exif["ColorSpace"] ?? NSNull()
            out["exif_keys"] = exif.count
        }
        out["has_iptc"] = p["{IPTC}"] != nil
    }
    if let img = CGImageSourceCreateImageAtIndex(src, 0, nil) {
        out["cg_bits_per_component"] = img.bitsPerComponent
        out["cg_bits_per_pixel"] = img.bitsPerPixel
        out["cg_alpha"] = String(describing: img.alphaInfo.rawValue)
        if let cs = img.colorSpace {
            out["cg_colorspace_name"] = (cs.name as String?) ?? "unnamed"
            if let icc = cs.copyICCData() as Data? { out["icc_bytes"] = icc.count }
        }
    }
    if let xmp = CGImageSourceCopyMetadataAtIndex(src, 0, nil),
       let tags = CGImageMetadataCopyTags(xmp) as? [CGImageMetadataTag] {
        // Look for Lightroom/Camera Raw develop settings baked into XMP.
        var crs: [String] = []
        for t in tags {
            if let prefix = CGImageMetadataTagCopyPrefix(t) as String?,
               let name = CGImageMetadataTagCopyName(t) as String? {
                if prefix == "crs" { crs.append(name) }
            }
        }
        out["xmp_tag_count"] = tags.count
        out["xmp_crs_tags"] = Array(crs.sorted().prefix(40))
    }
    return out
}

// MARK: - Writing

/// Re-encode `src` into `dest` in the same container format with a magenta
/// frame painted over the pixels, keeping metadata and the color space.
func writeMarkedCopy(from srcURL: URL, to destURL: URL) throws {
    guard let src = CGImageSourceCreateWithURL(srcURL as CFURL, nil),
          let type = CGImageSourceGetType(src),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        throw NSError(domain: "spike", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot decode source"])
    }
    let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [String: Any] ?? [:]
    let cs = img.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!
    let bpc = img.bitsPerComponent
    // Keep RGB-only output: Lightroom hands over 3-channel TIFFs, and an
    // alpha channel would change the file's layout and size.
    let alpha = img.alphaInfo == .none || img.alphaInfo == .noneSkipLast || img.alphaInfo == .noneSkipFirst
        ? CGImageAlphaInfo.noneSkipLast : CGImageAlphaInfo.premultipliedLast
    let info: UInt32 = bpc == 16
        ? alpha.rawValue | CGBitmapInfo.byteOrder16Little.rawValue
        : alpha.rawValue
    guard let ctx = CGContext(data: nil, width: img.width, height: img.height,
                              bitsPerComponent: bpc, bytesPerRow: 0, space: cs, bitmapInfo: info) else {
        throw NSError(domain: "spike", code: 2, userInfo: [NSLocalizedDescriptionKey: "cannot create \(bpc)-bit context in \(cs.name as String? ?? "?")"])
    }
    let r = CGRect(x: 0, y: 0, width: img.width, height: img.height)
    ctx.draw(img, in: r)
    let border = max(8.0, Double(min(img.width, img.height)) * 0.03)
    ctx.setStrokeColor(CGColor(srgbRed: 1, green: 0, blue: 1, alpha: 1))
    ctx.setLineWidth(border)
    ctx.stroke(r.insetBy(dx: border / 2, dy: border / 2))
    guard let marked = ctx.makeImage() else {
        throw NSError(domain: "spike", code: 3, userInfo: [NSLocalizedDescriptionKey: "makeImage failed"])
    }
    guard let dest = CGImageDestinationCreateWithURL(destURL as CFURL, type, 1, nil) else {
        throw NSError(domain: "spike", code: 4, userInfo: [NSLocalizedDescriptionKey: "cannot create destination of type \(type)"])
    }
    var outProps = props
    outProps[kCGImageDestinationLossyCompressionQuality as String] = 0.92
    CGImageDestinationAddImage(dest, marked, outProps as CFDictionary)
    guard CGImageDestinationFinalize(dest) else {
        throw NSError(domain: "spike", code: 5, userInfo: [NSLocalizedDescriptionKey: "finalize failed"])
    }
}

func tempURL(beside url: URL) -> URL {
    url.deletingLastPathComponent()
        .appendingPathComponent(".painite-\(UUID().uuidString.prefix(8))-" + url.lastPathComponent)
}

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var textView: NSTextView!
    var pathLabel: NSTextField!
    var current: URL?
    var didFinishLaunching = false
    var openCount = 0

    func applicationWillFinishLaunching(_ n: Notification) {
        log("will_finish_launching", ["argv": CommandLine.arguments])
    }

    func applicationDidFinishLaunching(_ n: Notification) {
        buildUI()
        didFinishLaunching = true
        log("did_finish_launching", ["log_file": logURL.path])
    }

    func application(_ app: NSApplication, open urls: [URL]) {
        // Lightroom launches the app and sends the open event after
        // didFinishLaunching (observed ~0.5 s), so treat the first open that
        // arrives within 2 s of process start as part of a cold launch.
        let early = Date().timeIntervalSince(processStart) < 2.0
        let phase = (!didFinishLaunching || (openCount == 0 && early)) ? "cold" : "warm"
        openCount += urls.count
        for url in urls { inspect(url, phase: phase) }
        if !didFinishLaunching { buildUI() }
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { false }

    func applicationWillTerminate(_ n: Notification) { log("will_terminate") }

    func inspect(_ url: URL, phase: String) {
        current = url
        var f: [String: Any] = ["launch": phase, "url": url.absoluteString]
        f["file"] = fileFacts(url)
        f["image"] = imageFacts(url)
        log("open", f)
        DispatchQueue.main.async { self.pathLabel?.stringValue = url.path }
    }

    // MARK: actions

    @objc func saveAtomic(_ s: Any?) {
        guard let url = current else { return }
        let before = fileFacts(url)
        let tmp = tempURL(beside: url)
        do {
            try writeMarkedCopy(from: url, to: tmp)
            if rename(tmp.path, url.path) != 0 {
                throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
            }
            log("save_atomic_ok", ["before": before, "after": fileFacts(url), "after_image": imageFacts(url)])
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            log("save_atomic_failed", ["error": "\(error)", "before": before, "after": fileFacts(url)])
        }
    }

    @objc func saveInPlace(_ s: Any?) {
        guard let url = current else { return }
        let before = fileFacts(url)
        let tmp = tempURL(beside: url)
        do {
            try writeMarkedCopy(from: url, to: tmp)
            let data = try Data(contentsOf: tmp)
            let h = try FileHandle(forWritingTo: url)  // keeps the inode
            try h.truncate(atOffset: 0)
            try h.write(contentsOf: data)
            try h.synchronize(); try h.close()
            try? FileManager.default.removeItem(at: tmp)
            log("save_inplace_ok", ["before": before, "after": fileFacts(url), "after_image": imageFacts(url)])
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            log("save_inplace_failed", ["error": "\(error)", "before": before, "after": fileFacts(url)])
        }
    }

    @objc func saveAs(_ s: Any?) {
        guard let url = current else { return }
        let panel = NSSavePanel()
        panel.directoryURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Pictures")
        panel.nameFieldStringValue = url.deletingPathExtension().lastPathComponent
            + "-painite." + url.pathExtension
        panel.begin { resp in
            guard resp == .OK, let dest = panel.url else { log("save_as_canceled"); return }
            let before = fileFacts(url)
            do {
                try writeMarkedCopy(from: url, to: dest)
                log("save_as_ok", ["source_before": before, "source_after": fileFacts(url),
                                   "dest": fileFacts(dest), "dest_image": imageFacts(dest)])
            } catch {
                log("save_as_failed", ["error": "\(error)"])
            }
        }
    }

    @objc func simulateFailedWrite(_ s: Any?) {
        guard let url = current else { return }
        let beforeHash = sha256(url)
        let tmp = tempURL(beside: url)
        var wroteTemp = false
        do { try writeMarkedCopy(from: url, to: tmp); wroteTemp = true } catch {}
        // Simulate a crash or upload failure before the replace step.
        try? FileManager.default.removeItem(at: tmp)
        let afterHash = sha256(url)
        log("simulated_failed_write", ["wrote_temp": wroteTemp,
                                       "handoff_unchanged": beforeHash == afterHash,
                                       "sha_before": beforeHash ?? "?", "sha_after": afterHash ?? "?"])
    }

    @objc func reinspect(_ s: Any?) {
        guard let url = current else { return }
        log("reinspect", ["file": fileFacts(url), "image": imageFacts(url)])
    }

    @objc func revealLog(_ s: Any?) {
        NSWorkspace.shared.activateFileViewerSelecting([logURL])
    }

    // MARK: UI

    func buildUI() {
        if window != nil { return }
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 900, height: 640),
                          styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered, defer: false)
        window.title = "PAINITE handoff spike"
        window.center()

        let buttons: [(String, Selector)] = [
            ("Save (atomic)", #selector(saveAtomic)),
            ("Save (in place)", #selector(saveInPlace)),
            ("Save As…", #selector(saveAs)),
            ("Simulate failed write", #selector(simulateFailedWrite)),
            ("Re-inspect", #selector(reinspect)),
            ("Show log", #selector(revealLog)),
        ]
        let bar = NSStackView(views: buttons.map { NSButton(title: $0.0, target: self, action: $0.1) })
        bar.orientation = .horizontal
        pathLabel = NSTextField(labelWithString: current?.path ?? "Waiting for a file from Lightroom…")
        pathLabel.lineBreakMode = .byTruncatingMiddle

        let scroll = NSTextView.scrollableTextView()
        textView = (scroll.documentView as! NSTextView)
        textView.isEditable = false
        textView.font = .monospacedSystemFont(ofSize: 11, weight: .regular)

        let stack = NSStackView(views: [bar, pathLabel, scroll])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.edgeInsets = NSEdgeInsets(top: 10, left: 10, bottom: 10, right: 10)
        scroll.translatesAutoresizingMaskIntoConstraints = false
        stack.addConstraint(scroll.widthAnchor.constraint(equalTo: stack.widthAnchor, constant: -20))
        window.contentView = stack
        window.makeKeyAndOrderFront(nil)

        if let existing = try? String(contentsOf: logURL, encoding: .utf8) {
            textView.string = "Earlier log lines are in \(logURL.path) (\(existing.split(separator: "\n").count) records)\n\n"
        }
        uiLog = { [weak self] s in
            guard let tv = self?.textView else { return }
            tv.textStorage?.append(NSAttributedString(string: s, attributes: [.font: tv.font!, .foregroundColor: NSColor.textColor]))
            tv.scrollToEndOfDocument(nil)
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)

// Minimal menu so Cmd-Q works.
let mainMenu = NSMenu()
let appItem = NSMenuItem()
mainMenu.addItem(appItem)
let appMenu = NSMenu()
appMenu.addItem(withTitle: "Quit PAINITE Handoff Spike", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
appItem.submenu = appMenu
app.mainMenu = mainMenu

app.run()
