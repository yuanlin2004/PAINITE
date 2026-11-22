// Recognizing a file that Lightroom desktop handed over through Edit in.
//
// spikes/lightroom-handoff/RESULTS.md: Lightroom 10.0 always exports a fresh
// 16-bit ProPhoto TIFF into $TMPDIR, with TIFF Software "Adobe Lightroom …",
// and opens it in the app. The cloud original is never handed over. Saving over
// the handoff file and choosing Finish works for JPEG sources; RAW sources only
// offer Select…, so the app always offers Save As and offers Save only when the
// handoff file is writable.

import Foundation
import ImageIO

public struct HandoffInfo: Equatable, Sendable {
    /// The file looks like a Lightroom Edit in copy.
    public var isLightroom: Bool
    /// The file and its folder are writable, so Save can replace it.
    public var isWritable: Bool

    public var canSaveInPlace: Bool { isLightroom && isWritable }

    public static let none = HandoffInfo(isLightroom: false, isWritable: false)
}

public enum Handoff {
    public static func inspect(_ url: URL, temporaryDirectory: URL = FileManager.default.temporaryDirectory) -> HandoffInfo {
        let software = softwareTag(url) ?? ""
        let inTemp = isInside(url, temporaryDirectory)
        let isLightroom = software.hasPrefix("Adobe Lightroom") && inTemp
        return HandoffInfo(isLightroom: isLightroom, isWritable: isLightroom && SafeWrite.isWritable(url))
    }

    static func softwareTag(_ url: URL) -> String? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let tiff = props[kCGImagePropertyTIFFDictionary] as? [CFString: Any] else { return nil }
        return tiff[kCGImagePropertyTIFFSoftware] as? String
    }

    static func isInside(_ url: URL, _ dir: URL) -> Bool {
        let file = url.resolvingSymlinksInPath().standardizedFileURL.path
        var base = dir.resolvingSymlinksInPath().standardizedFileURL.path
        if !base.hasSuffix("/") { base += "/" }
        return file.hasPrefix(base)
    }

    /// Default name for a Save As copy: "<name>-painite.<ext>".
    public static func suggestedName(for source: URL, format: ImageFormat) -> String {
        "\(source.deletingPathExtension().lastPathComponent)-painite.\(format.fileExtension)"
    }
}
