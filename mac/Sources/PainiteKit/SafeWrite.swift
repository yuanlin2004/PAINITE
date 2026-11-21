// Writing results without ever leaving a half-written file at the destination.

import Foundation

public enum SafeWrite {
    /// A hidden temporary path in the destination's folder, so the final rename
    /// stays on one volume and is atomic.
    public static func partialURL(for destination: URL) -> URL {
        let dir = destination.deletingLastPathComponent()
        return dir.appendingPathComponent(".\(destination.lastPathComponent).painite-\(UUID().uuidString.prefix(8)).part")
    }

    /// Atomically replaces (or creates) `destination` with `partial`.
    /// The existing destination stays intact if anything fails before this point.
    public static func commit(_ partial: URL, to destination: URL) throws {
        let fd = open(partial.path, O_RDONLY)
        if fd >= 0 {
            _ = fsync(fd)
            close(fd)
        }
        if rename(partial.path, destination.path) != 0 {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }

    /// Whether the file and its folder can be written by this user, which a direct save needs.
    public static func isWritable(_ url: URL) -> Bool {
        let fm = FileManager.default
        return fm.isWritableFile(atPath: url.path) && fm.isWritableFile(atPath: url.deletingLastPathComponent().path)
    }

    /// Whether two URLs point at the same file (after resolving symlinks such as /var → /private/var).
    public static func sameFile(_ a: URL, _ b: URL) -> Bool {
        a.resolvingSymlinksInPath().standardizedFileURL.path == b.resolvingSymlinksInPath().standardizedFileURL.path
    }
}
