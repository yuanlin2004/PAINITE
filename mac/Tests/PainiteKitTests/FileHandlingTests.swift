import Foundation
@testable import PainiteKit
import XCTest

final class FileHandlingTests: XCTestCase {
    var dir: URL!

    override func setUp() {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("painite-tests-\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: dir)
    }

    func testConfigRoundTripKeepsUnknownKeysAndIsPrivate() throws {
        let path = dir.appendingPathComponent("painite/config.json")
        try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(#"{"endpoint":"http://old:1","future_key":"keep me"}"#.utf8).write(to: path)

        let loaded = try ClientConfig.load(from: path)
        XCTAssertEqual(loaded.endpoint, "http://old:1")
        XCTAssertEqual(loaded.token, "")

        try ClientConfig(endpoint: "http://spark1:8443", token: "secret", caFile: "").save(to: path)
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
        XCTAssertEqual(raw["future_key"] as? String, "keep me")
        XCTAssertEqual(raw["token"] as? String, "secret")
        XCTAssertNil(raw["ca_file"])
        let perms = try FileManager.default.attributesOfItem(atPath: path.path)[.posixPermissions] as? Int
        XCTAssertEqual(perms, 0o600)
        XCTAssertEqual(try ClientConfig.load(from: path), ClientConfig(endpoint: "http://spark1:8443", token: "secret"))
    }

    func testMissingConfigIsEmpty() throws {
        XCTAssertEqual(try ClientConfig.load(from: dir.appendingPathComponent("nope.json")), ClientConfig())
    }

    func testSafeWriteReplacesAtomically() throws {
        let dest = dir.appendingPathComponent("photo-painite.tif")
        try Data("old".utf8).write(to: dest)
        let partial = SafeWrite.partialURL(for: dest)
        XCTAssertEqual(partial.deletingLastPathComponent().path, dir.path)
        XCTAssertTrue(partial.lastPathComponent.hasPrefix("."))
        try Data("new".utf8).write(to: partial)
        try SafeWrite.commit(partial, to: dest)
        XCTAssertEqual(try String(contentsOf: dest, encoding: .utf8), "new")
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
    }

    func testDigest() throws {
        let f = dir.appendingPathComponent("abc")
        try Data("abc".utf8).write(to: f)
        XCTAssertEqual(try FileDigest.sha256(of: f), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    func testHandoffNeedsLightroomTagAndTemporaryFolder() throws {
        // A plain file in a temp folder is not a handoff (no Lightroom Software tag).
        let f = dir.appendingPathComponent("x.tif")
        try Data("not an image".utf8).write(to: f)
        XCTAssertEqual(Handoff.inspect(f, temporaryDirectory: dir), .none)
        XCTAssertTrue(Handoff.isInside(f, dir))
        XCTAssertFalse(Handoff.isInside(URL(fileURLWithPath: "/Users/someone/Pictures/x.tif"), dir))
        XCTAssertEqual(Handoff.suggestedName(for: f, format: .jpeg), "x-painite.jpg")
    }

    func testPreviewRegionIsClippedAndCapped() {
        let size = CGSize(width: 6000, height: 4000)
        // Fully zoomed out: capped around the center.
        let r = PreviewGeometry.requestRegion(visible: CGRect(x: -500, y: -500, width: 7000, height: 5000),
                                              imageSize: size, operation: .denoise)!
        XCTAssertEqual(r.w, 2048)
        XCTAssertEqual(r.h, 2048)
        XCTAssertEqual(r.x, 3000 - 1024)
        // Upscale is capped smaller because it renders four times the pixels.
        let u = PreviewGeometry.requestRegion(visible: CGRect(x: 0, y: 0, width: 6000, height: 4000),
                                              imageSize: size, operation: .upscale2x)!
        XCTAssertEqual(u.w, 1024)
        // Partly off the image: clipped.
        let c = PreviewGeometry.requestRegion(visible: CGRect(x: 5800.5, y: -10, width: 400, height: 300),
                                              imageSize: size, operation: .sharpen)!
        XCTAssertEqual(c, Region(x: 5800, y: 0, w: 200, h: 290))
        XCTAssertNil(PreviewGeometry.requestRegion(visible: CGRect(x: 7000, y: 0, width: 10, height: 10),
                                                   imageSize: size, operation: .denoise))
        XCTAssertTrue(PreviewGeometry.covers(Region(x: 0, y: 0, w: 512, h: 512), Region(x: 10, y: 10, w: 100, h: 100)))
        XCTAssertFalse(PreviewGeometry.covers(Region(x: 0, y: 0, w: 512, h: 512), Region(x: 10, y: 10, w: 600, h: 100)))
    }
}

final class CATrustTests: XCTestCase {
    func testFingerprintNormalization() {
        XCTAssertEqual(CATrust.normalize("SHA256:AB12cd"), "SHA256:ab12cd")
        XCTAssertEqual(CATrust.normalize(" ab:12:CD \n"), "SHA256:ab12cd")
        XCTAssertEqual(CATrust.fingerprint(der: Data("abc".utf8)),
                       "SHA256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    func testFetchNeedsHTTPS() async {
        do {
            _ = try await CATrust.fetch(endpoint: "http://spark1:8443")
            XCTFail("expected an error")
        } catch {
            XCTAssertEqual((error as? APIError)?.code, "not_configured")
        }
    }

    func testCAPathIsNextToConfig() {
        XCTAssertEqual(CATrust.defaultCAPath(configPath: URL(fileURLWithPath: "/x/painite/config.json")).path, "/x/painite/ca.pem")
    }
}

final class TileGridTests: XCTestCase {
    func testSnapUnrotated() {
        let r = TileGrid.snap(Region(x: 600, y: 300, w: 200, h: 200), storedWidth: 1500, storedHeight: 1000, orientation: 1, tile: 512)
        XCTAssertEqual(r, Region(x: 512, y: 0, w: 512, h: 512))
        // Clipped at the image edge.
        let e = TileGrid.snap(Region(x: 1400, y: 900, w: 100, h: 100), storedWidth: 1500, storedHeight: 1000, orientation: 1, tile: 512)
        XCTAssertEqual(e, Region(x: 1024, y: 512, w: 476, h: 488))
    }

    func testSnapRotated() {
        // Orientation 6 (90° CW): stored 1500x1000 displays as 1000x1500.
        // Display top-left corner maps to stored x 0..., y from the bottom; the grid is
        // anchored at the stored origin, so the snapped box hugs the display's right/top edges.
        let r = TileGrid.snap(Region(x: 0, y: 0, w: 10, h: 10), storedWidth: 1500, storedHeight: 1000, orientation: 6, tile: 512)
        // Stored rect: x 0..9, y 990..999 -> snapped x 0..511, y 512..999 -> display x 0..487, y 0..511.
        XCTAssertEqual(r, Region(x: 0, y: 0, w: 488, h: 512))
        // Round trip is consistent for every orientation: the snapped box contains the request.
        for o in 1...8 {
            let want = Region(x: 333, y: 444, w: 50, h: 60)
            let (dw, dh) = o >= 5 ? (1000, 1500) : (1500, 1000)
            XCTAssertLessThan(want.x + want.w, dw)
            XCTAssertLessThan(want.y + want.h, dh)
            let s = TileGrid.snap(want, storedWidth: 1500, storedHeight: 1000, orientation: o, tile: 512)
            XCTAssertTrue(PreviewGeometry.covers(s, want), "orientation \(o): \(s)")
        }
    }
}
