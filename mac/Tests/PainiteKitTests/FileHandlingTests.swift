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
