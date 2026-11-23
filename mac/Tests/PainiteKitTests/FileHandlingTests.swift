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
    func testSnapInsetByBand() {
        // Matches server/tests/test_geometry.py test_snap_inset_by_blend_band.
        func snap(_ r: Region, max: Int? = nil) -> Region {
            TileGrid.snap(r, storedWidth: 6000, storedHeight: 4000, orientation: 1, tile: 512, band: 32, maxTiles: max)
        }
        XCTAssertEqual(snap(Region(x: 544, y: 544, w: 960, h: 960)), Region(x: 544, y: 544, w: 960, h: 960))
        XCTAssertEqual(snap(Region(x: 512, y: 512, w: 1024, h: 1024)), Region(x: 0, y: 0, w: 2016, h: 2016))
        XCTAssertEqual(snap(Region(x: 10, y: 10, w: 100, h: 100)), Region(x: 0, y: 0, w: 480, h: 480))
        XCTAssertEqual(snap(Region(x: 5900, y: 3900, w: 50, h: 50)), Region(x: 5664, y: 3616, w: 336, h: 384))
        // A 2048 px request capped to 2x2 tiles around its middle (stored x 2024..4072 -> tiles 5, 6).
        XCTAssertEqual(snap(Region(x: 2024, y: 976, w: 2048, h: 2048), max: 2), Region(x: 2592, y: 1568, w: 960, h: 960))
        // At most 2 tiles per side, also at the image edge.
        let e = snap(Region(x: 5000, y: 3000, w: 1000, h: 1000), max: 2)
        XCTAssertEqual(e, Region(x: 5152, y: 3104, w: 848, h: 896))
        // Idempotent, so the server keeps the app's region.
        for r in [Region(x: 100, y: 700, w: 1234, h: 77), Region(x: 0, y: 0, w: 6000, h: 4000)] {
            let s = snap(r, max: 2)
            XCTAssertEqual(snap(s), s)
        }
    }

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

final class RecipeTests: XCTestCase {
    func testSignatureSkipsDisabledAndRoundsStrength() {
        let steps = [RecipeStep(operation: .denoise, strength: 0.604), RecipeStep(operation: .sharpen, enabled: false),
                     RecipeStep(operation: .upscale2x, strength: 0.9)]
        let sig = Recipe.signature(steps) { $0 != .upscale2x }
        XCTAssertEqual(sig.items.map(\.operation), [.denoise, .upscale2x])
        XCTAssertEqual(sig.items[0].strength, 0.6)
        XCTAssertNil(sig.items[1].strength)
        XCTAssertEqual(Recipe.scale(sig), 2)
        XCTAssertEqual(Recipe.label(sig), "Denoise 60 → Upscale 2×")
    }

    func testCanAdd() {
        var steps = [RecipeStep(operation: .upscale2x)]
        XCTAssertFalse(Recipe.canAdd(.upscale2x, to: steps))
        XCTAssertTrue(Recipe.canAdd(.denoise, to: steps))
        steps += [RecipeStep(operation: .denoise), RecipeStep(operation: .sharpen), RecipeStep(operation: .denoise)]
        XCTAssertFalse(Recipe.canAdd(.sharpen, to: steps))
    }

    func testUnblurKindsAreSeparateOperations() throws {
        XCTAssertEqual(PainiteKit.Operation.unblurMotion.rawValue, "unblur_motion")
        XCTAssertEqual(PainiteKit.Operation.unblurFocus.rawValue, "unblur_focus")
        XCTAssertEqual(PainiteKit.Operation.unblurFocus.stepTitle, "Unblur")
        XCTAssertFalse(PainiteKit.Operation.addable.contains(.unblurFocus))
        let sig = Recipe.signature([RecipeStep(operation: .unblurFocus, strength: 1)]) { _ in true }
        XCTAssertEqual(Recipe.label(sig), "Unblur Focus 100")
        let req = JobRequest(sourceId: "sha256:x", kind: .full, steps: Recipe.jobSteps(sig, models: []),
                             region: nil, format: nil)
        let obj = try JSONSerialization.jsonObject(with: APIClient.encoder.encode(req)) as! [String: Any]
        XCTAssertEqual(obj["operation"] as? String, "unblur_focus")
    }

    func testOneStepIsSentAsPlainOperation() throws {
        let sig = Recipe.signature([RecipeStep(operation: .sharpen, strength: 0.3)]) { _ in true }
        let one = JobRequest(sourceId: "sha256:x", kind: .full, steps: Recipe.jobSteps(sig, models: ["m"]),
                             region: nil, format: nil)
        XCTAssertEqual(one.operation, .sharpen)
        XCTAssertNil(one.steps)
        let obj = try JSONSerialization.jsonObject(with: APIClient.encoder.encode(one)) as! [String: Any]
        XCTAssertEqual(Set(obj.keys), ["source_id", "kind", "operation", "params", "model"])

        let two = Recipe.signature([RecipeStep(operation: .denoise, strength: 0.6), RecipeStep(operation: .upscale2x)]) {
            $0 != .upscale2x
        }
        let req = JobRequest(sourceId: "sha256:x", kind: .preview, steps: Recipe.jobSteps(two, models: []),
                             region: Region(x: 0, y: 0, w: 1, h: 1), format: .tiff)
        let obj2 = try JSONSerialization.jsonObject(with: APIClient.encoder.encode(req)) as! [String: Any]
        XCTAssertEqual(Set(obj2.keys), ["source_id", "kind", "steps", "region", "output"])
        let steps = obj2["steps"] as! [[String: Any]]
        XCTAssertEqual(steps[0]["operation"] as? String, "denoise")
        XCTAssertEqual((steps[0]["params"] as? [String: Any])?["strength"] as? Double, 0.6)
        XCTAssertNil(steps[1]["params"])
    }
}

final class OldServerErrorTests: XCTestCase {
    func testPipelineRejectedByOldServerExplainsWhy() {
        let e = APIError(status: 400, code: "bad_request", message: "operation: Field required")
        XCTAssertTrue(e.localizedDescription.contains("several steps"))
    }
}
