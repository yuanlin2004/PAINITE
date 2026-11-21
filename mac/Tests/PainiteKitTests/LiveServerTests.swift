import CoreGraphics
import Foundation
import ImageIO
@testable import PainiteKit
import UniformTypeIdentifiers
import XCTest

/// Runs the app's job flow against a real server. Skipped unless
/// PAINITE_LIVE=1, which uses the shared config file, or PAINITE_TEST_ENDPOINT
/// and PAINITE_TEST_TOKEN are set.
final class LiveServerTests: XCTestCase {
    func liveClient() throws -> APIClient {
        let env = ProcessInfo.processInfo.environment
        if let endpoint = env["PAINITE_TEST_ENDPOINT"], let token = env["PAINITE_TEST_TOKEN"] {
            return try APIClient(config: ClientConfig(endpoint: endpoint, token: token))
        }
        guard env["PAINITE_LIVE"] == "1" else { throw XCTSkip("set PAINITE_LIVE=1 to run against the configured server") }
        return try APIClient(config: ClientConfig.load())
    }

    func testPreviewAndExportForEachOperation() async throws {
        let client = try liveClient()
        let health = try await client.health()
        XCTAssertEqual(health.status, "ok")
        let caps = try await client.capabilities()

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("painite-live-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let src = dir.appendingPathComponent("gradient.jpg")
        try writeTestJPEG(to: src, width: 1500, height: 1000)
        let sha = try FileDigest.sha256(of: src)
        let source = try await client.uploadSource(file: src, sha256: sha)
        XCTAssertEqual(source.width, 1500)
        let exists = try await client.sourceExists(sha256: sha)
        XCTAssertTrue(exists)

        for op in Operation.allCases {
            guard let model = caps.operation(op)?.defaultModel else { continue }
            let strength: Double? = op == .upscale2x ? nil : 0.6
            // Preview of a region.
            let previewReq = JobRequest(sourceId: source.sourceId, kind: .preview, operation: op, strength: strength,
                                        model: model, region: Region(x: 600, y: 300, w: 200, h: 200), format: .tiff)
            let preview = try await finish(client, try await client.createJob(previewReq, idempotencyKey: "mac-live-\(UUID())"))
            let pr = try XCTUnwrap(preview.region)
            XCTAssertLessThanOrEqual(pr.x, 600)
            XCTAssertEqual(preview.result?.width, pr.w * op.scale)
            let pfile = dir.appendingPathComponent("preview-\(op.rawValue).part")
            try await client.downloadResult(jobID: preview.id, sha256: preview.result?.sha256, to: pfile)
            try await client.deleteResult(jobID: preview.id)

            // Full export, idempotent on retry.
            let key = "mac-live-\(UUID())"
            let fullReq = JobRequest(sourceId: source.sourceId, kind: .full, operation: op, strength: strength,
                                     model: model, region: nil, format: .jpeg)
            let created = try await client.createJob(fullReq, idempotencyKey: key)
            let again = try await client.createJob(fullReq, idempotencyKey: key)
            XCTAssertEqual(created.id, again.id)
            let full = try await finish(client, created)
            XCTAssertEqual(full.result?.width, 1500 * op.scale)
            XCTAssertEqual(full.result?.height, 1000 * op.scale)
            let out = dir.appendingPathComponent("out-\(op.rawValue).jpg")
            let partial = SafeWrite.partialURL(for: out)
            try await client.downloadResult(jobID: full.id, sha256: full.result?.sha256, to: partial)
            try SafeWrite.commit(partial, to: out)
            let img = try XCTUnwrap(CGImageSourceCreateWithURL(out as CFURL, nil))
            let props = CGImageSourceCopyPropertiesAtIndex(img, 0, nil) as! [CFString: Any]
            XCTAssertEqual(props[kCGImagePropertyPixelWidth] as? Int, 1500 * op.scale)
            try await client.deleteResult(jobID: full.id)
        }
    }

    private func finish(_ client: APIClient, _ job: Job) async throws -> Job {
        var j = job
        let deadline = Date().addingTimeInterval(120)
        while !j.state.isTerminal {
            XCTAssertLessThan(Date(), deadline)
            try await Task.sleep(nanoseconds: 200_000_000)
            j = try await client.job(id: j.id)
        }
        XCTAssertEqual(j.state, .completed, "job \(j.id) ended \(j.state): \(j.error?.message ?? "")")
        return j
    }

    private func writeTestJPEG(to url: URL, width: Int, height: Int) throws {
        let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        for y in stride(from: 0, to: height, by: 10) {
            for x in stride(from: 0, to: width, by: 10) {
                ctx.setFillColor(red: CGFloat(x) / CGFloat(width), green: CGFloat(y) / CGFloat(height),
                                 blue: CGFloat((x + y) % 97) / 97, alpha: 1)
                ctx.fill(CGRect(x: x, y: y, width: 10, height: 10))
            }
        }
        let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
    }
}
