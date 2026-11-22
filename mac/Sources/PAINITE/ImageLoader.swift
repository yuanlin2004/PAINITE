import CoreImage
import Foundation
import ImageIO
import PainiteKit

/// Decodes photos for display only. Everything sent to the server is the original bytes.
enum ImageLoader {
    struct Loaded: @unchecked Sendable {
        /// Orientation applied, in the file's own color space (sRGB if untagged, matching the server).
        let image: CGImage
        let size: CGSize
        let format: ImageFormat?
        let bitsPerChannel: Int
        /// EXIF orientation (1-8) and the file's pixel size before it is applied,
        /// which the server's tile grid is anchored to.
        let orientation: Int
        let storedWidth: Int
        let storedHeight: Int
    }

    private static let context = CIContext(options: [
        .workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!,
        .workingFormat: CIFormat.RGBAh,
        .cacheIntermediates: false,
    ])

    static func load(_ url: URL) throws -> Loaded {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(src) > 0 else {
            throw APIError(code: "unsupported_format", message: "not an image")
        }
        let type = CGImageSourceGetType(src) as String?
        let format: ImageFormat? = switch type {
        case "public.jpeg": .jpeg
        case "public.tiff": .tiff
        default: nil
        }
        let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any] ?? [:]
        let depth = props[kCGImagePropertyDepth] as? Int ?? 8
        let orientation = props[kCGImagePropertyOrientation] as? Int ?? 1
        let storedWidth = props[kCGImagePropertyPixelWidth] as? Int ?? 0
        let storedHeight = props[kCGImagePropertyPixelHeight] as? Int ?? 0

        guard let ci = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]) else {
            throw APIError(code: "unsupported_format", message: "can't decode this image")
        }
        let space = ci.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!
        let extent = ci.extent.integral
        guard let image = context.createCGImage(ci, from: extent, format: depth > 8 ? .RGBA16 : .RGBA8,
                                                colorSpace: space) else {
            throw APIError(code: "unsupported_format", message: "can't render this image for display")
        }
        return Loaded(image: image, size: extent.size, format: format, bitsPerChannel: depth,
                      orientation: orientation, storedWidth: storedWidth, storedHeight: storedHeight)
    }
}
