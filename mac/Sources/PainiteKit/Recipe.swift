// The list of enhancement steps the photographer builds in the app.

import Foundation

public struct RecipeStep: Identifiable, Equatable, Sendable {
    public let id: UUID
    public var operation: Operation
    /// 0...1 for operations that take a strength; ignored otherwise.
    public var strength: Double
    public var enabled: Bool

    public init(operation: Operation, strength: Double = 0.5, enabled: Bool = true, id: UUID = UUID()) {
        self.id = id
        self.operation = operation
        self.strength = strength
        self.enabled = enabled
    }
}

/// What a set of steps renders: operations in order with strengths rounded to
/// what the slider shows. Two recipes with the same signature produce the same pixels.
public struct RecipeSignature: Hashable, Sendable {
    public struct Item: Hashable, Sendable {
        public var operation: Operation
        public var strength: Double?
    }

    public var items: [Item]
}

public enum Recipe {
    /// Matches the server's limit.
    public static let maxSteps = 4

    /// Whether another step of `op` can be added.
    public static func canAdd(_ op: Operation, to steps: [RecipeStep]) -> Bool {
        steps.count < maxSteps && !(op.scale > 1 && steps.contains { $0.operation.scale > 1 })
    }

    /// The enabled steps as the server sees them. `takesStrength` says which operations use a strength.
    public static func signature(_ steps: [RecipeStep], takesStrength: (Operation) -> Bool) -> RecipeSignature {
        RecipeSignature(items: steps.filter(\.enabled).map { st in
            RecipeSignature.Item(operation: st.operation,
                                 strength: takesStrength(st.operation) ? (st.strength * 100).rounded() / 100 : nil)
        })
    }

    /// Output pixels per input pixel along each axis.
    public static func scale(_ sig: RecipeSignature) -> Int {
        sig.items.reduce(1) { $0 * $1.operation.scale }
    }

    /// Job steps for a signature, pinning models where given (same order as the signature).
    public static func jobSteps(_ sig: RecipeSignature, models: [String?]) -> [JobRequest.Step] {
        sig.items.enumerated().map { i, item in
            JobRequest.Step(operation: item.operation, strength: item.strength, model: i < models.count ? models[i] : nil)
        }
    }

    /// A short label, e.g. "Denoise 60 → Sharpen 30 → Upscale 2×".
    public static func label(_ sig: RecipeSignature) -> String {
        if sig.items.isEmpty { return "No steps" }
        return sig.items.map { item in
            item.strength.map { "\(item.operation.title) \(Int(($0 * 100).rounded()))" } ?? item.operation.title
        }.joined(separator: " → ")
    }
}
