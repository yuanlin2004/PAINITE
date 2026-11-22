// Built-in recipes and the last recipe the photographer used.

import Foundation

/// A step as stored in a preset or in the user's defaults (no identity).
public struct StoredStep: Codable, Equatable, Sendable {
    public var operation: Operation
    public var strength: Double
    public var enabled: Bool

    public init(operation: Operation, strength: Double = 0.5, enabled: Bool = true) {
        self.operation = operation
        self.strength = strength
        self.enabled = enabled
    }

    public init(_ step: RecipeStep) {
        self.init(operation: step.operation, strength: step.strength, enabled: step.enabled)
    }

    public func makeStep() -> RecipeStep {
        RecipeStep(operation: operation, strength: strength, enabled: enabled)
    }
}

public struct Preset: Identifiable, Equatable, Sendable {
    public var name: String
    public var steps: [StoredStep]
    public var id: String { name }

    public init(name: String, steps: [StoredStep]) {
        self.name = name
        self.steps = steps
    }

    /// Whether `steps` are exactly this preset (same order, strengths as the slider shows them, switches).
    public func matches(_ steps: [RecipeStep]) -> Bool {
        guard steps.count == self.steps.count else { return false }
        return zip(steps, self.steps).allSatisfy { a, b in
            a.operation == b.operation && a.enabled == b.enabled
                && (a.strength * 100).rounded() == (b.strength * 100).rounded()
        }
    }

    public static let builtIn: [Preset] = [
        Preset(name: "Clean Up", steps: [StoredStep(operation: .denoise, strength: 0.5)]),
        Preset(name: "High ISO", steps: [StoredStep(operation: .denoise, strength: 0.7),
                                         StoredStep(operation: .sharpen, strength: 0.25)]),
        Preset(name: "Portrait", steps: [StoredStep(operation: .denoise, strength: 0.5),
                                         StoredStep(operation: .face, strength: 0.5)]),
        Preset(name: "Rescue Blur", steps: [StoredStep(operation: .unblurMotion, strength: 0.5),
                                            StoredStep(operation: .sharpen, strength: 0.2)]),
        Preset(name: "Print 2×", steps: [StoredStep(operation: .denoise, strength: 0.4),
                                         StoredStep(operation: .sharpen, strength: 0.2),
                                         StoredStep(operation: .upscale2x)]),
    ]

    public static func named(_ name: String?) -> Preset? {
        builtIn.first { $0.name == name }
    }
}

/// The recipe a new photo starts from: the last one used, else Clean Up.
public struct LastRecipe: Codable, Equatable, Sendable {
    public var preset: String?
    public var steps: [StoredStep]

    public init(preset: String?, steps: [StoredStep]) {
        self.preset = preset
        self.steps = steps
    }

    static let key = "lastRecipe"

    public static func load(_ defaults: UserDefaults = .standard) -> LastRecipe {
        if let data = defaults.data(forKey: key),
           let r = try? JSONDecoder().decode(LastRecipe.self, from: data),
           r.steps.count <= Recipe.maxSteps {
            return r
        }
        let clean = Preset.builtIn[0]
        return LastRecipe(preset: clean.name, steps: clean.steps)
    }

    public func save(_ defaults: UserDefaults = .standard) {
        if let data = try? JSONEncoder().encode(self) { defaults.set(data, forKey: LastRecipe.key) }
    }
}

extension Recipe {
    /// Moves steps like `Array.move(fromOffsets:toOffset:)` (SwiftUI's onMove).
    public static func moving(_ steps: [RecipeStep], from source: IndexSet, to destination: Int) -> [RecipeStep] {
        let moved = source.sorted().map { steps[$0] }
        var rest = steps
        for i in source.sorted(by: >) { rest.remove(at: i) }
        let shift = source.filter { $0 < destination }.count
        rest.insert(contentsOf: moved, at: max(0, min(rest.count, destination - shift)))
        return rest
    }

    /// The recipe as a compact label for the After tag, e.g. "Denoise 60 · Sharpen 25".
    public static func shortLabel(_ sig: RecipeSignature) -> String {
        sig.items.map { item in
            let name = item.operation.shortTitle
            return item.strength.map { "\(name) \(Int(($0 * 100).rounded()))" } ?? name
        }.joined(separator: " · ")
    }
}

extension Operation {
    /// The name on a step card and in the After tag.
    public var shortTitle: String {
        switch self {
        case .denoise: "Denoise"
        case .sharpen: "Sharpen"
        case .unblurMotion, .unblurFocus: "Deblur"
        case .upscale2x: "Upscale 2×"
        case .face: "Faces"
        }
    }

    /// SF Symbol for the step's icon tile.
    public var symbolName: String {
        switch self {
        case .denoise: "aqi.medium"
        case .sharpen: "triangle.lefthalf.filled"
        case .unblurMotion, .unblurFocus: "scope"
        case .upscale2x: "arrow.up.left.and.arrow.down.right"
        case .face: "face.smiling"
        }
    }
}
