/// Makes the coefficient-zero contract explicit: the app must take the exact
/// baseline generation path instead of evaluating an algebraically-zero edit.
public enum ActAddPassRoute: Equatable, Sendable {
    case baseline
    case activationAddition
}

public enum ActAddPassPlanner {
    public static func route(coefficient: Double) -> ActAddPassRoute {
        coefficient == 0 ? .baseline : .activationAddition
    }

    /// Executes the same route decision used by the app. The zero case returns
    /// the baseline closure's value directly; it never evaluates an
    /// algebraically-zero residual intervention.
    public static func run<Output>(
        coefficient: Double,
        baseline: () throws -> Output,
        activationAddition: () throws -> Output
    ) rethrows -> Output {
        switch route(coefficient: coefficient) {
        case .baseline:
            return try baseline()
        case .activationAddition:
            return try activationAddition()
        }
    }
}
