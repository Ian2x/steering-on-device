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
}
