

class ScenarioError(Exception):
    pass


class LocationGenerationError(ScenarioError):
    pass


class CannotCreatePath(ScenarioError):
    pass


__all__ = ['ScenarioError', 'LocationGenerationError', 'CannotCreatePath']
