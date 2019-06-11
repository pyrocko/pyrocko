

class ScenarioError(Exception):
    pass


class LocationGenerationError(ScenarioError):
    pass


class CannotCreate(ScenarioError):
    pass


__all__ = ['ScenarioError', 'LocationGenerationError', 'CannotCreate']
