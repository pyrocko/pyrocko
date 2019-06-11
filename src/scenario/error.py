

class ScenarioError(Exception):
    pass


class CannotCreate(ScenarioError):
    pass


__all__ = ['ScenarioError', 'CannotCreate']
