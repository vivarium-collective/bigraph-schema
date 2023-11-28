from abc import ABC, abstractmethod


"""
required_schema_keys = set([
    '_default',
    '_apply',
    '_serialize',
    '_deserialize',
    '_divide',
])

"""


##############################
## Base Type abstract class ##
##############################
class BaseType(ABC):
    name: str

    def __init__(self):
        super().__init__()

    @abstractmethod
    def set_default(self) -> Any:
        """Abstract method for setting the default value for the given type.
        """
        pass

    @abstractmethod
    def _apply(self):
        pass

    @abstractmethod
    def _serialize(self):
        pass

    @abstractmethod
    def _deserialize(self):
        pass

    @abstractmethod
    def _divide(self):
        pass


##############################
## Implementation classes   ##
##############################
class BooleanType(BaseType):
    def __init__(self):
        super().__init__()
        self.name = 'boolean'

    def set_default(self) -> Any:
        return False

    def _apply(self, current: bool, update: bool, bindings=None, types=None) -> bool:
        """Performs a bit flip if `current` does not match `update`, returning update. Returns current if they match."""
        if current != update:
            return update
        else:
            return current

    def _serialize(self, value: bool, bindings=None, types=None) -> ByteString:
        return value.to_bytes()

    def _deserialize(self, serialized: ByteString, bindings=None, types=None) -> bool:
        return bool(serialized)

    def _divide(self, value: bool, bindings=None, types=None):
        """Convert the given boolean to an `int` and pass that value into `divide_int`.

            Args:
                value:`bool`
        """
        value_int = int(value)
        return divide_int(value_int)
