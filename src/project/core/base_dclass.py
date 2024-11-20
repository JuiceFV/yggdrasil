import dataclasses
from typing import Any, Self


class BaseDataClass:
    r"""
    The base dataclass provides a convenient way to create modified copies of
    dataclass instances, ensuring type safety and immutability (if the dataclass is
    frozen). This is particularly useful for working with immutable data structures
    where you need to create new instances with some updated fields.

    Example::

        from dataclasses import dataclass

        @dataclass
        class Person(BaseDataClass):
            name: str
            age: int

        person = Person(name="Alice", age=30)
        new_person = person._replace(age=31)
        print(new_person)  # Output: Person(name='Alice', age=31)
    """

    def _replace(self, **kwargs: Any) -> Self:
        if not dataclasses.is_dataclass(self):
            msg = f"Expected dataclass, got {type(self)}"
            raise TypeError(msg)
        return dataclasses.replace(self, **kwargs)
