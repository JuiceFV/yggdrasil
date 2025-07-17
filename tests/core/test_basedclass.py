from dataclasses import dataclass

import pytest

from yggdrasil.core.base_dclass import BaseDataClass


@dataclass
class Person(BaseDataClass):
    name: str
    age: int


class TestConfigParser:
    def test_replace_updates_field(self) -> None:
        person = Person(name="Alice", age=30)
        new_age = 31
        new_person = person._replace(age=new_age)
        assert new_person.age == new_age
        assert new_person.name == "Alice"

    def test_replace_raises_type_error_on_non_dataclass(self) -> None:
        class NotADataclass(BaseDataClass):
            pass

        not_a_dataclass_instance = NotADataclass()
        with pytest.raises(TypeError, match="Expected dataclass, got"):
            not_a_dataclass_instance._replace(some_field="value")

    def test_replace_returns_new_instance(self) -> None:
        person = Person(name="Alice", age=30)
        new_age = 31
        new_person = person._replace(age=new_age)
        assert new_person is not person
        assert new_person.age == new_age
        assert new_person.name == "Alice"

    def test_replace_multiple_fields(self) -> None:
        person = Person(name="Alice", age=30)
        new_age = 25
        new_person = person._replace(name="Bob", age=new_age)
        assert new_person.name == "Bob"
        assert new_person.age == new_age
        assert new_person is not person

    def test_replace_no_changes(self) -> None:
        person = Person(name="Alice", age=30)
        new_age = 30
        new_person = person._replace()
        assert new_person.name == "Alice"
        assert new_person.age == new_age
        assert new_person is not person
