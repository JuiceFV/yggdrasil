import pytest
from pydantic import ValidationError

from yggdrasil.core.dataclasses import dataclass


class TestDataclasses:
    @pytest.fixture(autouse=True)
    def _setup_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch

    def test_vanilla_dataclass(self) -> None:
        self.monkeypatch.setenv("USE_VANILLA_DATACLASS", "1")
        self.monkeypatch.setenv("ARBITRARY_TYPES_ALLOWED", "1")

        @dataclass
        class VanillaClass:
            x: int
            y: str

        test_x_value = 10
        instance = VanillaClass(x=test_x_value, y="test")
        assert instance.x == test_x_value
        assert instance.y == "test"

    def test_pydantic_dataclass(self) -> None:
        self.monkeypatch.setenv("USE_VANILLA_DATACLASS", "0")
        self.monkeypatch.setenv("ARBITRARY_TYPES_ALLOWED", "1")

        @dataclass
        class PydanticClass:
            x: int
            y: str

        test_x_value = 10
        instance = PydanticClass(x=test_x_value, y="test")
        assert instance.x == test_x_value
        assert instance.y == "test"

    def test_type_mismatch(self) -> None:
        self.monkeypatch.setenv("USE_VANILLA_DATACLASS", "0")
        self.monkeypatch.setenv("ARBITRARY_TYPES_ALLOWED", "1")

        @dataclass
        class PydanticClass:
            x: int
            y: str

        with pytest.raises(ValidationError):
            PydanticClass(x="not an int", y="test")  # type: ignore

    def test_config_duplication_error(self) -> None:
        self.monkeypatch.setenv("USE_VANILLA_DATACLASS", "0")
        self.monkeypatch.setenv("ARBITRARY_TYPES_ALLOWED", "1")

        with pytest.raises(KeyError, match="Config duplication occures"):

            @dataclass(config={"arbitrary_types_allowed": True})  # type: ignore
            class PydanticClass:
                x: int
                y: str
