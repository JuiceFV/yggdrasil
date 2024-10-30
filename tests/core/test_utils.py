import warnings

import pytest

from project.core.utils import deprecated_attrs


class TestDeprecatedAttrs:
    def test_deprecated_attrs_warning(self) -> None:
        @deprecated_attrs("old_attr")
        class MyClass:
            def __init__(self) -> None:
                self.old_attr = "deprecated"
                self.new_attr = "active"

        obj = MyClass()

        with pytest.warns(
            DeprecationWarning,
            match="MyClass.old_attr is deprecated and will "
            "be removed in future versions.",
        ):
            _ = obj.old_attr

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _ = obj.new_attr
        assert len(record) == 0

    def test_deprecated_attrs_no_warning(self) -> None:
        @deprecated_attrs("another_old_attr")
        class AnotherClass:
            def __init__(self) -> None:
                self.another_old_attr = "deprecated"
                self.another_new_attr = "active"

        obj = AnotherClass()

        with pytest.warns(
            DeprecationWarning,
            match="AnotherClass.another_old_attr is deprecated and will be removed "
            "in future versions.",
        ):
            _ = obj.another_old_attr

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _ = obj.another_new_attr
        assert len(record) == 0

    def test_multiple_deprecated_attrs(self) -> None:
        @deprecated_attrs("old_attr1", "old_attr2")
        class MultiDeprecatedClass:
            def __init__(self) -> None:
                self.old_attr1 = "deprecated1"
                self.old_attr2 = "deprecated2"
                self.new_attr = "active"

        obj = MultiDeprecatedClass()

        with pytest.warns(
            DeprecationWarning,
            match="MultiDeprecatedClass.old_attr1 is deprecated and will be removed "
            "in future versions.",
        ):
            _ = obj.old_attr1

        with pytest.warns(
            DeprecationWarning,
            match="MultiDeprecatedClass.old_attr2 is deprecated and will be removed"
            " in future versions.",
        ):
            _ = obj.old_attr2

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _ = obj.new_attr
        assert len(record) == 0

    def test_no_deprecated_attrs(self) -> None:
        @deprecated_attrs()
        class NoDeprecatedClass:
            def __init__(self) -> None:
                self.attr1 = "active1"
                self.attr2 = "active2"

        obj = NoDeprecatedClass()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _ = obj.attr1
            _ = obj.attr2
        assert len(record) == 0
