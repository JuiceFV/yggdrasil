__version__ = "0.1.0"

from collections.abc import Iterable
from typing import Any


class _LazyImport:
    """Wraps around classes lazy imported from packaging.version
    Output of the function ``v`` in following snippets are identical::

       from packaging.version import Version
       def v():
           return Version('1.2.3')

    and::

       Version = _LazyImport('Version')
       def v():
           return Version('1.2.3')

    The difference here is that in later example imports
    do not happen until ``v`` is called
    """

    def __init__(self, cls_name: str) -> None:
        self._cls_name = cls_name

    def get_cls(self) -> type:
        try:
            import packaging.version  # type: ignore[import]
        except ImportError:
            # If packaging isn't installed, try and use the vendored copy
            # in pkg_resources
            from pkg_resources import packaging  # type: ignore[attr-defined, no-redef]
        return getattr(packaging.version, self._cls_name)

    def __call__(self, *args: Any, **kwargs: Any) -> object:
        return self.get_cls()(*args, **kwargs)

    def __instancecheck__(self, obj: object) -> bool:
        return isinstance(obj, self.get_cls())


Version = _LazyImport("Version")
InvalidVersion = _LazyImport("InvalidVersion")


class ProjectVersion(str):
    def _convert_to_version(self, inp: Any) -> Any:
        if isinstance(inp, Version.get_cls()):
            return inp
        if isinstance(inp, str):
            return Version(inp)
        if isinstance(inp, Iterable):
            # Ideally this should work for most cases by attempting to group
            # the version tuple, assuming the tuple looks (MAJOR, MINOR, ?PATCH)
            # Examples:
            #   * (1)         -> Version("1")
            #   * (1, 20)     -> Version("1.20")
            #   * (1, 20, 1)  -> Version("1.20.1")
            return Version(".".join(str(item) for item in inp))
        raise InvalidVersion(inp)  # type: ignore

    def _cmp_wrapper(self, cmp: Any, method: str) -> bool:
        try:
            return getattr(Version(self), method)(self._convert_to_version(cmp))
        except BaseException as e:
            if not isinstance(e, InvalidVersion.get_cls()):
                raise
            # Fall back to regular string comparison if dealing with an invalid
            # version like 'parrot'
            return getattr(super(), method)(cmp)


for cmp_method in ["__gt__", "__lt__", "__eq__", "__ge__", "__le__"]:
    setattr(
        ProjectVersion,
        cmp_method,
        lambda x, y, method=cmp_method: x._cmp_wrapper(y, method),
    )

__version__ = ProjectVersion(__version__)
