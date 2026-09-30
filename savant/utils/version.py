"""Parses VERSION file."""

from pathlib import Path

from .platform import is_aarch64
from .singleton import SingletonMeta

VERSION_FILE_PATH = str(Path(__file__).parent.parent / 'VERSION')


__all__ = ['version']


class Version(metaclass=SingletonMeta):
    __slots__ = '_versions'

    def __init__(self, version_file_path: str = VERSION_FILE_PATH):
        with open(version_file_path, 'r') as file_obj:
            self._versions = dict(
                [
                    map(lambda s: s.strip(), line.split('=', 2))
                    for line in file_obj.read().splitlines()
                ]
            )

    @property
    def SAVANT(self):
        return self._versions['SAVANT']

    @property
    def SAVANT_RS(self):
        return self._versions['SAVANT_RS']

    @property
    def DEEPSTREAM(self):
        if is_aarch64():
            return self._versions['DEEPSTREAM_L4T']
        return self._versions['DEEPSTREAM']


version = Version()
