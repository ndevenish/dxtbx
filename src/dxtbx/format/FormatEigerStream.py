from __future__ import annotations

from dxtbx.format.Format import Format


class FormatEigerStream(Format):
    @staticmethod
    def understand(image_file):
        return False
