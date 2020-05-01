import ast
import pathlib

import minio
import numpy
import yarl

from scitbx.array_family import flex

from .Format import Format

try:
    import bitshuffle
except ImportError:
    pass


class FormatMinio(Format):
    """Uses minio API to access an object store image"""

    schemes = ["minio"]

    @staticmethod
    def understand(endpoint: str):
        # print(f"Asking to understand() {endpoint}")
        return True

    # class FormatEiger0MQDump(Format):
    #     @staticmethod
    #     def understand(image_file):
    #         if os.path.exists(os.path.join(os.path.split(image_file)[0], "header")):
    #             return True
    #         return False

    def _start(self):
        # print(f"Initializing with endpoint {self._image_file}")
        # Other things look this up
        # self._image_file = endpoint
        # Extract parts from the endpoint and create the client
        url = yarl.URL(self._image_file)
        self._client = minio.Minio(
            f"{url.host}{':' if url.port else ''}{url.port}",
            access_key=url.user,
            secret_key=url.password,
            secure=False,
        )
        # Extract bucket and image from the URL
        path = pathlib.PurePosixPath(url.path)
        self._bucket = path.parts[1]
        self._image_number = path.parts[2]

        # Get the header
        self._header = ast.literal_eval(
            self._client.get_object(self._bucket, "header").read().decode("utf-8")
        )
        # pprint(self._header)
        pass

    def _goniometer(self):
        return None  # return self._goniometer_factory.single_axis()

    def _detector(self):
        """Return a model for a simple detector, presuming no one has
        one of these on a two-theta stage. Assert that the beam centre is
        provided in the Mosflm coordinate frame."""

        distance = self._header["detector_distance"] * 1000
        if distance == 0:
            # XXX hack for development
            distance = 175

        pixel_size_x = self._header["x_pixel_size"]
        pixel_size_y = self._header["y_pixel_size"]
        beam_x = self._header["beam_center_x"] * pixel_size_x * 1000
        beam_y = self._header["beam_center_y"] * pixel_size_y * 1000
        if beam_x == 0 and beam_y == 0:
            # hack for development
            beam_x = 154.87
            beam_y = 165.66

        pixel_size_x = 1000 * self._header["x_pixel_size"]
        pixel_size_y = 1000 * self._header["y_pixel_size"]
        image_size = (
            self._header["x_pixels_in_detector"],
            self._header["y_pixels_in_detector"],
        )

        # XXX fixme hard coded
        overload = 0xFFFF
        underload = -1

        return self._detector_factory.simple(
            "PAD",
            distance,
            (beam_x, beam_y),
            "+x",
            "-y",
            (pixel_size_x, pixel_size_y),
            image_size,
            (underload, overload),
            [],
        )

    def _beam(self, index=None):
        return self._beam_factory.simple(self._header["wavelength"])

    def _scan(self):
        return None

    def get_goniometer(self, index=None):
        return self._goniometer()

    def get_detector(self, index=None):
        return self._detector()

    def get_beam(self, index=None):
        return self._beam()

    def get_scan(self, index=None):
        if index is None:
            return self._scan()
        scan = self._scan()
        if scan is not None:
            return scan[index]
        return scan

    def chunk_to_image(self, chunk, dim, dtype):
        """Decompress chunk to image."""

        blob = numpy.fromstring(chunk[12:], dtype=numpy.uint8)
        image = bitshuffle.decompress_lz4(blob, dim, dtype)
        image = flex.int(image.astype("int32"))
        return image

    def get_raw_data(self):
        nx = self._header["x_pixels_in_detector"]
        ny = self._header["y_pixels_in_detector"]
        depth = self._header["bit_depth_image"]

        assert self._header["pixel_mask_applied"] is True

        if depth == 16:
            dtype = numpy.uint16
        elif depth == 32:
            dtype = numpy.uint32
        else:
            dtype = 1 / 0

        dt = numpy.dtype(dtype)

        # Get the data from the object store
        raw_data = self._client.get_object(self._bucket, self._image_number).read()
        image = self.chunk_to_image(raw_data, dim=(ny, nx), dtype=dt)
        # data = msgpack.unpackb(self.open_file(self._image_file).read(), raw=False)[2]
        # blob = numpy.fromstring(data[12:], dtype=numpy.uint8)
        # if dtype == numpy.uint32:
        #     block = numpy.ndarray(shape=(), dtype=">u4", buffer=data[8:12]) / 4
        #     image = bitshuffle.decompress_lz4(blob, (ny, nx), dt, block)
        # else:
        #     image = bitshuffle.decompress_lz4(blob, (ny, nx), dt)

        # image = flex.int(image.astype("int32"))

        # only need to overwrite values if read and used in 16-bit mode
        if dtype == numpy.uint16:
            bad = 2 ** 16 - 1
            sel = image.as_1d() >= bad
            image.as_1d().set_selected(sel, -1)

        return image

    def get_detectorbase(self, index=None):
        raise NotImplementedError
