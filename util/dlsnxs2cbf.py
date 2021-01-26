import binascii

import h5py
import numpy as np

from scitbx.array_family import flex

from dxtbx.ext import compress


def get_mask(nfast, nslow):
    module_size_fast, module_size_slow = (1028, 512)
    gap_size_fast, gap_size_slow = (12, 38)
    n_fast, remainder = divmod(nfast, module_size_fast)
    assert (n_fast - 1) * gap_size_fast == remainder

    n_slow, remainder = divmod(nslow, module_size_slow)
    assert (n_slow - 1) * gap_size_slow == remainder

    mask = flex.bool(flex.grid(nslow, nfast), True)
    blit = flex.bool(flex.grid(module_size_slow, module_size_fast), False)

    for j in range(n_slow):
        for i in range(n_fast):
            o_i = i * (module_size_fast + gap_size_fast)
            o_j = j * (module_size_slow + gap_size_slow)
            mask.matrix_paste_block_in_place(blit, i_row=o_j, i_column=o_i)

    return mask


def depends_on(f):
    def finder(thing, path):
        if hasattr(thing, "keys"):
            for k in thing:
                try:
                    finder(thing[k], path=f"{path}/{k}")
                except (OSError, TypeError, ValueError, KeyError):
                    pass

    finder(f, path="")


def get_distance_in_mm(f):
    try:
        D = f["/entry/instrument/detector_distance"]
    except KeyError:
        D = f["/entry/instrument/detector/detector_distance"]
    d = D[()]
    if D.attrs["units"] == np.string_("m"):
        d *= 1000
    elif D.attrs["units"] != np.string_("mm"):
        raise RuntimeError(
            f"unknown distance unit '{D.attrs['units'].decode('latin-1')}'"
        )
    return d


def compute_cbf_header(f, nn=0):

    result = []

    D = float(get_distance_in_mm(f))
    instrument = f["/entry/instrument"]
    name = instrument.attrs.get("short_name", "")
    T = instrument["detector/count_time"][()]
    L = instrument["beam/incident_wavelength"][()]
    A = instrument["attenuator/attenuator_transmission"][()]

    # this timestamp _should_ be in UTC - so can ignore timezone info - at
    # least for purposes here (causes errors for old versions of dxtbx reading
    # the data) - 19 chars needed
    timestamp = f["/entry/start_time"][()].decode()[:19]
    omega = f["/entry/sample/transformations/omega"][()]
    omega_increment = f["/entry/sample/transformations/omega_increment_set"][()]
    chi = float(f["/entry/sample/transformations/chi"][()])
    phi = float(f["/entry/sample/transformations/phi"][()])

    if "/entry/instrument/detector/beam_centre_x" in f:
        Bx = instrument["detector/beam_centre_x"][()]
        By = instrument["detector/beam_centre_y"][()]
    else:
        Bx = instrument["detector/beam_center_x"][()]
        By = instrument["detector/beam_center_y"][()]

    result.append("###CBF: VERSION 1.5, CBFlib v0.7.8 - Eiger detectors")
    result.append("")
    result.append("data_%06d" % (nn + 1))
    result.append("")
    result.append(
        """_array_data.header_convention "PILATUS_1.2"
_array_data.header_contents
;"""
    )
    result.append(f"# Detector: EIGER 2XE 16M S/N 160-0001 Diamond {name}")
    result.append(f"# {timestamp}")
    result.append("# Pixel_size 75e-6 m x 75e-6 m")
    result.append("# Silicon sensor, thickness 0.000450 m")
    result.append(f"# Exposure_time {T:.5f} s")
    result.append(f"# Exposure_period {T:.5f} s")
    result.append("# Tau = 1e-9 s")
    result.append("# Count_cutoff 65535 counts")
    result.append("# Threshold_setting: 0 eV")
    result.append("# Gain_setting: mid gain (vrf = -0.200)")
    result.append("# N_excluded_pixels = 0")
    result.append("# Excluded_pixels: badpix_mask.tif")
    result.append("# Flat_field: (nil)")
    result.append(f"# Wavelength {L:.5f} A")
    result.append(f"# Detector_distance {float(D / 1000.0):.5f} m")
    result.append(f"# Beam_xy ({Bx:.2f}, {By:.2f}) pixels")
    result.append("# Flux 0.000000")
    result.append(f"# Filter_transmission {A:.3f}")
    result.append(f"# Start_angle {omega[nn]:.4f} deg.")
    result.append(f"# Angle_increment {omega_increment[nn]:.4f} deg.")
    result.append("# Detector_2theta 0.0000 deg.")
    result.append("# Polarization 0.990")
    result.append("# Alpha 0.0000 deg.")
    result.append("# Kappa 0.0000 deg.")
    result.append(f"# Phi {phi:.4f} deg.")
    result.append("# Phi_increment 0.0000 deg.")
    result.append(f"# Omega {omega[nn]:.4f} deg.")
    result.append(f"# Omega_increment {omega_increment[nn]:.4f} deg.")
    result.append(f"# Chi {chi:.4f} deg.")
    result.append("# Chi_increment 0.0000 deg.")
    result.append("# Oscillation_axis X.CW")
    result.append("# N_oscillations 1")
    result.append(";")

    return "\n".join(result)


def make_cbf(in_name, template):
    f = h5py.File(in_name, "r")
    depends_on(f)

    mask = None

    start_tag = binascii.unhexlify("0c1a04d5")

    for j in range(len(f["/entry/sample/transformations/omega"][()])):
        block = 1 + (j // 1000)
        i = j % 1000
        header = compute_cbf_header(f, j)
        depth, height, width = f["/entry/data/data_%06d" % block].shape

        data = flex.int(np.int32(f["/entry/data/data_%06d" % block][i]))
        good = data.as_1d() < 65535
        data.as_1d().set_selected(~good, -2)

        # set the tile join regions to -1 - MOSFLM cares about this apparently
        if mask is None:
            mask = get_mask(width, height)
        data.as_1d().set_selected(mask.as_1d(), -1)

        compressed = compress(data)

        mime = """

_array_data.data
;
--CIF-BINARY-FORMAT-SECTION--
Content-Type: application/octet-stream;
     conversions="x-CBF_BYTE_OFFSET"
Content-Transfer-Encoding: BINARY
X-Binary-Size: %d
X-Binary-ID: 1
X-Binary-Element-Type: "signed 32-bit integer"
X-Binary-Element-Byte-Order: LITTLE_ENDIAN
X-Binary-Number-of-Elements: %d
X-Binary-Size-Fastest-Dimension: %d
X-Binary-Size-Second-Dimension: %d
X-Binary-Size-Padding: 4095

""" % (
            len(compressed),
            data.size(),
            data.focus()[1],
            data.focus()[0],
        )

        padding = (
            bytearray(4095)
            + b"""--CIF-BINARY-FORMAT-SECTION----
;"""
        )

        with open(template % (j + 1), "wb") as fout:
            print(template % (j + 1))
            fout.write(("".join(header) + mime).replace("\n", "\r\n").encode("latin-1"))
            fout.write(start_tag)
            fout.write(compressed)
            fout.write(padding)

    f.close()
