import numpy as np


class BitPacker:
    @staticmethod
    def pack(data: np.ndarray, nbits: int) -> bytes:
        """
        Packs a NumPy array of unsigned integers into a bytes object
        using exactly `nbits` bits per value.
        """
        # Check that arr is an unsigned integer array:
        if not np.issubdtype(data.dtype, np.unsignedinteger):
            raise ValueError("arr must be an unsigned integer array")

        # Ensure all values fit within nbits:
        max_val = (1 << nbits) - 1
        if data.max() > max_val:
            raise ValueError(
                f"Array contains values that do not fit in {nbits} bits."
            )

        length = data.size
        # Create a bit-matrix of shape (length, nbits).
        # bits[i, 0] will be the most significant bit of arr[i],
        # bits[i, nbits-1] the least significant bit.
        bits = np.zeros((length, nbits), dtype=np.uint8)
        for i in range(nbits):
            # Extract the i-th bit from the right.
            bits[:, nbits - 1 - i] = (data >> i) & 1

        # Flatten to a 1-D array of bits and pack into bytes.
        bits_flat = bits.reshape(-1)
        packed = np.packbits(bits_flat, bitorder="big")
        return packed.tobytes()

    @staticmethod
    def unpack(packed: bytes, nbits: int, length: int) -> np.ndarray:
        """
        Unpacks a bytes object back into a NumPy array of unsigned
        integers (using `nbits` bits per value).
        """
        total_bits = length * nbits

        # Convert packed bytes to a bit array. Limit unpacking to total_bits.
        packed_arr = np.frombuffer(packed, dtype=np.uint8)
        bits = np.unpackbits(packed_arr, count=total_bits, bitorder="big")

        # Reshape so each row corresponds to one array element's bits.
        bits = bits.reshape(length, nbits)

        # Convert bits back to integers.
        # bits[i, 0] is the most significant bit, so weight = 2^(nbits-1).
        # bits[i, j] has weight = 2^(nbits-1-j).
        powers = 1 << np.arange(nbits - 1, -1, -1)
        out = bits.dot(powers)

        # Return as unsigned 64-bit (you can cast to another type if needed).
        return out.astype(np.int32)
