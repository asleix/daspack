use super::{CodecParams, CodecError};

pub type Shape = (usize, usize);

/// Encoding parameters (kept outside the bit-stream).
#[derive(Debug, Clone)]
pub struct CompressParams {
    pub block_height: usize,
    pub block_width: usize,
    pub lx: usize,
    pub lt: usize,
    pub lpc_order: usize,
    // optional tuning
    pub lpc_bits: u8,
    pub lpc_range: (f64, f64),
    pub row_demean: bool,
}

impl CompressParams {
    pub fn new(
        block_height: usize,
        block_width: usize,
        lx: usize,
        lt: usize,
        lpc_order: usize,
    ) -> Self {
        Self {
            block_height,
            block_width,
            lx,
            lt,
            lpc_order,
            lpc_bits: 8,
            lpc_range: (-1.5, 1.5),
            row_demean: true,
        }
    }

    #[inline]
    pub fn block_shape(&self) -> Shape {
        (self.block_height, self.block_width)
    }
}

impl CodecParams for CompressParams {
    fn serialize(&self) -> Result<Vec<u8>, CodecError> {
        // convert to smaller types, error if out of range
        let bh = u16::try_from(self.block_height)?;
        let bw = u16::try_from(self.block_width)?;
        let lx  = u8::try_from(self.lx)?;
        let lt  = u8::try_from(self.lt)?;
        let ord = u8::try_from(self.lpc_order)?;

        // silently convert f64 to f32
        let r0 = self.lpc_range.0 as f32;
        let r1 = self.lpc_range.1 as f32;

        let mut buf = Vec::with_capacity(2*2 + 4 + 4*2 + 1);
        buf.extend(&bh.to_le_bytes());
        buf.extend(&bw.to_le_bytes());
        buf.push(lx);
        buf.push(lt);
        buf.push(ord);
        buf.push(self.lpc_bits);
        buf.extend(&r0.to_le_bytes());
        buf.extend(&r1.to_le_bytes());
        buf.push(self.row_demean as u8);
        Ok(buf)
    }

    fn read(data: &[u8]) -> Self {
        let mut offset = 0;
        // helper to read u16
        let read_u16 = |data: &[u8], off: &mut usize| -> u16 {
            let bytes: [u8; 2] = data[*off..*off + 2]
                .try_into()
                .expect("Failed to read u16 from bytes");
            *off += 2;
            u16::from_le_bytes(bytes)
        };

        let bh = read_u16(data, &mut offset) as usize;
        let bw = read_u16(data, &mut offset) as usize;
        let lx = data[offset] as usize; offset += 1;
        let lt = data[offset] as usize; offset += 1;
        let ord = data[offset] as usize; offset += 1;
        let bits = data[offset]; offset += 1;

        // read two f32s
        let mut arr4 = [0u8; 4];
        arr4.copy_from_slice(&data[offset..offset + 4]);
        let f0 = f32::from_le_bytes(arr4) as f64;
        offset += 4;
        arr4.copy_from_slice(&data[offset..offset + 4]);
        let f1 = f32::from_le_bytes(arr4) as f64;
        offset += 4;

        let demean = data[offset] != 0;

        CompressParams {
            block_height: bh,
            block_width: bw,
            lx,
            lt,
            lpc_order: ord,
            lpc_bits: bits,
            lpc_range: (f0, f1),
            row_demean: demean,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_default() {
        let params = CompressParams::new(4, 8, 16, 32, 5);
        let bytes = params.serialize().expect("serialize should succeed");
        let decoded = CompressParams::read(&bytes);

        // compare each field since PartialEq isn't derived
        assert_eq!(decoded.block_height, params.block_height);
        assert_eq!(decoded.block_width,  params.block_width);
        assert_eq!(decoded.lx,           params.lx);
        assert_eq!(decoded.lt,           params.lt);
        assert_eq!(decoded.lpc_order,    params.lpc_order);
        assert_eq!(decoded.lpc_bits,     params.lpc_bits);
        assert_eq!(decoded.row_demean,   params.row_demean);

        // lpc_range is cast via f32, so exactly representable here
        assert!((decoded.lpc_range.0 - params.lpc_range.0).abs() < 1e-6);
        assert!((decoded.lpc_range.1 - params.lpc_range.1).abs() < 1e-6);
    }

    #[test]
    fn test_roundtrip_custom() {
        let mut p = CompressParams::new(255, 512, 200, 100, 10);
        p.lpc_bits = 12;
        p.lpc_range = (0.125, 3.75);
        p.row_demean = false;

        let bytes = p.serialize().unwrap();
        let q = CompressParams::read(&bytes);

        assert_eq!(q.block_height, p.block_height);
        assert_eq!(q.block_width,  p.block_width);
        assert_eq!(q.lx,           p.lx);
        assert_eq!(q.lt,           p.lt);
        assert_eq!(q.lpc_order,    p.lpc_order);
        assert_eq!(q.lpc_bits,     p.lpc_bits);
        assert_eq!(q.row_demean,   p.row_demean);

        // check f32 round-trip
        assert!((q.lpc_range.0 - p.lpc_range.0).abs() < 1e-6);
        assert!((q.lpc_range.1 - p.lpc_range.1).abs() < 1e-6);
    }

    #[test]
    fn test_serialize_overflow() {
        // block_height too large for u16
        let p = CompressParams {
            block_height: (u16::MAX as usize) + 1,
            block_width: 8,
            lx: 1,
            lt: 1,
            lpc_order: 1,
            lpc_bits: 8,
            lpc_range: (-1.0, 1.0),
            row_demean: true,
        };

        assert!(p.serialize().is_err(), "should err when block_height > u16::MAX");
    }
}

