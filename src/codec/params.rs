//────────────────────────────── PARAMETERS ──────────────────────────────

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

/// A convenience alias used by `BlockTransform::decode`.
pub type Shape = (usize, usize);
