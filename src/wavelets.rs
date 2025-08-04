use ndarray::{
    Array1, Array2, ArrayViewMut1, ArrayViewMut2, s
};

/// Forward 1D 5/3 transform (in-place) on a mutable 1D view.
pub fn fwd_txfm_1d_inplace(mut data: ArrayViewMut1<i32>) {
    let n = data.len();
    if n < 2 { return; }

    let s_len = (n + 1) >> 1;  // even samples
    let d_len = n >> 1;        // odd samples

    let mut s = Array1::<i32>::zeros(s_len);
    let mut d = Array1::<i32>::zeros(d_len);

    // split
    for i in 0..s_len { s[i] = data[2*i]; }
    for i in 0..d_len { d[i] = data[2*i + 1]; }

    // predict: d[i] -= (s[i] + s[i+1]) >> 1   with periodic wrap
    for i in 0..d_len {
        let s_l = s[i];
        let s_r = s[(i + 1) % s_len];
        d[i] -= (s_l + s_r) >> 1;
    }

    // update: s[i] += (d[i-1] + d[i] + 2) >> 2   with periodic wrap
    for i in 0..s_len {
        let dl = d[(i + d_len - 1) % d_len];
        let dr = d[i % d_len];
        s[i] += (dl + dr + 2) >> 2;
    }

    // pack back as [s | d]
    for i in 0..s_len { data[i] = s[i]; }
    for i in 0..d_len { data[s_len + i] = d[i]; }
}

/// Inverse 1D 5/3 transform (in-place) on a mutable 1D view.
pub fn inv_txfm_1d_inplace(mut data: ArrayViewMut1<i32>) {
    let n = data.len();
    if n < 2 { return; }

    let d_len = n >> 1;
    let s_len = (n + 1) >> 1;

    let s_vals = data.slice(s![0..s_len]).to_owned();
    let d_vals = data.slice(s![s_len..]).to_owned();

    let mut s0 = Array1::<i32>::zeros(s_len);
    let mut d0 = Array1::<i32>::zeros(d_len);

    // undo update: s0[i] = s[i] - ((d[i-1] + d[i] + 2) >> 2)
    for i in 0..s_len {
        let dl = d_vals[(i + d_len - 1) % d_len];
        let dr = d_vals[i % d_len];
        s0[i] = s_vals[i] - ((dl + dr + 2) >> 2);
    }

    // undo predict: d0[i] = d[i] + ((s0[i] + s0[i+1]) >> 1)
    for i in 0..d_len {
        let sl = s0[i];
        let sr = s0[(i + 1) % s_len];
        d0[i] = d_vals[i] + ((sl + sr) >> 1);
    }

    // interleave back
    let mut si = 0;
    let mut di = 0;
    for idx in 0..n {
        if idx & 1 == 0 {
            data[idx] = s0[si]; si += 1;
        } else {
            data[idx] = d0[di]; di += 1;
        }
    }
}


/// Forward 2D transform in-place: transform each row, then each column.
pub fn fwd_txfm2d_inplace(matrix: &mut Array2<i32>) {
    let (rows, cols) = matrix.dim();
    if rows == 0 || cols < 2 {
        return;
    }

    // Transform each row
    for r in 0..rows {
        let row_view = matrix.row_mut(r);
        fwd_txfm_1d_inplace(row_view);
    }

    // Transform each column
    // We'll gather the column into a 1D array, do fwd transform, and scatter back.
    let mut col_scratch = Array1::<i32>::zeros(rows);
    for c in 0..cols {
        // Gather column c
        for r in 0..rows {
            col_scratch[r] = matrix[(r, c)];
        }

        // Forward 1D
        {
            let view = col_scratch.view_mut();
            fwd_txfm_1d_inplace(view);
        }

        // Scatter back
        for r in 0..rows {
            matrix[(r, c)] = col_scratch[r];
        }
    }
}

/// Inverse 2D transform in-place: inverse each column first, then each row.
pub fn inv_txfm2d_inplace(matrix: &mut Array2<i32>) {
    let (rows, cols) = matrix.dim();
    if rows == 0 || cols < 2 {
        return;
    }

    // Inverse each column
    let mut col_scratch = Array1::<i32>::zeros(rows);
    for c in 0..cols {
        // Gather column c
        for r in 0..rows {
            col_scratch[r] = matrix[(r, c)];
        }
        // Inverse 1D
        {
            let view = col_scratch.view_mut();
            inv_txfm_1d_inplace(view);
        }
        // Scatter back
        for r in 0..rows {
            matrix[(r, c)] = col_scratch[r];
        }
    }

    // Inverse each row
    for r in 0..rows {
        let row_view = matrix.row_mut(r);
        inv_txfm_1d_inplace(row_view);
    }
}


/// Forward multi-level 2D DWT in-place using 5/3 transform.
pub fn fwd_txfm2d_levels_inplace(matrix: &mut Array2<i32>, levels: usize) {
    if levels == 0 {
        return;
    }
    // One level on the full matrix
    fwd_txfm2d_inplace(matrix);

    let (rows, cols) = matrix.dim();
    if rows == 0 || cols == 0 {
        return;
    }

    // LL size = ceil halves
    let low_r = (rows + 1) >> 1;
    let low_c = (cols + 1) >> 1;

    // No more recursion if we've reached the level budget or LL is 1x1
    if levels == 1 || (low_r <= 1 && low_c <= 1) {
        return;
    }

    // Recurse on LL
    let mut ll_sub = matrix.slice(s![0..low_r, 0..low_c]).to_owned();
    fwd_txfm2d_levels_inplace(&mut ll_sub, levels - 1);
    matrix.slice_mut(s![0..low_r, 0..low_c]).assign(&ll_sub);
}


/// Inverse multi-level 2D DWT in-place using 5/3 transform.
pub fn inv_txfm2d_levels_inplace(matrix: &mut Array2<i32>, levels: usize) {
    if levels == 0 {
        return;
    }
    let (rows, cols) = matrix.dim();
    if rows == 0 || cols == 0 {
        return;
    }

    // LL size = ceil halves (matches forward)
    let low_r = (rows + 1) >> 1;
    let low_c = (cols + 1) >> 1;

    // First undo the deeper levels inside LL (if any)
    if levels > 1 && !(low_r <= 1 && low_c <= 1) {
        let mut ll_sub = matrix.slice(s![0..low_r, 0..low_c]).to_owned();
        inv_txfm2d_levels_inplace(&mut ll_sub, levels - 1);
        matrix.slice_mut(s![0..low_r, 0..low_c]).assign(&ll_sub);
    }

    // Then undo this level (always attempt; will no-op if cols < 2)
    inv_txfm2d_inplace(matrix);
}

/// --- helpers: row / col passes on a mutable 2D view ---
fn fwd_rows_inplace(mut ll: ArrayViewMut2<i32>) {
    let (rows, cols) = ll.dim();
    if cols < 2 { return; }
    for r in 0..rows {
        let row = ll.row_mut(r);
        fwd_txfm_1d_inplace(row);
    }
}

fn inv_rows_inplace(mut ll: ArrayViewMut2<i32>) {
    let (rows, cols) = ll.dim();
    if cols < 2 { return; }
    for r in 0..rows {
        let row = ll.row_mut(r);
        inv_txfm_1d_inplace(row);
    }
}

fn fwd_cols_inplace(mut ll: ArrayViewMut2<i32>) {
    let (rows, cols) = ll.dim();
    if rows < 2 { return; }

    let mut col_scratch = Array1::<i32>::zeros(rows);
    for c in 0..cols {
        // gather column
        for r in 0..rows {
            col_scratch[r] = ll[(r, c)];
        }
        // transform 1D
        fwd_txfm_1d_inplace(col_scratch.view_mut());
        // scatter back
        for r in 0..rows {
            ll[(r, c)] = col_scratch[r];
        }
    }
}

fn inv_cols_inplace(mut ll: ArrayViewMut2<i32>) {
    let (rows, cols) = ll.dim();
    if rows < 2 { return; }

    let mut col_scratch = Array1::<i32>::zeros(rows);
    for c in 0..cols {
        // gather column
        for r in 0..rows {
            col_scratch[r] = ll[(r, c)];
        }
        // inverse 1D
        inv_txfm_1d_inplace(col_scratch.view_mut());
        // scatter back
        for r in 0..rows {
            ll[(r, c)] = col_scratch[r];
        }
    }
}

/// Forward multi-level 2D 5/3 on a **view**, with anisotropic level counts.
/// - `lx`: number of horizontal (row-wise) levels to apply
/// - `ly`: number of vertical (column-wise) levels to apply
///
/// Order: at each "level" we apply rows if `lx>0`, then columns if `ly>0`,
/// then recurse into the LL region whose size shrinks along the axes we just transformed.
pub fn fwd_txfm2d_levels_view_xy(mut ll: ArrayViewMut2<i32>, lx: usize, ly: usize) {
    let (rows, cols) = ll.dim();
    if rows == 0 || cols == 0 || (lx == 0 && ly == 0) {
        return;
    }

    // --- do this level ---
    if lx > 0 { fwd_rows_inplace(ll.view_mut()); }
    if ly > 0 { fwd_cols_inplace(ll.view_mut()); }

    // --- compute LL size based on which axes we transformed ---
    let low_c = if lx > 0 { (cols + 1) >> 1 } else { cols };
    let low_r = if ly > 0 { (rows + 1) >> 1 } else { rows };

    // --- recurse into LL if we still have work to do ---
    let next_lx = lx.saturating_sub((lx > 0) as usize);
    let next_ly = ly.saturating_sub((ly > 0) as usize);

    if next_lx > 0 || next_ly > 0 {
        // guard against degenerate subview (still safe, 1D passes will no-op)
        let ll_sub = ll.slice_mut(s![0..low_r, 0..low_c]);
        fwd_txfm2d_levels_view_xy(ll_sub, next_lx, next_ly);
    }
}

/// Inverse multi-level 2D 5/3 on a **view**, mirroring `fwd_txfm2d_levels_view_xy`.
/// Applies the inverse in reverse order: first recurse into LL, then undo columns (if any),
/// then undo rows (if any).
pub fn inv_txfm2d_levels_view_xy(mut ll: ArrayViewMut2<i32>, lx: usize, ly: usize) {
    let (rows, cols) = ll.dim();
    if rows == 0 || cols == 0 || (lx == 0 && ly == 0) {
        return;
    }

    // Compute LL size *as created by the forward at this level*.
    let low_c = if lx > 0 { (cols + 1) >> 1 } else { cols };
    let low_r = if ly > 0 { (rows + 1) >> 1 } else { rows };

    // Recurse first (undo deeper levels inside LL)
    let next_lx = lx.saturating_sub((lx > 0) as usize);
    let next_ly = ly.saturating_sub((ly > 0) as usize);
    if next_lx > 0 || next_ly > 0 {
        let ll_sub = ll.slice_mut(s![0..low_r, 0..low_c]);
        inv_txfm2d_levels_view_xy(ll_sub, next_lx, next_ly);
    }

    // Undo this level in reverse order of forward
    if ly > 0 { inv_cols_inplace(ll.view_mut()); }
    if lx > 0 { inv_rows_inplace(ll.view_mut()); }
}




#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array1, Array2};
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Normal, Uniform};

    // -------- helpers --------

    fn seeded_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn roundtrip_1d_inplace(v: &mut Array1<i32>) {
        {
            let view = v.view_mut();
            fwd_txfm_1d_inplace(view);
        }
        {
            let view = v.view_mut();
            inv_txfm_1d_inplace(view);
        }
    }

    fn roundtrip_2d_inplace(m: &mut Array2<i32>) {
        fwd_txfm2d_inplace(m);
        inv_txfm2d_inplace(m);
    }

    fn roundtrip_2d_levels_inplace(m: &mut Array2<i32>, levels: usize) {
        fwd_txfm2d_levels_inplace(m, levels);
        inv_txfm2d_levels_inplace(m, levels);
    }

    // -------- 1D tests --------

    #[test]
    fn one_d_fixed_cases() {
        let mut cases = vec![
            arr1::<i32>(&[]),
            arr1(&[7]),
            arr1(&[3, -1]),
            arr1(&[7, 0, -2]),
            arr1(&[10, 20, 30, 40]),
            arr1(&[1, -2, 3, -4, 5]),
            arr1(&[1000, -1000, 1000, -1000, 1000, -1000, 1000]),
        ];

        for v in cases.iter_mut() {
            let original = v.clone();
            roundtrip_1d_inplace(v);
            assert_eq!(&*v, &original);
        }
    }

    #[test]
    fn one_d_random_uniform() {
        let mut rng = seeded_rng(0x532D_0001);
        // Use unwrap() because in rand_distr the constructor returns Result.
        let dist: Uniform<i32> = Uniform::new_inclusive(-750_000, 750_000).unwrap();

        for n in 0..64 {
            for _ in 0..16 {
                let mut v = Array1::from_iter((0..n).map(|_| dist.sample(&mut rng)));
                let original = v.clone();
                roundtrip_1d_inplace(&mut v);
                assert_eq!(v, original, "1D round trip failed at len={n}");
            }
        }
    }

    #[test]
    fn one_d_random_normal() {
        let mut rng = seeded_rng(0x532D_0002);
        let normal = Normal::<f64>::new(0.0, 10_000.0).unwrap();

        for n in 0..64 {
            for _ in 0..16 {
                let mut v =
                    Array1::from_iter((0..n).map(|_| normal.sample(&mut rng).round() as i32));
                let original = v.clone();
                roundtrip_1d_inplace(&mut v);
                assert_eq!(v, original, "1D (normal) round trip failed at len={n}");
            }
        }
    }

    // -------- 2D tests (single level) --------

    #[test]
    fn two_d_fixed_cases() {
        let mut mats: Vec<Array2<i32>> = vec![
            Array2::<i32>::zeros((0, 0)),
            arr2(&[[1], [2], [3]]), // cols=1 -> no-op
            arr2(&[[1, 2]]),
            arr2(&[[1, 2, 3], [4, 5, 6]]),
            arr2(&[[1, -1, 2], [-2, 3, -3], [4, -4, 5]]),
            arr2(&[
                [1000, -1000, 500, -500],
                [200, -200, 100, -100],
                [7, 8, 9, 10],
            ]),
        ];

        for m in mats.iter_mut() {
            let original = m.clone();
            roundtrip_2d_inplace(m);
            assert_eq!(&*m, &original, "2D round trip failed for dims {:?}", original.dim());
        }
    }

    #[test]
    fn two_d_random_uniform() {
        let mut rng = seeded_rng(0x532D_1001);
        let dist: Uniform<i32> = Uniform::new_inclusive(-500_000, 500_000).unwrap();

        let shapes = [
            (0, 0),
            (1, 1),
            (1, 2),
            (2, 1), // cols < 2 -> no-op
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
            (4, 5),
            (5, 4),
            (7, 7),
            (8, 13),
            (11, 9),
        ];

        for &(r, c) in &shapes {
            for _ in 0..8 {
                let mut m = Array2::from_shape_fn((r, c), |_| dist.sample(&mut rng));
                let original = m.clone();
                roundtrip_2d_inplace(&mut m);
                assert_eq!(m, original, "2D round trip failed for ({r},{c})");
            }
        }
    }

    // -------- multi-level 2D tests --------

    #[test]
    fn two_d_multilevel_uniform() {
        let mut rng = seeded_rng(0x532D_2001);
        let dist: Uniform<i32> = Uniform::new_inclusive(-400_000, 400_000).unwrap();

        let shapes = [(1, 1), (1, 2), (3, 3), (5, 7), (8, 8), (9, 12), (17, 10)];
        let level_options = [0usize, 1, 2, 3, 5, 10];

        for &(r, c) in &shapes {
            for &levels in &level_options {
                for _ in 0..3 {
                    let mut m = Array2::from_shape_fn((r, c), |_| dist.sample(&mut rng));
                    let original = m.clone();
                    roundtrip_2d_levels_inplace(&mut m, levels);
                    assert_eq!(
                        m, original,
                        "2D multi-level round trip failed for ({r},{c}), levels={levels}"
                    );
                }
            }
        }
    }

    #[test]
    fn two_d_multilevel_pathological_thin() {
        // cols=1 => top-level no-op; recursion also early-returns
        let mut m1 = Array2::<i32>::from_shape_vec((10, 1), (0..10).collect()).unwrap();
        let orig1 = m1.clone();
        roundtrip_2d_levels_inplace(&mut m1, 5);
        assert_eq!(m1, orig1);

        // single row, many cols (valid)
        let mut m2 = Array2::<i32>::from_shape_vec((1, 17), (0..17).map(|x| x  - 8).collect()).unwrap();
        let orig2 = m2.clone();
        roundtrip_2d_levels_inplace(&mut m2, 10);
        assert_eq!(m2, orig2);
    }
}

#[cfg(test)]
mod anisotropic_view_tests {
    use super::*;
    use ndarray::{arr2, Array2, s};
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Uniform};

    fn seeded_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    /// Convenience: run fwd+inv on a full matrix using the view-based anisotropic API.
    fn roundtrip_2d_levels_view_xy(m: &mut Array2<i32>, lx: usize, ly: usize) {
        {
            let view = m.view_mut();
            fwd_txfm2d_levels_view_xy(view, lx, ly);
        }
        {
            let view = m.view_mut();
            inv_txfm2d_levels_view_xy(view, lx, ly);
        }
    }

    #[test]
    fn two_d_anisotropic_view_fixed_cases_roundtrip() {
        let mut mats: Vec<Array2<i32>> = vec![
            Array2::<i32>::zeros((0, 0)),
            arr2(&[[1], [2], [3]]), // single column
            arr2(&[[1, 2]]),        // single row
            arr2(&[[1, 2, 3], [4, 5, 6]]),
            arr2(&[[1, -1, 2], [-2, 3, -3], [4, -4, 5]]),
        ];

        // (lx, ly) pairs: none, rows only, cols only, both, and "too many" (should safely no-op beyond limits)
        let level_pairs = &[
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (1, 2),
            (10, 10),
        ];

        for m in mats.iter_mut() {
            for &(lx, ly) in level_pairs {
                let original = m.clone();
                roundtrip_2d_levels_view_xy(m, lx, ly);
                assert_eq!(
                    &*m, &original,
                    "Anisotropic view round-trip failed for dims {:?} with (lx, ly)=({lx},{ly})",
                    original.dim()
                );
            }
        }
    }

    #[test]
    fn two_d_anisotropic_view_random_uniform_roundtrip() {
        let mut rng = seeded_rng(0xA93F_0001);
        let dist: Uniform<i32> = Uniform::new_inclusive(-400_000, 400_000).unwrap();

        let shapes = [
            (0, 0),
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
            (4, 5),
            (5, 4),
            (7, 7),
            (8, 13),
            (9, 12),
        ];

        let levels = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 1),
            (1, 3),
            (5, 5),
        ];

        for &(r, c) in &shapes {
            for &(lx, ly) in &levels {
                for _ in 0..4 {
                    let mut m = Array2::from_shape_fn((r, c), |_| dist.sample(&mut rng));
                    let original = m.clone();
                    roundtrip_2d_levels_view_xy(&mut m, lx, ly);
                    assert_eq!(
                        m, original,
                        "Anisotropic view round-trip failed for ({r},{c}) with (lx, ly)=({lx},{ly})"
                    );
                }
            }
        }
    }

    #[test]
    fn two_d_anisotropic_view_thin_pathological_roundtrip() {
        // Column-only transforms on a tall, single-column matrix
        let mut m1 = Array2::<i32>::from_shape_vec((10, 1), (0..10).collect()).unwrap();
        let orig1 = m1.clone();
        roundtrip_2d_levels_view_xy(&mut m1, /*lx=*/0, /*ly=*/6);
        assert_eq!(m1, orig1, "Column-only anisotropic round-trip failed for (10,1)");

        // Row-only transforms on a single-row, many-columns matrix
        let mut m2 = Array2::<i32>::from_shape_vec((1, 17), (0..17).map(|x| x - 8).collect()).unwrap();
        let orig2 = m2.clone();
        roundtrip_2d_levels_view_xy(&mut m2, /*lx=*/10, /*ly=*/0);
        assert_eq!(m2, orig2, "Row-only anisotropic round-trip failed for (1,17)");

        // Mixed (more vertical than horizontal) on a thin-ish matrix
        let mut m3 = arr2(&[
            [10, -5, 7, 4],
            [3,  2, 1, 0],
            [-1, 6, -2, 8],
        ]);
        let orig3 = m3.clone();
        roundtrip_2d_levels_view_xy(&mut m3, /*lx=*/1, /*ly=*/3);
        assert_eq!(m3, orig3, "Mixed anisotropic round-trip failed for (3,4)");
    }

    #[test]
    fn two_d_anisotropic_view_subregion_roundtrip() {
        // Apply the transform only on a top-left subregion view; the whole matrix must round-trip.
        let mut rng = seeded_rng(0xA93F_1001);
        let dist: Uniform<i32> = Uniform::new_inclusive(-300_000, 300_000).unwrap();

        let (rows, cols) = (9, 12);
        let mut m = Array2::from_shape_fn((rows, cols), |_| dist.sample(&mut rng));
        let original = m.clone();

        let (lr, lc) = (7usize, 9usize); // subregion
        {
            let sub = m.slice_mut(s![0..lr, 0..lc]);
            fwd_txfm2d_levels_view_xy(sub, /*lx=*/3, /*ly=*/2);
        }
        {
            let sub = m.slice_mut(s![0..lr, 0..lc]);
            inv_txfm2d_levels_view_xy(sub, /*lx=*/3, /*ly=*/2);
        }

        assert_eq!(
            m, original,
            "Anisotropic view round-trip failed when applied to subregion (0..{lr}, 0..{lc}) of ({rows},{cols})"
        );
    }
}

