use numpy::ndarray::{
    Array1, Array2, ArrayViewMut1, Axis, s
};

/// Forward 1D 5/3 transform (in-place) on a mutable 1D view.
pub fn fwd_txfm_1d_inplace(mut data: ArrayViewMut1<i32>) {
    let n = data.len();
    if n < 2 {
        return;
    }

    let even_len = (n + 1) >> 1; 
    let odd_len = n >> 1;

    // Allocate a scratch array
    let mut scratch = Array1::<i32>::zeros(n);

    // Split the scratch array into (even_part, odd_part)
    {
        let (mut even_part, mut odd_part) = {
            let view = scratch.view_mut();   
            // Split the *view* along Axis(0) at `even_len`.
            view.split_at(Axis(0), even_len)
        };

        // Split data -> (s) = even samples, (d) = odd samples
        for i in 0..even_len {
            even_part[i] = data[2 * i];
        }
        for i in 0..odd_len {
            odd_part[i] = data[2 * i + 1];
        }

        // Predict: d[i] -= (s[i] + s[i+1]) >> 1
        for i in 0..(odd_len-1) {
            let pred = (even_part[i] + even_part[i+1]) >> 1;
            odd_part[i] -= pred;
        }
        odd_part[odd_len-1] -= (even_part[odd_len-1] + even_part[0]) >> 1;

        // Update: s[i] += (d[i] + d[i-1] + 2) >> 2
        for i in 1..even_len {
            let upd = (odd_part[i] + odd_part[i-1] + 2) >> 2;
            even_part[i] += upd;
        }
        even_part[0] += (odd_part[0] + odd_part[even_len-1] + 2) >> 2;
        
    }

    // Write the results from `scratch` back into `data`.
    for i in 0..even_len {
        data[i] = scratch[i];
    }
    for i in 0..odd_len {
        data[even_len + i] = scratch[even_len + i];
    }
}

/// Inverse 1D 5/3 transform (in-place) on a mutable 1D view.
pub fn inv_txfm_1d_inplace(mut data: ArrayViewMut1<i32>) {
    let n = data.len();
    if n < 2 {
        return;
    }

    let half = n >> 1;
    let s_len = half + (n & 1);
    let d_len = half;

    // We'll call the first s_len elements "s_vals", the rest "d_vals"
    let mut scratch = Array1::<i32>::zeros(n);

    // Copy s_vals and d_vals out of data into scratch
    {
        let s_vals = data.slice(s![0..s_len]);
        let d_vals = data.slice(s![s_len..]);

        // s0[i] = s[i] - ((d[i] + d[i-1] + 2) >> 2)
        for i in 1..s_len {
            let corr = (d_vals[i] + d_vals[i-1] + 2) >> 2;
            scratch[i] = s_vals[i] - corr;
        }
        scratch[0] = s_vals[0] - ((d_vals[0] + d_vals[s_len-1] + 2) >> 2);

        // d0[i] = d[i] + ((s0[i] + s0[i+1]) >> 1)
        for i in 0..(d_len-1) {
            let pred = (scratch[i] + scratch[i+1]) >> 1;
            scratch[s_len + i] = d_vals[i] + pred;
        }
        scratch[s_len + d_len-1] = d_vals[d_len-1] + ((scratch[d_len-1] + scratch[0]) >> 1);

    }

    // Interleave back into `data`: even index gets s0, odd index gets d0
    let s0 = scratch.slice(s![0..s_len]);
    let d0 = scratch.slice(s![s_len..]);

    let mut s_i = 0;
    let mut d_i = 0;
    for idx in 0..n {
        if idx & 1 == 0 {
            data[idx] = s0[s_i];
            s_i += 1;
        } else {
            data[idx] = d0[d_i];
            d_i += 1;
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
    // Single-level 2D transform
    fwd_txfm2d_inplace(matrix);

    let (rows, cols) = matrix.dim();
    let half_r = rows >> 1;
    let half_c = cols >> 1;

    // If there's no smaller region, we can't recurse further
    if half_r == 0 || half_c == 0 {
        return;
    }

    // Slice out the top-left quadrant (the LL band)
    let mut ll_view = matrix.slice_mut(s![0..half_r, 0..half_c]);

    // Copy out
    let mut ll_sub = Array2::<i32>::zeros((half_r, half_c));
    ll_sub.assign(&ll_view);  // copy from view

    // Recurse
    fwd_txfm2d_levels_inplace(&mut ll_sub, levels - 1);

    // Copy it back
    ll_view.assign(&ll_sub);
}

/// Inverse multi-level 2D DWT in-place using 5/3 transform.
pub fn inv_txfm2d_levels_inplace(matrix: &mut Array2<i32>, levels: usize) {
    if levels == 0 {
        return;
    }
    let (rows, cols) = matrix.dim();
    let half_r = rows >> 1;
    let half_c = cols >> 1;

    if half_r == 0 || half_c == 0 {
        return;
    }

    // Slice out the LL quadrant
    let mut ll_view = matrix.slice_mut(s![0..half_r, 0..half_c]);

    // Copy to smaller Array2
    let mut ll_sub = Array2::<i32>::zeros((half_r, half_c));
    ll_sub.assign(&ll_view);

    // Recursively invert that submatrix first
    inv_txfm2d_levels_inplace(&mut ll_sub, levels - 1);

    // Copy submatrix back
    ll_view.assign(&ll_sub);

    // Invert at this level
    inv_txfm2d_inplace(matrix);
}
