use half::f16;
use std::ptr;
use crate::tensor::MAX_DIMS;

fn to_strides(shape: [usize; MAX_DIMS]) -> [usize; MAX_DIMS] {
    let mut strides = [1; MAX_DIMS];
    for i in (0..MAX_DIMS - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub unsafe fn scalev_raw_f16(a: *const f16, dst: *mut f16, n: usize, scale: f32) {
    for i in 0..n {
        let a = a.add(i).read();
        let dst = dst.add(i);
        dst.write(f16::from_f32(a.to_f32() * scale));
    }
}

pub unsafe fn dotv_raw_f32(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut sum = 0.0;

    for i in 0..n {
        sum = f32::mul_add(a.add(i).read(), b.add(i).read(), sum);
    }

    sum
}

pub unsafe fn dotv_raw_f16(a: *const f16, b: *const f16, n: usize) -> f32 {
    let mut acc = 0.0;

    for i in 0..n {
        let x = a.add(i).read().to_f32();
        let y = b.add(i).read().to_f32();

        acc = f32::mul_add(x, y, acc);
    }

    acc
}

pub unsafe fn dotv_raw_f16_f32(a: *const f16, b: *const f32, n: usize) -> f32 {
    let mut acc = 0.0;

    for i in 0..n {
        let x = a.add(i).read().to_f32();
        let y = b.add(i).read();

        acc = f32::mul_add(x, y, acc);
    }

    acc
}

// pub unsafe fn get_rows_raw<T, I: Into<usize>>(
//     a: *const T,
//     idxs: *const I,
//     dst: *mut T,
//     n: usize,
//     stride: usize,
//     indices: usize,
// ) {
//     assert!(n <= stride);
//     for i in 0..indices {
//         let idx = idxs.add(i).read();
//         let src = a.add(idx.into() * stride);
//         let dst = dst.add(i * stride);
//         ptr::copy_nonoverlapping(src, dst, n);
//     }
// }

pub unsafe fn rms_norm_f16(a: *const f16, dst: *mut f16, shape: [usize; MAX_DIMS]) {
    let strides = to_strides(shape);

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let av = a.add(strides[0] * i + strides[1] * j + strides[2] * k);
                let dv = dst.add(strides[0] * i + strides[1] * j + strides[2] * k);

                let rms = (dotv_raw_f16(av, av, shape[3]) / shape[3] as f32).sqrt();

                scalev_raw_f16(av, dv, shape[3], 1.0 / rms);
            }
        }
    }
}

/// a_shape: [b1, b0, m, n]
/// bT_shape: [b1, b0, p, n]
/// c_shape: [b1, b0, m, p]
pub unsafe fn generic_dot_f16(
    a: *const f16,
    bt: *const f16,
    c: *mut f16,
    a_shape: [usize; MAX_DIMS],
    bt_shape: [usize; MAX_DIMS],
) {
    // Check batch dimensions.
    assert!(a_shape[0] == bt_shape[0]); // b1
    assert!(a_shape[1] == bt_shape[1]); // b0

    assert!(a_shape[3] == bt_shape[2]); // n

    let a_strides = to_strides(a_shape);
    let b_strides = to_strides(bt_shape);
    let c_strides = to_strides([a_shape[0], a_shape[1], a_shape[2], bt_shape[2]]);

    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            // The top two levels of the matrix are the batch dimensions.

            // 0..m
            for k in 0..a_shape[2] {
                // 0..p
                for l in 0..bt_shape[3] {
                    let a = a.add(i * a_strides[0] + j * a_strides[1] + k * a_strides[2]);
                    let b = bt.add(i * b_strides[0] + j * b_strides[1] + l * b_strides[2]);
                    let c = c.add(i * c_strides[0] + j * c_strides[1] + k * c_strides[2] + l);

                    let x = dotv_raw_f16(a, b, a_shape[3]);
                    c.write(f16::from_f32(x));
                }
            }
        }
    }
}

/// a_shape: [b1, b0, m, n]
/// bT_shape: [b1, b0, p, n]
/// c_shape: [b1, b0, m, p]
pub unsafe fn generic_dot_f32(
    a: *const f32,
    bT: *const f32,
    c: *mut f32,
    a_shape: [usize; MAX_DIMS],
    bT_shape: [usize; MAX_DIMS],
) {
    // Check batch dimensions.
    assert!(a_shape[0] == bT_shape[0]); // b1
    assert!(a_shape[1] == bT_shape[1]); // b0

    assert!(a_shape[3] == bT_shape[2]); // n

    let a_strides = to_strides(a_shape);
    let b_strides = to_strides(bT_shape);
    let c_strides = to_strides([a_shape[0], a_shape[1], a_shape[2], bT_shape[2]]);

    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            // The top two levels of the matrix are the batch dimensions.

            // 0..m
            for k in 0..a_shape[2] {
                // 0..p
                for l in 0..bT_shape[3] {
                    let a = a.add(i * a_strides[0] + j * a_strides[1] + k * a_strides[2]);
                    let b = bT.add(i * b_strides[0] + j * b_strides[1] + l * b_strides[2]);
                    let c = c.add(i * c_strides[0] + j * c_strides[1] + k * c_strides[2] + l);

                    let x = dotv_raw_f32(a, b, a_shape[3]);
                    c.write(x);
                }
            }
        }
    }
}

/// a_shape: [b1, b0, m, n]
/// bT_shape: [b1, b0, p, n]
/// c_shape: [b1, b0, m, p]
pub unsafe fn generic_dot_f32_f16(
    a: *const f32,
    bT: *const f16,
    c: *mut f16,
    a_shape: [usize; MAX_DIMS],
    bT_shape: [usize; MAX_DIMS],
) {
    // Check batch dimensions.
    assert!(a_shape[0] == bT_shape[0]); // b1
    assert!(a_shape[1] == bT_shape[1]); // b0

    assert!(a_shape[3] == bT_shape[2]); // n

    let a_strides = to_strides(a_shape);
    let b_strides = to_strides(bT_shape);
    let c_strides = to_strides([a_shape[0], a_shape[1], a_shape[2], bT_shape[2]]);

    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            // The top two levels of the matrix are the batch dimensions.

            // 0..m
            for k in 0..a_shape[2] {
                // 0..p
                for l in 0..bT_shape[3] {
                    let a = a.add(i * a_strides[0] + j * a_strides[1] + k * a_strides[2]);
                    let b = bT.add(i * b_strides[0] + j * b_strides[1] + l * b_strides[2]);
                    let c = c.add(i * c_strides[0] + j * c_strides[1] + k * c_strides[2] + l);

                    let x = dotv_raw_f16_f32(b, a, a_shape[3]);
                    c.write(f16::from_f32(x));
                }
            }
        }
    }
}

pub unsafe fn silu_raw_f16(a: *const f16, b: *mut f16, shape: [usize; MAX_DIMS]) {
    let strides = to_strides(shape);

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                for l in 0..shape[3] {
                    let a = a.add(i * strides[0] + j * strides[1] + k * strides[2] + l);
                    let b = b.add(i * strides[0] + j * strides[1] + k * strides[2] + l);

                    let x = a.read().to_f32();
                    let y = x * (1.0 / (1.0 + (-x).exp()));

                    b.write(f16::from_f32(y));
                }
            }
        }
    }
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |acc, x| acc.max(*x));

    for y in x.iter_mut() {
        *y = (*y - max).exp();
    }

    let sum: f32 = x.iter().sum();

    for y in x.iter_mut() {
        *y /= sum;
    }
}

pub unsafe fn flash_attn_raw_f16(
    // [D, N]
    q: *const f16,
    // [D, N]
    k: *const f16,
    // [D, N]
    v: *const f16,
    // [D, N]
    o: *mut f16,

    n: usize,
    d: usize,

    stride_q1: usize,
    stride_k1: usize,
    stride_v1: usize,
    stride_o1: usize,
) {
    let scale = 1.0 / (d as f32).sqrt();

    let mut s = vec![0.0; n];

    // Initialize o
    for i in 0..n {
        for j in 0..d {
            *o.add(i * stride_o1 + j) = f16::from_f32(0.0);
        }
    }

    for i in 0..n {
        for j in 0..n {
            s[j] = scale * dotv_raw_f16(q.add(i * stride_q1), k.add(j * stride_k1), d);
        }

        softmax_inplace(&mut s);

        for j in 0..d {
            let x = dotv_raw_f16_f32(v.add(j * stride_v1), s.as_ptr(), n);
            *o.add(i * stride_o1 + j) = f16::from_f32(x);
        }
    }
}

pub unsafe fn repeat<T>(src: *const T, dst: *mut T, src_shape: [usize; MAX_DIMS], dst_shape:[usize; MAX_DIMS]) {
    assert!(dst_shape[0] % src_shape[0] == 0);
    assert!(dst_shape[1] % src_shape[1] == 0);
    assert!(dst_shape[2] % src_shape[2] == 0);
    assert!(dst_shape[3] % src_shape[3] == 0);

    let src_strides = to_strides(src_shape);
    let dst_strides = to_strides(dst_shape);

    for i in 0..dst_shape[0] {
        for j in 0..dst_shape[1] {
            for k in 0..dst_shape[2] {
                let src_i = i % src_shape[0];
                let src_j = j % src_shape[1];
                let src_k = k % src_shape[2];

                for l in 0..(dst_shape[3] / src_shape[3]) {
                    let from = src.add(src_i * src_strides[0] + src_j * src_strides[1] + src_k * src_strides[2]);
                    let to = dst.add(i * dst_strides[0] + j * dst_strides[1] + k * dst_strides[2] + l * src_shape[3]);
                    ptr::copy_nonoverlapping(from, to, src_shape[3]);
                }
            }
        }
    }
}
