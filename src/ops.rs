use crate::{Tensor, Type, CHUNK_SIZE};

use half::f16;

pub unsafe fn dot_raw_f32(n: usize, a: *const f32, b: *const f32) -> f32 {
    let mut sum = 0.0;

    for i in 0..n {
        sum = f32::mul_add(a.add(i).read(), b.add(i).read(), sum);
    }

    sum
}

pub unsafe fn dot_raw_f16(a: *const f16, b: *const f16, n: usize) -> f32 {
    let mut acc = 0.0;

    for i in 0..n {
        let x = a.add(i).read().to_f32();
        let y = b.add(i).read().to_f32();

        acc = f32::mul_add(x, y, acc);
    }

    acc
}

pub unsafe fn dot_raw_f16_f32(a: *const f16, b: *const f32, n: usize) -> f32 {
    let mut acc = 0.0;

    for i in 0..n {
        let x = a.add(i).read().to_f32();
        let y = b.add(i).read();

        acc = f32::mul_add(x, y, acc);
    }

    acc
}

pub unsafe fn linear_raw_f16(
    a: *const f16,
    b: *const f16,
    c: *mut f16,
    n: usize,
    stride_a1: usize,
) {
    assert!(n <= stride_a1);

    for i in 0..n {
        let x = dot_raw_f16(a.add(i * stride_a1), b, n);
        *c.add(i) = f16::from_f32(x);
    }
}

pub unsafe fn silu_raw_f16(n: usize, a: *const f16, b: *mut f16) {
    for i in 0..n {
        let x = a.add(i).read().to_f32();
        let y = x * (1.0 / (1.0 + (-x).exp()));

        *b.add(i) = f16::from_f32(y);
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
            s[j] = scale * dot_raw_f16(q.add(i * stride_q1), k.add(j * stride_k1), d);
        }

        softmax_inplace(&mut s);

        for j in 0..d {
            let x = dot_raw_f16_f32(v.add(j * stride_v1), s.as_ptr(), n);
            *o.add(i * stride_o1 + j) = f16::from_f32(x);
        }
    }
}
