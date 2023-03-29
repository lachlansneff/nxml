use crate::{Tensor, Type, CHUNK_SIZE};

use std::{
    mem,
    simd::{f32x8, StdFloat},
};

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

pub unsafe fn flash_attn_raw_f16<
    const M: usize,
>(
    scratch: *mut f32,
    

    q: *const f16,
    k: *const f16,
    v: *const f16,
    o: *mut f16,

    n: usize,
    d: usize,
    stride_q1: usize,
    stride_k1: usize,
    stride_v1: usize,
    stride_o1: usize,
) {
    let bc = M / (4 * d);
    let br = bc.min(d);


}
