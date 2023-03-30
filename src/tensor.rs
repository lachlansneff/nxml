use crate::ops;
use half::f16;

pub struct Tensor1<T> {
    data: Box<[T]>,
    shape: [usize; 1],
}

pub struct Tensor2<T> {
    data: Box<[T]>,
    shape: [usize; 2],
    stride: [usize; 1],
}

impl<T> Tensor1<T> {
    pub fn new(data: Vec<T>) -> Self {
        let shape = [data.len()];
        Self {
            data: data.into_boxed_slice(),
            shape,
        }
    }
}

impl Tensor1<f16> {
    pub fn silu(&self) -> Tensor1<f16> {
        let mut y = Tensor1::new(vec![f16::ZERO; self.shape[0]]);

        unsafe {
            ops::silu_raw_f16(self.data.as_ptr(), y.data.as_mut_ptr(), self.shape[0]);
        }

        y
    }
}

impl<T> Tensor2<T> {
    pub fn new(data: Vec<T>, shape: [usize; 2], stride: [usize; 1]) -> Self {
        assert!(shape[0] * stride[0] == data.len());
        assert!(stride[0] >= shape[1]);
        Self {
            data: data.into_boxed_slice(),
            shape,
            stride,
        }
    }
}

impl Tensor2<f16> {
    pub fn matvec(&self, x: &Tensor1<f16>) -> Tensor1<f16> {
        assert!(self.shape[1] == x.shape[0]);
        let mut y = Tensor1::new(vec![f16::ZERO; self.shape[0]]);

        unsafe {
            ops::matvec_raw_f16(
                self.data.as_ptr(),
                x.data.as_ptr(),
                y.data.as_mut_ptr(),
                self.shape[1],
                self.shape[0],
                self.stride[0],
            );
        }

        y
    }

    pub fn flash_attn(&self, q: Tensor2<f16>, k: Tensor2<f16>) -> Tensor2<f16> {
        assert!(self.shape[0] == q.shape[0]); // D
        assert!(self.shape[0] == k.shape[0]);
        assert!(self.shape[1] == q.shape[1]); // N
        assert!(self.shape[1] == k.shape[1]);

        let mut o = Tensor2::new(
            vec![f16::ZERO; self.shape[0] * self.shape[1]],
            self.shape,
            self.stride,
        );

        unsafe {
            ops::flash_attn_raw_f16(
                q.data.as_ptr(),
                k.data.as_ptr(),
                self.data.as_ptr(),
                o.data.as_mut_ptr(),
                self.shape[1],
                self.shape[0],
                self.stride[0],
                k.stride[0],
                self.stride[0],
                o.stride[0],
            );
        }

        todo!()
    }
}
