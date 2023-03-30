use crate::ops;
use half::f16;
use std::fmt;
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor<T, const DIMS: usize> {
    data: Arc<Box<[T]>>,
    shape: [usize; DIMS],
}

pub trait TensorElement: Copy {
    const ZERO: Self;
}
impl TensorElement for f16 {
    const ZERO: Self = f16::ZERO;
}
impl TensorElement for f32 {
    const ZERO: Self = 0.0;
}
impl TensorElement for u32 {
    const ZERO: Self = 0;
}
impl TensorElement for usize {
    const ZERO: Self = 0;
}

impl<T: fmt::Debug, const DIMS: usize> fmt::Debug for Tensor<T, DIMS> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor{:?}", self.shape)
    }
}

impl<T: TensorElement, const DIMS: usize> Tensor<T, DIMS> {
    pub fn new(data: Vec<T>, shape: [usize; DIMS]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            data: Arc::new(data.into_boxed_slice()),
            shape,
        }
    }

    pub fn zeros(shape: [usize; DIMS]) -> Self {
        Self::new(vec![T::ZERO; shape.iter().product::<usize>()], shape)
    }

    pub fn shape(&self) -> [usize; DIMS] {
        self.shape
    }

    pub fn tile<const DIMS2: usize>(&self, shape: [usize; DIMS2]) -> Tensor<T, DIMS2> {
        assert!(DIMS <= DIMS2);
        todo!()
    }

    pub fn reshape<const DIMS2: usize>(&self, shape: [usize; DIMS2]) -> Tensor<T, DIMS2> {
        assert_eq!(shape.iter().product::<usize>(), self.data.len());
        Tensor {
            data: Arc::clone(&self.data),
            shape,
        }
    }
}

impl<T: TensorElement> From<Vec<T>> for Tensor<T, 1> {
    fn from(data: Vec<T>) -> Self {
        let len = data.len();
        Self::new(data, [len])
    }
}

impl Tensor<f16, 1> {
    pub fn silu(&self) -> Self {
        let mut y = Self::zeros(self.shape);

        unsafe {
            ops::silu_raw_f16(
                self.data.as_ptr(),
                Arc::get_mut(&mut y.data).unwrap().as_mut_ptr(),
                self.shape[0],
            );
        }

        y
    }

    pub fn rms_norm(&self) -> Self {
        let mut o = Self::zeros(self.shape);

        unsafe {
            ops::rms_norm_f16(
                self.data.as_ptr(),
                Arc::get_mut(&mut o.data).unwrap().as_mut_ptr(),
                self.shape[0],
                1,
                self.shape[0],
            );
        }

        o
    }
}

impl<T: TensorElement> Tensor<T, 2> {
    pub fn get_rows<I: Into<usize>>(&self, idxs: Tensor<I, 1>) -> Self {
        let mut o = Self::zeros([self.shape[0], idxs.shape[0]]);

        unsafe {
            ops::get_rows_raw(
                self.data.as_ptr(),
                idxs.data.as_ptr(),
                Arc::get_mut(&mut o.data).unwrap().as_mut_ptr(),
                self.shape[0],
                self.shape[1],
                idxs.shape[0],
            );
        }

        o
    }
}

impl Tensor<f16, 2> {
    pub fn matmul<const DIMS: usize>(&self, x: &Tensor<f16, DIMS>) -> Tensor<f16, DIMS> {
        assert!(DIMS <= 2);
        assert_eq!(self.shape[1], x.shape[0]);

        let mut shape = x.shape;
        shape[0] = self.shape[0];
        let mut y = Tensor::<f16, DIMS>::zeros(shape);

        unsafe {
            ops::matmul_raw_f16(
                self.data.as_ptr(),
                x.data.as_ptr(),
                Arc::get_mut(&mut y.data).unwrap().as_mut_ptr(),
                self.shape[0],
                self.shape[1],
                x.shape.get(1).copied().unwrap_or(1),
                self.shape[0],
                x.shape[0],
            );
        }

        y
    }

    pub fn flash_attn(&self, q: Self, k: Self) -> Self {
        assert_eq!(self.shape[0], q.shape[0]); // D
        assert_eq!(self.shape[0], k.shape[0]);
        assert_eq!(self.shape[1], q.shape[1]); // N
        assert_eq!(self.shape[1], k.shape[1]);

        let mut o = Self::zeros(self.shape);

        unsafe {
            ops::flash_attn_raw_f16(
                q.data.as_ptr(),
                k.data.as_ptr(),
                self.data.as_ptr(),
                Arc::get_mut(&mut o.data).unwrap().as_mut_ptr(),
                self.shape[1],
                self.shape[0],
                self.shape[1],
                k.shape[1],
                self.shape[1],
                o.shape[1],
            );
        }

        o
    }

    pub fn rms_norm(&self) -> Self {
        let mut o = Self::zeros(self.shape);

        unsafe {
            ops::rms_norm_f16(
                self.data.as_ptr(),
                Arc::get_mut(&mut o.data).unwrap().as_mut_ptr(),
                self.shape[0],
                self.shape[1],
                self.shape[0],
            );
        }

        o
    }
}
