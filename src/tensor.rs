use crate::ops;
use half::f16;

pub struct Tensor<T, const DIMS: usize> {
    data: Box<[T]>,
    shape: [usize; DIMS],
}

impl<T, const DIMS: usize> Tensor<T, DIMS> {
    pub fn new(data: Vec<T>, shape: [usize; DIMS]) -> Self {
        assert!(shape.iter().product::<usize>() == data.len());
        Self {
            data: data.into_boxed_slice(),
            shape,
        }
    }

    pub fn shape(&self) -> [usize; DIMS] {
        self.shape
    }
}

impl<T> From<Vec<T>> for Tensor<T, 1> {
    fn from(data: Vec<T>) -> Self {
        let len = data.len();
        Self::new(data, [len])
    }
}

impl<const DIMS: usize> Tensor<f16, DIMS> {
    pub fn zeros(shape: [usize; DIMS]) -> Self {
        Self::new(vec![f16::ZERO; shape.iter().product::<usize>()], shape)
    }
}

impl Tensor<f16, 1> {
    pub fn silu(&self) -> Self {
        let mut y = Self::zeros(self.shape);

        unsafe {
            ops::silu_raw_f16(self.data.as_ptr(), y.data.as_mut_ptr(), self.shape[0]);
        }

        y
    }
}

impl Tensor<f16, 2> {
    pub fn get_rows<I: Into<usize>>(&self, idxs: Tensor<I, 1>) -> Self {
        let mut o = Self::zeros([self.shape[0], idxs.shape[0]]);

        unsafe {
            ops::get_rows_raw(
                self.data.as_ptr(),
                idxs.data.as_ptr(),
                o.data.as_mut_ptr(),
                self.shape[0],
                self.shape[1],
                idxs.shape[0],
            );
        }

        o
    }

    pub fn matvec(&self, x: &Tensor<f16, 1>) -> Tensor<f16, 1> {
        assert!(self.shape[1] == x.shape[0]);
        let mut y = Tensor::<f16, 1>::zeros([self.shape[0]]);

        unsafe {
            ops::matvec_raw_f16(
                self.data.as_ptr(),
                x.data.as_ptr(),
                y.data.as_mut_ptr(),
                self.shape[1],
                self.shape[0],
                self.shape[1],
            );
        }

        y
    }

    pub fn flash_attn(&self, q: Self, k: Self) -> Self {
        assert!(self.shape[0] == q.shape[0]); // D
        assert!(self.shape[0] == k.shape[0]);
        assert!(self.shape[1] == q.shape[1]); // N
        assert!(self.shape[1] == k.shape[1]);

        let mut o = Self::zeros(self.shape);

        unsafe {
            ops::flash_attn_raw_f16(
                q.data.as_ptr(),
                k.data.as_ptr(),
                self.data.as_ptr(),
                o.data.as_mut_ptr(),
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
}
