pub mod ggml;
mod ops;
pub mod tokenizer;

pub const MAX_DIMS: usize = 4;

/// The size of a chunk in bytes.
/// (512 bits)
const CHUNK_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    // I4,
    // I8,
    // I16,
    // F16,
    F32,
}

pub struct Tensor {
    ty: Type,
    dims: usize,
    shape: [usize; MAX_DIMS],
    /// Stride per dimension in bytes.
    /// For example:
    /// stride[0] = ty.bytes()
    /// stride[1] = stride[0] * shape[0] + padding
    /// stride[i] = stride[i-1] * shape[i-1]
    stride: [usize; MAX_DIMS],

    data: *const u8,
}

impl Type {
    pub const fn bytes(&self) -> usize {
        match self {
            // Type::I4 => 1,
            // Type::I8 => 1,
            // Type::I16 => 2,
            // Type::F16 => 2,
            Type::F32 => 4,
        }
    }

    pub const fn per_chunk(&self) -> usize {
        match self {
            // Type::I4 => 128,
            // Type::I8 => 64,
            // Type::I16 => 32,
            // Type::F16 => 32,
            Type::F32 => 16,
        }
    }
}

/// Round a number up to a multiple of another number, which is a power of two.
fn round_up_to_multiple(n: usize, multiple: usize) -> usize {
    assert!(multiple.is_power_of_two());
    (n + multiple - 1) & !(multiple - 1)
}

impl Tensor {
    pub fn null(ty: Type, shape: &[usize]) -> Tensor {
        let dims = shape.len();
        assert!(dims > 0 && dims <= MAX_DIMS);

        let shape = {
            let mut ashape = [0; MAX_DIMS];
            ashape[..dims].copy_from_slice(shape);
            ashape
        };

        let stride = {
            let mut stride = [0; MAX_DIMS];
            stride[0] = ty.bytes();
            for i in 1..dims {
                // Probably only need to round up for stride[1].
                stride[i] = round_up_to_multiple(stride[i - 1] * shape[i - 1], CHUNK_SIZE);
            }
            stride
        };

        let data = std::ptr::null();

        Tensor {
            ty,
            dims,
            shape,
            stride,
            data,
        }
    }

    pub unsafe fn uninit(ty: Type, shape: &[usize]) -> Tensor {
        let mut tensor = Self::null(ty, shape);

        let length = tensor.stride[tensor.dims - 1] * tensor.shape[tensor.dims - 1];
        tensor.data = Vec::with_capacity(length).as_ptr();

        tensor
    }

    pub fn zeros(ty: Type, shape: &[usize]) -> Tensor {
        let mut tensor = Self::null(ty, shape);

        tensor.data =
            vec![0; tensor.stride[tensor.dims - 1] * tensor.shape[tensor.dims - 1]].as_ptr();

        tensor
    }

    pub fn from_slice1(ty: Type, data: &[f32]) -> Tensor {
        let mut tensor = unsafe { Self::uninit(ty, &[data.len()]) };

        match ty {
            Type::F32 => {
                let length = data.len() * std::mem::size_of::<f32>();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        tensor.data as *mut u8,
                        length,
                    );
                }
            }
        }

        tensor
    }

    pub fn ty(&self) -> Type {
        self.ty
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.dims]
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride[..self.dims]
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.data.is_null() {
            let length = self.stride[self.dims - 1] * self.shape[self.dims - 1];
            unsafe {
                Vec::from_raw_parts(self.data as *mut u8, length, length);
            }
        }
    }
}
