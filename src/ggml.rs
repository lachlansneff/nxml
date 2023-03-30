use std::{
    collections::HashMap,
    fs::File,
    io::{self, Read},
    mem,
    path::Path,
};

use bstr::BString;
use half::f16;

use crate::tokenizer::{Token, Vocab};

#[derive(Debug)]
pub enum ScalarType {
    F32 = 0,
    F16 = 1,
}

pub enum Data {
    F32(Vec<f32>),
    F16(Vec<f16>),
}

pub struct Var {
    pub dims: Vec<usize>,
    pub data: Data,
}

#[derive(Debug)]
pub struct HParams {
    pub vocab_size: usize,
    pub dim: usize,
    pub multiple_of: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub scalar_ty: ScalarType,
}

pub struct Ggml {
    pub hparams: HParams,
    pub vars: HashMap<String, Var>,
}

fn read_u32(data: &[u8], offset: usize) -> io::Result<u32> {
    if offset + 4 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unexpected EOF",
        ));
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

impl Ggml {
    pub fn load(p: impl AsRef<Path>) -> io::Result<(Vocab, Self)> {
        let mut f = File::open(p)?;

        const HEADER_LEN: usize = mem::size_of::<u32>() * 9;
        let mut header = [0; HEADER_LEN];
        f.read_exact(&mut header)?;

        let magic = read_u32(&header, 0)?;
        if magic != 0x67676d66 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic"));
        }

        let version = read_u32(&header, 4)?;
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid version",
            ));
        }

        let vocab_size = read_u32(&header, 8)? as usize;
        let dim = read_u32(&header, 12)? as usize;
        let multiple_of = read_u32(&header, 16)? as usize;
        let n_heads = read_u32(&header, 20)? as usize;
        let n_layers = read_u32(&header, 24)? as usize;
        let _unused = read_u32(&header, 28)?;
        let scalar_type = read_u32(&header, 32)? as usize;

        let scalar_type = match scalar_type {
            // 0 => ScalarType::F32,
            1 => ScalarType::F16,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid scalar type in header",
                ))
            }
        };

        let hparams = HParams {
            vocab_size,
            dim,
            multiple_of,
            n_heads,
            n_layers,
            scalar_ty: scalar_type,
        };

        println!("{hparams:#?}");

        let mut vocab = Vocab {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        };

        for i in 0..vocab_size {
            let mut buf = [0; 4];
            f.read_exact(&mut buf)?;
            let len = u32::from_le_bytes(buf) as usize;

            let mut text_buf = vec![0; len];
            f.read_exact(&mut text_buf)?;
            let token = BString::new(text_buf);

            f.read_exact(&mut buf)?;
            let score = f32::from_le_bytes(buf);

            vocab.token_to_id.insert(token.clone(), i);
            vocab.id_to_token.push(Token { token, score });
        }

        let mut vars = HashMap::new();
        // While the file has contents
        let mut var_header = [0; mem::size_of::<u32>() * 3];
        loop {
            match f.read_exact(&mut var_header) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let n_dims = read_u32(&var_header, 0)? as usize;
            let name_len = read_u32(&var_header, 4)? as usize;
            let ftype = read_u32(&var_header, 8)? as usize;

            let mut dims = vec![0; n_dims * mem::size_of::<u32>()];
            f.read_exact(&mut dims)?;
            let dims = dims
                .chunks_exact(mem::size_of::<u32>())
                .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]) as usize)
                .collect::<Vec<_>>();

            let mut name = vec![0; name_len];
            f.read_exact(&mut name)?;
            let name = String::from_utf8(name)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid name"))?;

            println!("loading parameters: \"{name}\" ({dims:?})");

            let element_count = dims.iter().product::<usize>();

            let data = match ftype {
                0 => {
                    let mut data = vec![0.0; element_count];
                    {
                        let mut data_u8 = unsafe {
                            std::slice::from_raw_parts_mut(
                                data.as_mut_ptr() as *mut u8,
                                data.len() * mem::size_of::<f32>(),
                            )
                        };
                        f.read_exact(&mut data_u8)?;
                    }
                    Data::F32(data)
                }
                1 => {
                    let mut data = vec![f16::ZERO; element_count];
                    {
                        let mut data_u8 = unsafe {
                            std::slice::from_raw_parts_mut(
                                data.as_mut_ptr() as *mut u8,
                                data.len() * mem::size_of::<f16>(),
                            )
                        };
                        f.read_exact(&mut data_u8)?;
                    }
                    Data::F16(data)
                }
                _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid ftype")),
            };

            if vars.insert(name, Var { dims, data }).is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "duplicate variable name",
                ));
            }
        }

        Ok((vocab, Self { hparams, vars }))
    }
}
