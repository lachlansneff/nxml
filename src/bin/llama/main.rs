use std::{io, path::PathBuf};

use half::f16;
use nxml::{ggml::Ggml, tensor::Tensor, tokenizer::Tokenizer};

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,
}

struct Layer {
    attn_norm: Tensor<f16, 1>,

    wq: Tensor<f16, 2>,
    wk: Tensor<f16, 2>,
    wv: Tensor<f16, 2>,
    wo: Tensor<f16, 2>,

    ffn_norm: Tensor<f16, 1>,

    w1: Tensor<f16, 2>,
    w2: Tensor<f16, 2>,
    w3: Tensor<f16, 2>,
}

struct Model {
    tok_embeddings: Tensor<f16, 2>,
    norm: Tensor<f16, 1>,
    output: Tensor<f16, 2>,

    layers: Vec<Layer>,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let (vocab, mut ggml) = Ggml::load(&args.model)?;

    for name in ggml.vars.keys() {
        println!("{}", name);
    }

    let tokenizer = Tokenizer::new(vocab);

    println!("{:?}", tokenizer.encode("Hello, world!"));

    let prompt = " Building a website can be done in 10 simple steps:";
    let tokens = tokenizer.encode(prompt);

    let model = build_model(ggml);

    Ok(())
}

fn build_model(mut ggml: Ggml) -> Model {
    let mut layers = vec![];
    for i in 0..ggml.hparams.n_layers {
        let layer = Layer {
            attn_norm: ggml
                .vars
                .remove(&format!("layers.{i}.attention.norm.weight"))
                .unwrap()
                .as_tensor_f16()
                .unwrap(),

            wq: ggml
                .vars
                .remove(&format!("layers.{i}.attention.wq.weight"))
                .unwrap()
                .as_tensor_f16()
                .unwrap(),
            wk: ggml
                .vars
                .remove(&format!("layers.{i}.attention.wk.weight"))
                .unwrap()
                .as_tensor_f16()
                .unwrap(),
            wv: ggml
                .vars
                .remove(&format!("layers.{i}.attention.wv.weight"))
                .unwrap()
                .as_tensor_f16()
                .unwrap(),
            wo: ggml
                .vars
                .remove(&format!("layers.{i}.attention.wo.weight"))
                .unwrap()
                .as_tensor_f16()
                .unwrap(),

            ffn_norm: ggml
                .vars
                .remove(&format!("layers.{i}.ffn_norm.weight"))
                .unwrap()
                .as_tensor_f16()
                .unwrap(),

            w1: ggml
                .vars
                .remove(&format!("layers.{i}.feed_forward.w1.weight"))
                .unwrap()
                .as_tensor_f16()
                .unwrap(),
            w2: ggml
                .vars
                .remove(&format!("layers.{i}.feed_forward.w2.weight"))
                .unwrap()
                .as_tensor_f16()
                .unwrap(),
            w3: ggml
                .vars
                .remove(&format!("layers.{i}.feed_forward.w3.weight"))
                .unwrap()
                .as_tensor_f16()
                .unwrap(),
        };

        layers.push(layer);
    }

    Model {
        tok_embeddings: ggml
            .vars
            .remove(&format!("tok_embeddings.weight"))
            .unwrap()
            .as_tensor_f16()
            .unwrap(),
        norm: ggml
            .vars
            .remove(&format!("norm.weight"))
            .unwrap()
            .as_tensor_f16()
            .unwrap(),
        output: ggml
            .vars
            .remove(&format!("output.weight"))
            .unwrap()
            .as_tensor_f16()
            .unwrap(),

        layers,
    }
}
