use std::{io, path::PathBuf};

use nxml;

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let (vocab, model) = nxml::ggml::Ggml::load(&args.model)?;

    for name in model.vars.keys() {
        println!("{}", name);
    }

    let tokenizer = nxml::tokenizer::Tokenizer::new(vocab);

    println!("{:?}", tokenizer.encode("Hello, world!"));

    Ok(())
}
