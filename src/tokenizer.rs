use std::collections::{BinaryHeap, HashMap};

use bstr::{BStr, BString};
use ordered_float::OrderedFloat;

pub struct Token {
    pub token: BString,
    pub score: f32,
}

pub struct Vocab {
    pub token_to_id: HashMap<BString, usize>,
    pub id_to_token: Vec<Token>,
}

struct Symbol {
    start_byte: usize,
    end_byte: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

impl Symbol {
    fn size(&self) -> usize {
        self.end_byte - self.start_byte
    }
}

#[derive(PartialEq)]
struct Bigram {
    left: Option<usize>,
    right: Option<usize>,
    score: f32,
    n: usize,
}

impl Eq for Bigram {}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.score)
            .cmp(&OrderedFloat(other.score))
            .then_with(|| self.left.cmp(&other.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct Tokenizer {
    vocab: Vocab,
}

impl Tokenizer {
    pub fn new(vocab: Vocab) -> Self {
        Self { vocab }
    }

    /// Not sure if this even makes sense
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut output = vec![1];

        let mut symbols = text
            .char_indices()
            .enumerate()
            .map(|(i, (start_byte, c))| {
                let c_len = c.len_utf8();
                let next = if start_byte + c_len == text.len() {
                    None
                } else {
                    Some(i + 1)
                };

                Symbol {
                    start_byte,
                    end_byte: start_byte + c_len,
                    prev: if i == 0 { None } else { Some(i - 1) },
                    next,
                }
            })
            .collect::<Vec<_>>();

        let mut bigrams = BinaryHeap::new();

        for i in 1..symbols.len() {
            self.try_add_bigram(&mut bigrams, text, &symbols, Some(i - 1), Some(i));
        }

        while let Some(bigram) = bigrams.pop() {
            let l = bigram.left.unwrap();
            let r = bigram.right.unwrap();

            if symbols[l].size() == 0
                || symbols[r].size() == 0
                || symbols[l].size() + symbols[r].size() != bigram.n
            {
                continue;
            }

            // Merge the right symbol into the left one.
            symbols[l].end_byte = symbols[r].end_byte;
            symbols[r].end_byte = symbols[r].start_byte;

            // Remove the right symbol from the linked list.
            symbols[l].next = symbols[r].next;
            if let Some(next) = symbols[l].next {
                symbols[next].prev = Some(l);
            }

            self.try_add_bigram(&mut bigrams, text, &symbols, symbols[l].prev, Some(l));
            self.try_add_bigram(&mut bigrams, text, &symbols, Some(l), symbols[l].next);
        }

        let mut i = 0;
        loop {
            let token = &text[symbols[i].start_byte..symbols[i].end_byte];
            if let Some(&id) = self.vocab.token_to_id.get(BStr::new(token)) {
                output.push(id);
            } else {
                for c in token.bytes() {
                    output.push(c as usize + 3);
                }
            }

            if let Some(next) = symbols[i].next {
                i = next;
            } else {
                break;
            }
        }

        for id in output.iter() {
            println!("token: \"{}\"", self.vocab.id_to_token[*id].token);
        }

        output
    }

    fn try_add_bigram(
        &self,
        bigrams: &mut BinaryHeap<Bigram>,
        text: &str,
        symbols: &[Symbol],
        left_sym_idx: Option<usize>,
        right_sym_idx: Option<usize>,
    ) {
        if let (Some(left_sym_idx), Some(right_sym_idx)) = (left_sym_idx, right_sym_idx) {
            if let (Some(left_sym), Some(right_sym)) =
                (symbols.get(left_sym_idx), symbols.get(right_sym_idx))
            {
                let merged_text = &text[left_sym.start_byte..right_sym.end_byte];
                if let Some(score) = self
                    .vocab
                    .token_to_id
                    .get(BStr::new(merged_text))
                    .map(|id| self.vocab.id_to_token[*id].score)
                {
                    bigrams.push(Bigram {
                        left: Some(left_sym_idx),
                        right: Some(right_sym_idx),
                        score,
                        n: left_sym.size() + right_sym.size(),
                    });
                }
            }
        }
    }
}
