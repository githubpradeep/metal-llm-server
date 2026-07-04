//! Background token printer: decode + stdout off the GPU hot path.
//!
//! The main thread sends raw `u32` token IDs; a worker thread decodes, prints,
//! and flushes each token so TTY output stays incremental.

use std::io::{self, Write};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

use tokenizers::Tokenizer;

enum PrinterMsg {
    Token(u32),
    Done,
}

/// Spawns a background thread that decodes token IDs and prints them,
/// flushing after every token. Drop or call [`finish`] to drain and join.
pub struct TokenPrinter {
    tx: Sender<PrinterMsg>,
    join: Option<JoinHandle<String>>,
}

impl TokenPrinter {
    pub fn spawn(tokenizer: &Tokenizer) -> Self {
        let (tx, rx) = mpsc::channel();
        let tok = tokenizer.clone();
        let join = thread::spawn(move || printer_loop(rx, tok));
        Self {
            tx,
            join: Some(join),
        }
    }

    /// Queue one token for decode + print. Non-blocking for the caller.
    pub fn send(&self, token_id: u32) {
        let _ = self.tx.send(PrinterMsg::Token(token_id));
    }

    /// Signal end-of-stream, flush, join the worker, and return accumulated text.
    pub fn finish(mut self) -> String {
        let _ = self.tx.send(PrinterMsg::Done);
        self.join
            .take()
            .expect("finish called twice")
            .join()
            .unwrap_or_default()
    }
}

fn printer_loop(rx: Receiver<PrinterMsg>, tokenizer: Tokenizer) -> String {
    let mut out = io::stdout();
    let mut accumulated = String::new();

    while let Ok(msg) = rx.recv() {
        match msg {
            PrinterMsg::Token(id) => {
                let piece = tokenizer.decode(&[id], false).unwrap_or_default();
                accumulated.push_str(&piece);
                print!("{}", piece);
                out.flush().unwrap();
            }
            PrinterMsg::Done => {
                out.flush().unwrap();
                break;
            }
        }
    }
    accumulated
}
