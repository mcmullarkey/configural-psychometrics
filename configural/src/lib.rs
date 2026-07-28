//! # configural
//!
//! Binary matrix validation and column-major packed-bit storage for
//! configural psychometrics.
//!
//! ## What this crate provides
//!
//! - [`BinaryMatrix`] — validates 0/1 input and packs into column-major
//!   `u64` words (V <= 64 variables per column).
//! - [`stats`] — statistical primitives: Wilson score lower bound, mutual
//!   information, G-test, chi-squared survival function, normal quantile.
//! - [`pairmi`] — constructive ascent engine: depth-2 combn, depth≥3
//!   expand.grid, canonical-set dedup, first-significant retention.
//!
//! ## What this crate does NOT provide
//!
//! - File I/O or serialization
//! - R/C++ bindings
//!
//! ## Storage layout
//!
//! Column-major packed `Vec<u64>`. Column `j` occupies words
//! `[j*nwords, (j+1)*nwords)` where `nwords = (n_rows + 63) / 64`.
//! Bit `(r, c)` lives at word `c*nwords + r/64`, bit `r%64`.

pub mod bitset;
pub mod error;
pub mod matrix;
pub mod pairmi;
pub mod stats;

pub use error::ConfiguralError;
pub use matrix::BinaryMatrix;
pub use pairmi::{DepthStats, EvaluationMode, PairMiEngine, PairMiResultRow};
pub use stats::GTestResult;
