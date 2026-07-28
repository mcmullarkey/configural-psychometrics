//! # configural
//!
//! Binary matrix validation and column-major packed-bit storage for
//! configural psychometrics.
//!
//! ## What this crate provides
//!
//! - [`BinaryMatrix`] — validates 0/1 input and packs into column-major
//!   `u64` words (V <= 64 variables per column).
//!
//! ## What this crate does NOT provide
//!
//! - Statistical computation (mutual information, G-squared, etc.)
//! - File I/O or serialization
//! - R/C++ bindings
//!
//! ## Storage layout
//!
//! Column-major packed `Vec<u64>`. Column `j` occupies words
//! `[j*nwords, (j+1)*nwords)` where `nwords = (n_rows + 63) / 64`.
//! Bit `(r, c)` lives at word `c*nwords + r/64`, bit `r%64`.

pub mod error;
pub mod matrix;
pub mod stats;

pub use error::ConfiguralError;
pub use matrix::BinaryMatrix;
pub use stats::GTestResult;
