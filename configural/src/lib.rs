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
//! - [`exhaustive`] — exhaustive lattice ascent via Apriori ordered extension.
//! - [`cell_eval`] — cell evaluation + greedy antichain reduction (Wilson/point gates).
//! - [`cell`] — unified Cell representation + adapters from both engines.
//! - [`satisfaction`] — satisfaction matrix primitive (S[p][k] = all elements
//!   in member k present in row p).
//! - [`coverage`] — coverage: share of cases satisfying ≥1 antichain member.
//! - [`diagnostics`] — antichain-level diagnostics (necessity index + summary).
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
pub mod cell;
pub mod cell_eval;
pub mod coverage;
pub mod diagnostics;
pub mod error;
pub mod exhaustive;
pub mod matrix;
pub mod pairmi;
pub mod satisfaction;
pub mod stats;

pub use cell::{from_exhaustive, from_pairmi, Cell, CellMeta, CellSource};
pub use cell_eval::{evaluate_cell, CellOutput, CellRow, Criterion, Detail};
pub use coverage::{coverage, CoverageResult};
pub use error::ConfiguralError;
pub use exhaustive::{AscentResult, EmscStore};
pub use matrix::BinaryMatrix;
pub use pairmi::{DepthStats, EvaluationMode, PairMiEngine, PairMiResultRow};
pub use satisfaction::satisfaction_matrix;
pub use stats::GTestResult;
