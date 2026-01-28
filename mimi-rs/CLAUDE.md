# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

mimi-rs is a Rust tensor library implementing neural network operations with a focus on audio processing models. It uses Rust 2024 edition and targets native CPU optimizations.

## Build Commands

```bash
# Build (uses native CPU optimizations via .cargo/config.toml)
cargo build --release

# Run tests
cargo test

# Run a single test
cargo test test_name

# Run examples
cargo run --release --example mimi -- input.wav -o output.wav
cargo run --release --example llama
```

## Architecture

### Core Tensor System

The library uses a generic `Tensor<T, B>` type where:
- `T: WithDType` - the element type (f32, f16, bf16, i64, u8)
- `B: Backend` - the compute backend (currently CPU via `()`)

Key types:
- `CpuTensor<T>` = `Tensor<T, ()>` - convenience alias for CPU tensors
- `WithDTypeF` - trait for float types that support transcendental functions

### Backend Trait (`src/backend.rs`)

The `Backend` trait defines all low-level operations. The CPU implementation is in `src/cpu_backend.rs`. Operations are split into:
- Basic ops: add, mul, copy, fill, transpose
- Float ops (require `WithDTypeF`): softmax, rms_norm, layer_norm, rope, conv1d, etc.
- In-place variants in `src/inplace_ops.rs` with `_` suffix (e.g., `add_`, `matmul_`)

### Neural Network Layers (`src/nn/`)

- `Linear`, `RmsNorm` - basic layers
- `VB` (VarBuilder) - loads weights from safetensors files with automatic dtype conversion

### Models (`src/models/`)

- `llama.rs` - Llama architecture with KV-cache support
- `mimi.rs` - Mimi audio tokenizer (encoder/decoder for audio compression)

### Streaming Support

`mimi.rs` defines streaming primitives:
- `StreamTensor<T, B>` - optional tensor for streaming contexts
- `StreamMask` - batch element mask
- `StreamingModule` trait - step-by-step processing with state

### Shape System (`src/shape.rs`)

Supports dynamic shapes with:
- `D` type for dimension indexing (supports negative indices)
- `ShapeWithOneHole` - allows `()` as placeholder in reshape (e.g., `(3, ())` infers second dim)

## Conventions

- Operations return new tensors; in-place variants have `_` suffix
- Use `cargo clippy` - the codebase has specific clippy configurations
- Tests are in `tests/tensor_tests.rs` for core tensor operations
