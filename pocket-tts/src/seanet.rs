use anyhow::Result;
use mimi::nn::var_builder::Path;
use mimi::{CpuDevice, CpuTensor};

use crate::conv::{StreamingConv1d, StreamingConvTranspose1d};

/// SEANet residual block: ELU → Conv(k=residual_kernel_size, dilation) → ELU → Conv(k=1) + skip
pub struct SEANetResBlock {
    conv1: StreamingConv1d,
    conv2: StreamingConv1d,
}

impl SEANetResBlock {
    pub fn load(
        vb: &Path<CpuDevice>,
        dim: usize,
        kernel_size: usize,
        dilation: usize,
        compress: usize,
    ) -> Result<Self> {
        let hidden = dim / compress;
        // block.{j}.block.0 = ELU (no params)
        // block.{j}.block.1 = Conv1d(dim, hidden, kernel_size, dilation)
        // block.{j}.block.2 = ELU (no params)
        // block.{j}.block.3 = Conv1d(hidden, dim, 1)
        let conv1 = StreamingConv1d::load(
            &vb.pp("block.1"),
            dim,
            hidden,
            kernel_size,
            1, // stride
            dilation,
            1, // groups
            true,
        )?;
        let conv2 = StreamingConv1d::load(
            &vb.pp("block.3"),
            hidden,
            dim,
            1, // kernel_size
            1, // stride
            1, // dilation
            1, // groups
            true,
        )?;
        Ok(Self { conv1, conv2 })
    }

    pub fn forward(&mut self, xs: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        let h = xs.elu(1.0)?;
        let h = self.conv1.forward(&h)?;
        let h = h.elu(1.0)?;
        let h = self.conv2.forward(&h)?;
        // true_skip: add input directly (no shortcut conv)
        Ok(xs.add(&h)?)
    }

    pub fn reset_state(&mut self) {
        self.conv1.reset_state();
        self.conv2.reset_state();
    }
}

/// SEANet decoder: init_conv → [ELU + ConvTr(upsample) + ResBlocks] × n_ratios → ELU + final_conv
pub struct SEANetDecoder {
    init_conv: StreamingConv1d,
    upsample_blocks: Vec<SEANetUpsampleBlock>,
    final_conv: StreamingConv1d,
}

struct SEANetUpsampleBlock {
    convtr: StreamingConvTranspose1d,
    res_blocks: Vec<SEANetResBlock>,
}

impl SEANetDecoder {
    pub fn load(
        vb: &Path<CpuDevice>,
        dimension: usize,
        n_filters: usize,
        ratios: &[usize],
        n_residual_layers: usize,
        kernel_size: usize,
        residual_kernel_size: usize,
        last_kernel_size: usize,
        dilation_base: usize,
        compress: usize,
    ) -> Result<Self> {
        let n_ratios = ratios.len();
        // Initial number of channels (at the input of the decoder)
        let mult = 1 << n_ratios; // 2^n_ratios
        let init_channels = mult * n_filters; // e.g., 8 * 64 = 512

        // model.0 = Conv1d(dimension, init_channels, kernel_size)
        let init_conv = StreamingConv1d::load(
            &vb.pp("model.0"),
            dimension,
            init_channels,
            kernel_size,
            1,
            1,
            1,
            true,
        )?;

        let mut upsample_blocks = Vec::with_capacity(n_ratios);
        let mut current_channels = init_channels;
        let mut model_idx = 1; // model index counter

        for &ratio in ratios {
            let next_channels = current_channels / 2;

            // model.{idx} = ELU (no params, skip index)
            model_idx += 1;

            // model.{idx} = ConvTranspose1d for upsampling
            let convtr_kernel = ratio * 2;
            let convtr = StreamingConvTranspose1d::load(
                &vb.pp(format!("model.{model_idx}")),
                current_channels,
                next_channels,
                convtr_kernel,
                ratio,
                1, // groups
                false,
            )?;
            model_idx += 1;

            // Residual blocks
            let mut res_blocks = Vec::with_capacity(n_residual_layers);
            for j in 0..n_residual_layers {
                let dilation = dilation_base.pow(j as u32);
                let res_block = SEANetResBlock::load(
                    &vb.pp(format!("model.{model_idx}")),
                    next_channels,
                    residual_kernel_size,
                    dilation,
                    compress,
                )?;
                model_idx += 1;
                res_blocks.push(res_block);
            }

            upsample_blocks.push(SEANetUpsampleBlock { convtr, res_blocks });
            current_channels = next_channels;
        }

        // model.{idx} = ELU (skip)
        model_idx += 1;

        // model.{idx} = Conv1d(final_channels, 1, last_kernel_size)
        let final_conv = StreamingConv1d::load(
            &vb.pp(format!("model.{model_idx}")),
            current_channels,
            1, // output 1 channel (mono audio)
            last_kernel_size,
            1,
            1,
            1,
            true,
        )?;

        Ok(Self { init_conv, upsample_blocks, final_conv })
    }

    /// Input: (batch, dimension, time) → Output: (batch, 1, time * product(ratios))
    pub fn forward(&mut self, xs: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        let mut h = self.init_conv.forward(xs)?;

        for block in &mut self.upsample_blocks {
            h = h.elu(1.0)?;
            h = block.convtr.forward(&h)?;
            for res in &mut block.res_blocks {
                h = res.forward(&h)?;
            }
        }

        h = h.elu(1.0)?;
        h = self.final_conv.forward(&h)?;
        Ok(h)
    }

    pub fn reset_state(&mut self) {
        self.init_conv.reset_state();
        for block in &mut self.upsample_blocks {
            block.convtr.reset_state();
            for res in &mut block.res_blocks {
                res.reset_state();
            }
        }
        self.final_conv.reset_state();
    }
}
