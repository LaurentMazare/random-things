use anyhow::Result;
use mimi::nn::var_builder::Path;
use mimi::{CpuDevice, CpuTensor};

/// Streaming 1D convolution with causal padding via state buffer.
pub struct StreamingConv1d {
    weight: CpuTensor<f32>,
    bias: Option<CpuTensor<f32>>,
    stride: usize,
    dilation: usize,
    groups: usize,
    /// Causal padding state: stores the last (kernel_size - 1) * dilation input frames.
    previous: Option<CpuTensor<f32>>,
    pad_len: usize,
}

impl StreamingConv1d {
    pub fn load(
        vb: &Path<CpuDevice>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight: CpuTensor<f32> =
            vb.tensor("conv.weight", (out_channels, in_channels / groups, kernel_size))?;
        let bias = if bias {
            Some(vb.tensor("conv.bias", (out_channels,))?)
        } else {
            None
        };
        let pad_len = (kernel_size - 1) * dilation;
        Ok(Self {
            weight,
            bias,
            stride,
            dilation,
            groups,
            previous: None,
            pad_len,
        })
    }

    /// Forward pass with causal streaming.
    /// Input shape: (batch, channels, time)
    pub fn forward(&mut self, xs: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        let dev = &mimi::CPU;
        // Prepend causal padding state
        let xs = if let Some(prev) = &self.previous {
            CpuTensor::cat(&[prev, xs], 2)?
        } else {
            // First call: pad with zeros
            let dims = xs.dims();
            let padding = CpuTensor::<f32>::zeros((dims[0], dims[1], self.pad_len), dev)?;
            CpuTensor::cat(&[&padding, xs], 2)?
        };

        // Save tail as new state
        let total_len = xs.dims()[2];
        if total_len >= self.pad_len {
            let start = total_len - self.pad_len;
            self.previous = Some(xs.narrow(2, start..total_len)?.contiguous()?);
        }

        // Apply convolution (no padding, causal padding handled by state)
        let out = xs.conv1d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            0, // no padding, we handle it ourselves
            self.dilation,
            self.groups,
        )?;
        Ok(out)
    }

    pub fn reset_state(&mut self) {
        self.previous = None;
    }
}

/// Streaming transposed 1D convolution with overlap-add state.
pub struct StreamingConvTranspose1d {
    weight: CpuTensor<f32>,
    bias: Option<CpuTensor<f32>>,
    stride: usize,
    groups: usize,
    kernel_size: usize,
    /// Overlap-add state: stores the trailing overlap from previous forward.
    partial: Option<CpuTensor<f32>>,
}

impl StreamingConvTranspose1d {
    pub fn load(
        vb: &Path<CpuDevice>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight: CpuTensor<f32> =
            vb.tensor("convtr.weight", (in_channels, out_channels / groups, kernel_size))?;
        let bias = if bias {
            Some(vb.tensor("convtr.bias", (out_channels,))?)
        } else {
            None
        };
        Ok(Self { weight, bias, stride, groups, kernel_size, partial: None })
    }

    /// Forward pass with overlap-add streaming.
    /// Input shape: (batch, channels, time)
    pub fn forward(&mut self, xs: &CpuTensor<f32>) -> Result<CpuTensor<f32>> {
        // Apply transposed convolution
        let mut out = xs.conv_transpose1d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            0, // padding
            0, // output_padding
            self.groups,
        )?;

        let overlap = self.kernel_size - self.stride;

        // Add overlap from previous call
        if let Some(prev) = self.partial.take() {
            let prev_len = prev.dims()[2];
            let out_len = out.dims()[2];
            let add_len = prev_len.min(out_len);

            // Add the overlap portion
            let out_head = out.narrow(2, ..add_len)?.contiguous()?;
            let prev_head = prev.narrow(2, ..add_len)?.contiguous()?;
            let added = out_head.add(&prev_head)?;

            // Reassemble: added + rest of out
            if add_len < out_len {
                let out_tail = out.narrow(2, add_len..out_len)?.contiguous()?;
                out = CpuTensor::cat(&[&added, &out_tail], 2)?;
            } else {
                out = added;
            }

            // If prev was longer than out, keep the remaining
            if prev_len > out_len {
                let extra = prev.narrow(2, out_len..prev_len)?.contiguous()?;
                out = CpuTensor::cat(&[&out, &extra], 2)?;
            }
        }

        // Save the overlap tail as partial state
        let out_len = out.dims()[2];
        if overlap > 0 && out_len > overlap {
            let keep_len = out_len - overlap;
            self.partial = Some(out.narrow(2, keep_len..out_len)?.contiguous()?);
            out = out.narrow(2, ..keep_len)?.contiguous()?;
        }

        Ok(out)
    }

    pub fn reset_state(&mut self) {
        self.partial = None;
    }
}
