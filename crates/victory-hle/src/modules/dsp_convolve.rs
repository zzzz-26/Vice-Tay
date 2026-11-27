//! Partitioned frequency-domain convolution for the console's audio DSP,
//! kept host-independent so its arithmetic is covered on every platform.
//!
//! The console hands the impulse response to `XDspActivate` already in its
//! frequency-domain layout: a run of complex (real, imaginary) float pairs,
//! split into filter blocks of `blockFrameCount` pairs each. A stereo IR
//! holds one such run per channel. Each block is the spectrum of one
//! time-domain slice of the filter; a real slice of `block` samples has
//! `block + 1` unique bins of a `2 * block` point FFT, of which the console
//! stores the first `block` (the Nyquist bin is dropped). The missing half
//! is reconstructed by conjugate symmetry before an inverse transform.

const TAU: f32 = std::f32::consts::TAU;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn scale(self, by: f32) -> Self {
        Self {
            re: self.re * by,
            im: self.im * by,
        }
    }
}

/// In-place radix-2 FFT. `inverse` uses the opposite sign and divides by N.
fn fft(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(n.is_power_of_two() && n > 1);
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = if inverse {
            TAU / len as f32
        } else {
            -TAU / len as f32
        };
        let step = Complex::new(angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let mut twiddle = Complex::new(1.0, 0.0);
            for k in 0..len / 2 {
                let even = data[start + k];
                let odd = data[start + k + len / 2].mul(twiddle);
                data[start + k] = even.add(odd);
                data[start + k + len / 2] = even.sub(odd);
                twiddle = twiddle.mul(step);
            }
        }
        len *= 2;
    }
    if inverse {
        let by = 1.0 / n as f32;
        for value in data.iter_mut() {
            *value = value.scale(by);
        }
    }
}

/// One stream's convolution state.
pub(crate) struct Convolver {
    block: usize,
    fft_len: usize,
    channels: usize,
    /// [channel][filter block][spectrum]
    filters: Vec<Vec<Vec<Complex>>>,
    /// Past input spectra, newest first: [channel][history][spectrum].
    history: Vec<Vec<Vec<Complex>>>,
    /// Samples carried from one output block to the next.
    tail: Vec<Vec<f32>>,
}

impl Convolver {
    /// `ir` holds `ir_floats` floats per channel. With `stereo_ir` each
    /// channel has its own filter; otherwise one filter feeds both.
    pub(crate) fn new(
        ir: &[f32],
        ir_floats: usize,
        block: usize,
        channels: usize,
        stereo_ir: bool,
    ) -> Option<Self> {
        if !block.is_power_of_two() || block == 0 || !(1..=2).contains(&channels) {
            return None;
        }
        let fft_len = block * 2;
        let pairs = ir_floats / 2;
        let mut blocks = pairs / block;
        if blocks == 0 || ir.len() < ir_floats * if stereo_ir { channels } else { 1 } {
            return None;
        }
        // The title's IRs end in one partition that is not a filter: it
        // starts with the word 0xfffface1 (not a number as a float), seen in
        // every IR of build 1.0.2.1, 2,048 floats before `ir_floats` ends.
        // The filter is the partitions before the first such one.
        let ends = |at: usize| ir.get(at).is_some_and(|v| v.to_bits() == 0xffff_ace1 || !v.is_finite());
        if let Some(first) = (0..blocks).find(|&filter| ends(filter * block * 2)) {
            blocks = first;
            if blocks == 0 {
                return None;
            }
        }
        let mut filters = Vec::new();
        for channel in 0..channels {
            let source = if stereo_ir { channel } else { 0 };
            let base = source * ir_floats;
            let mut per_block = Vec::with_capacity(blocks);
            for filter in 0..blocks {
                let at = base + filter * block * 2;
                let stored = &ir[at..at + block * 2];
                let mut spectrum = vec![Complex::default(); fft_len];
                for (bin, pair) in stored.as_chunks::<2>().0.iter().enumerate() {
                    // A stray word that is not a number would make every
                    // output sample infinite: it counts as zero.
                    let clean = |v: f32| if v.is_finite() { v } else { 0.0 };
                    spectrum[bin] = Complex::new(clean(pair[0]), clean(pair[1]));
                }
                // Bin `block` (Nyquist) was not stored; the upper half is
                // the conjugate mirror of the lower half.
                for bin in 1..block {
                    spectrum[fft_len - bin] = spectrum[bin].conj();
                }
                // The title's partitions are windows of 2 * `block` samples
                // a `block` apart (each one's second half is the next one's
                // first: measured on build 1.0.2.1's IRs). The overlap-add
                // below wants `block` samples zero-padded: keep the first
                // half of each window, the next partition holds the rest.
                fft(&mut spectrum, true);
                for sample in &mut spectrum[block..] {
                    *sample = Complex::default();
                }
                for sample in &mut spectrum[..block] {
                    *sample = Complex::new(sample.re, 0.0);
                }
                fft(&mut spectrum, false);
                per_block.push(spectrum);
            }
            filters.push(per_block);
        }
        let history = (0..channels)
            .map(|_| vec![vec![Complex::default(); fft_len]; blocks])
            .collect();
        let tail = vec![vec![0.0; block]; channels];
        Some(Self {
            block,
            fft_len,
            channels,
            filters,
            history,
            tail,
        })
    }

    /// Clears the block history, for the tests.
    #[cfg(test)]
    pub(crate) fn reset(&mut self) {
        for channel in &mut self.history {
            for spectrum in channel {
                spectrum.fill(Complex::default());
            }
        }
        for tail in &mut self.tail {
            tail.fill(0.0);
        }
    }

    /// Convolves one block. `input` and `output` are interleaved unless the
    /// matching deinterleaved flag is set; `gains` is `[begin, end]`, ramped
    /// linearly over the block as the console command specifies.
    // shortcut: the block's FFT scratch is allocated per call; reuse one
    // buffer per stream once the title's block rate makes that worthwhile.
    pub(crate) fn process(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        gains: [f32; 2],
        deinterleaved_input: bool,
        deinterleaved_output: bool,
    ) -> bool {
        let needed = self.block * self.channels;
        if input.len() < needed || output.len() < needed {
            return false;
        }
        for channel in 0..self.channels {
            let mut spectrum = vec![Complex::default(); self.fft_len];
            for (frame, slot) in spectrum.iter_mut().take(self.block).enumerate() {
                let index = if deinterleaved_input {
                    channel * self.block + frame
                } else {
                    frame * self.channels + channel
                };
                *slot = Complex::new(input[index], 0.0);
            }
            fft(&mut spectrum, false);
            let history = &mut self.history[channel];
            history.rotate_right(1);
            history[0] = spectrum;

            let mut sum = vec![Complex::default(); self.fft_len];
            for (past, filter) in history.iter().zip(&self.filters[channel]) {
                for (total, (x, h)) in sum.iter_mut().zip(past.iter().zip(filter)) {
                    *total = total.add(x.mul(*h));
                }
            }
            fft(&mut sum, true);

            let step = (gains[1] - gains[0]) / self.block as f32;
            let (tail, filter) = (&mut self.tail[channel], &mut sum);
            for frame in 0..self.block {
                let wet = filter[frame].re + tail[frame];
                let gain = gains[0] + step * frame as f32;
                let index = if deinterleaved_output {
                    channel * self.block + frame
                } else {
                    frame * self.channels + channel
                };
                output[index] = wet * gain;
            }
            for frame in 0..self.block {
                tail[frame] = filter[self.block + frame].re;
            }
        }
        true
    }
}

/// Builds the console's frequency-domain IR from a time-domain filter, for
/// the tests. `block` complex pairs are stored per filter block.
#[cfg(test)]
pub(crate) fn encode_ir(time: &[f32], block: usize) -> Vec<f32> {
    let blocks = time.len() / block;
    let mut out = Vec::with_capacity(blocks * block * 2);
    for block_index in 0..blocks {
        let slice = &time[block_index * block..(block_index + 1) * block];
        let mut spectrum = vec![Complex::default(); block * 2];
        for (i, &sample) in slice.iter().enumerate() {
            spectrum[i] = Complex::new(sample, 0.0);
        }
        fft(&mut spectrum, false);
        for value in &spectrum[..block] {
            out.push(value.re);
            out.push(value.im);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct time-domain convolution, the reference for the FFT path.
    fn direct(input: &[f32], ir: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; input.len()];
        for (i, &x) in input.iter().enumerate() {
            for (j, &h) in ir.iter().enumerate() {
                if i + j < out.len() {
                    out[i + j] += x * h;
                }
            }
        }
        out
    }

    fn stream(ir: &[f32], input: &[f32], block: usize, channels: usize) -> Vec<f32> {
        let encoded = encode_ir(ir, block);
        let mut convolver =
            Convolver::new(&encoded, encoded.len(), block, channels, false).unwrap();
        let mut output = vec![0.0f32; input.len()];
        for (block_index, chunk) in input.as_chunks::<512>().0.iter().enumerate() {
            assert!(convolver.process(
                &chunk[..block],
                &mut output[block_index * block..(block_index + 1) * block],
                [1.0, 1.0],
                false,
                false,
            ));
        }
        output
    }

    #[test]
    fn fft_round_trips_a_real_signal() {
        let mut values: Vec<Complex> = (0..16)
            .map(|i| Complex::new((i as f32 * 0.3).sin(), 0.0))
            .collect();
        let original = values.clone();
        fft(&mut values, false);
        fft(&mut values, true);
        for (a, b) in values.iter().zip(&original) {
            assert!((a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4);
        }
    }

    #[test]
    fn impulse_response_copies_the_input() {
        let mut input = vec![0.0f32; 1024];
        input[3] = 1.0;
        let mut ir = vec![0.0f32; 512];
        ir[0] = 1.0;
        let got = stream(&ir, &input, 512, 1);
        let expected = direct(&input, &ir);
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-3, "{a} != {b}");
        }
    }

    #[test]
    fn matches_direct_convolution_over_several_blocks() {
        const BLOCK: usize = 512;
        let input: Vec<f32> = (0..BLOCK * 4)
            .map(|i| (i as f32 * 0.07).sin() * 0.6)
            .collect();
        let ir: Vec<f32> = (0..BLOCK * 2)
            .map(|i| (-(i as f32) / 300.0).exp() * (i as f32 * 0.11).sin() * 0.2)
            .collect();
        let got = stream(&ir, &input, BLOCK, 1);
        let expected = direct(&input, &ir);
        let error: f32 = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(error < 0.02, "largest error {error}");
    }

    #[test]
    fn stereo_streams_keep_their_channels() {
        const BLOCK: usize = 512;
        let mut ir_left = vec![0.0f32; BLOCK];
        ir_left[0] = 1.0;
        let mut ir_right = vec![0.0f32; BLOCK];
        ir_right[4] = 0.5;
        let mut encoded = encode_ir(&ir_left, BLOCK);
        encoded.extend(encode_ir(&ir_right, BLOCK));
        // `impulseResponseLengthInFloats` is per channel for a stereo IR.
        let per_channel = encoded.len() / 2;
        let mut convolver = Convolver::new(&encoded, per_channel, BLOCK, 2, true).unwrap();
        let mut input = vec![0.0f32; BLOCK * 2];
        input[0] = 1.0; // left frame 0
        input[1] = 1.0; // right frame 0
        let mut output = vec![0.0f32; BLOCK * 2];
        assert!(convolver.process(&input, &mut output, [1.0, 1.0], false, false));
        assert!((output[0] - 1.0).abs() < 1e-3, "left passed through");
        assert!(output[1].abs() < 1e-3, "right delayed by four frames");
        assert!((output[4 * 2 + 1] - 0.5).abs() < 1e-3, "right echo");
    }

    #[test]
    fn ramped_gains_and_reset_change_the_output() {
        const BLOCK: usize = 512;
        let mut ir = vec![0.0f32; BLOCK];
        ir[0] = 1.0;
        let encoded = encode_ir(&ir, BLOCK);
        let mut convolver = Convolver::new(&encoded, encoded.len(), BLOCK, 1, false).unwrap();
        let input = vec![1.0f32; BLOCK];
        let mut output = vec![0.0f32; BLOCK];
        assert!(convolver.process(&input, &mut output, [0.0, 1.0], false, false));
        assert!(output[0].abs() < 1e-3);
        // The console does not store the Nyquist bin, so a reconstructed
        // impulse is off by a fraction of a percent.
        assert!(
            (output[BLOCK - 1] - 1.0).abs() < 5e-3,
            "gain ramps to one, got {}",
            output[BLOCK - 1]
        );
        convolver.reset();
        output.fill(0.0);
        assert!(convolver.process(&input, &mut output, [1.0, 1.0], false, false));
        assert!((output[0] - 1.0).abs() < 1e-3, "reset clears the history");
    }

    #[test]
    fn rejects_bad_layouts_and_buffers() {
        const BLOCK: usize = 512;
        let encoded = encode_ir(&vec![0.0f32; BLOCK], BLOCK);
        assert!(Convolver::new(&encoded, encoded.len(), 0, 1, false).is_none());
        assert!(Convolver::new(&encoded, encoded.len(), BLOCK, 3, false).is_none());
        assert!(Convolver::new(&encoded, 4, BLOCK, 1, false).is_none());
        let mut convolver = Convolver::new(&encoded, encoded.len(), BLOCK, 1, false).unwrap();
        let short = Vec::new();
        let mut output = Vec::new();
        assert!(!convolver.process(&short, &mut output, [1.0, 1.0], false, false));
    }

    /// The IR as build 1.0.2.1 of the title hands it over: partition `k`
    /// is the spectrum of `time[k * block..k * block + 2 * block]` (2-block
    /// windows a block apart), and one last partition starts with the word
    /// 0xfffface1.
    fn title_ir(time: &[f32], block: usize) -> Vec<f32> {
        let blocks = time.len() / block;
        let mut out = Vec::with_capacity((blocks + 1) * block * 2);
        for block_index in 0..blocks {
            let mut spectrum = vec![Complex::default(); block * 2];
            for (i, slot) in spectrum.iter_mut().enumerate() {
                let sample = time.get(block_index * block + i).copied().unwrap_or(0.0);
                *slot = Complex::new(sample, 0.0);
            }
            fft(&mut spectrum, false);
            for value in &spectrum[..block] {
                out.push(value.re);
                out.push(value.im);
            }
        }
        let mut trailer = vec![0.0f32; block * 2];
        trailer[0] = f32::from_bits(0xffff_ace1);
        out.extend(trailer);
        out
    }

    #[test]
    fn the_title_s_ir_layout_convolves_exactly() {
        const BLOCK: usize = 512;
        let ir: Vec<f32> = (0..BLOCK * 3).map(|i| ((i * 7919) % 97) as f32 / 97.0 - 0.5).collect();
        let input: Vec<f32> = (0..BLOCK * 6).map(|i| ((i * 104_729) % 89) as f32 / 89.0 - 0.5).collect();
        let encoded = title_ir(&ir, BLOCK);
        let mut convolver = Convolver::new(&encoded, encoded.len(), BLOCK, 1, false).unwrap();
        let mut output = vec![0.0f32; input.len()];
        for (block_index, chunk) in input.chunks(BLOCK).enumerate() {
            assert!(convolver.process(
                chunk,
                &mut output[block_index * BLOCK..(block_index + 1) * BLOCK],
                [1.0, 1.0],
                false,
                false,
            ));
        }
        let reference = direct(&input, &ir);
        assert!(output.iter().all(|s| s.is_finite()), "the trailer partition reached the output");
        // As matches_direct_convolution_over_several_blocks: f32 FFTs. Windows
        // used whole would double their overlaps, an error the size of the
        // signal itself.
        let error = output.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(error < 0.02, "largest error {error}");
    }
}
