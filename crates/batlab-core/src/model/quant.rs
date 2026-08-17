//! File purpose: per-tensor 8-bit affine quantisation of checkpoint weights.
//!
//! The weights inference generates from are the only thing a browser needs, and
//! at f32 they are still four bytes each. Every source of these weights fits a
//! narrow range per tensor (a convolution kernel, a bias vector), so an 8-bit
//! affine code — one `(min, scale)` pair for the whole tensor, one byte per
//! weight — cuts that to a byte plus a negligible header. This module is that
//! code and its inverse; whether the resulting images are worth the loss is a
//! measured question the CLI answers, not one this module assumes.
//!
//! `q = round((w − min) / scale)`, `w' = min + q·scale`, with
//! `scale = (max − min) / 255`. The reconstruction error is bounded by
//! `scale / 2` per weight — half a quantisation step — and nothing accumulates:
//! each tensor is coded against its own extremes.
//!
//! **Dequantisation is a 256-entry lookup, not the formula.** The dataset decode
//! (`dataset_decode.wgsl`) learned this the hard way: written as arithmetic in a
//! shader, `min + b·scale` on Metal disagreed with the CPU by one ULP on ~40 %
//! of byte values, because the backend reassociates float ops. Here the
//! dequantisation runs on the host — the reconstructed f32 are uploaded to GPU
//! buffers with `write_buffer`, never recomputed in a shader — so the drift
//! cannot bite. The table is used anyway: it is the honest shape of the
//! operation (256 possible outputs, computed once, indexed per weight) and it
//! keeps the guarantee true if a shader ever does the dequantising.

/// A tensor coded to 8 bits: the reconstruction is `min + byte·scale`.
///
/// `scale` is `0.0` for a tensor whose values are all equal (or empty), in which
/// case every byte is `0` and every reconstruction is `min` — no division was
/// ever taken.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    pub min: f32,
    pub scale: f32,
    pub bytes: Vec<u8>,
}

/// Code a tensor to 8 bits against its own extremes.
///
/// Each byte is the index of the reconstruction level nearest the original
/// value, i.e. `round((v − min) / scale)` clamped to `[0, 255]` — the same level
/// [`dequantize`]'s table will hand back, so encode and decode cannot disagree
/// on which of the 256 values a byte means.
pub fn quantize(values: &[f32]) -> QuantizedTensor {
    if values.is_empty() {
        return QuantizedTensor {
            min: 0.0,
            scale: 0.0,
            bytes: Vec::new(),
        };
    }

    let mut min = values[0];
    let mut max = values[0];
    for &v in &values[1..] {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let range = max - min;
    if range == 0.0 {
        // A constant tensor reconstructs exactly from `min` alone.
        return QuantizedTensor {
            min,
            scale: 0.0,
            bytes: vec![0u8; values.len()],
        };
    }

    let scale = range / 255.0;
    let bytes = values
        .iter()
        .map(|&v| {
            let level = ((v - min) / scale).round();
            level.clamp(0.0, 255.0) as u8
        })
        .collect();

    QuantizedTensor { min, scale, bytes }
}

/// Reconstruct a tensor coded by [`quantize`], through a 256-entry table.
///
/// The table is `lut[b] = min + b·scale`, built once with a multiply — see the
/// module note on why it is a table and not the inline formula.
pub fn dequantize(min: f32, scale: f32, bytes: &[u8]) -> Vec<f32> {
    let mut lut = [0.0f32; 256];
    for (b, entry) in lut.iter_mut().enumerate() {
        *entry = min + (b as f32) * scale;
    }
    bytes.iter().map(|&b| lut[b as usize]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reconstruction is within half a step of the original — the whole
    /// error budget of an affine 8-bit code, and no more.
    #[test]
    fn a_round_trip_stays_within_half_a_step() {
        let values: Vec<f32> = (0..500)
            .map(|i| ((i as f32) * 0.013).sin() * 0.4 - 0.1)
            .collect();
        let q = quantize(&values);
        let back = dequantize(q.min, q.scale, &q.bytes);
        let half_step = q.scale / 2.0;
        for (got, want) in back.iter().zip(&values) {
            assert!(
                (got - want).abs() <= half_step + 1e-6,
                "{got} vs {want} exceeds half a step ({half_step})"
            );
        }
    }

    /// The extremes are representable exactly-ish: the smallest value maps to
    /// byte 0 and reconstructs as `min`, the largest to byte 255.
    #[test]
    fn the_endpoints_land_on_the_first_and_last_level() {
        let values = vec![-0.7f32, 0.0, 0.3, 0.9, -0.2];
        let q = quantize(&values);
        assert_eq!(q.bytes[0], 0, "the minimum is level 0");
        assert_eq!(q.bytes[3], 255, "the maximum is level 255");
        let back = dequantize(q.min, q.scale, &q.bytes);
        assert!((back[0] - (-0.7)).abs() < 1e-6, "min reconstructs to itself");
        assert!((back[3] - 0.9).abs() < 1e-4, "max reconstructs near itself");
    }

    /// A tensor with no spread costs no precision: it reconstructs exactly, and
    /// no division was taken to get there.
    #[test]
    fn a_constant_tensor_reconstructs_exactly() {
        let values = vec![0.42f32; 16];
        let q = quantize(&values);
        assert_eq!(q.scale, 0.0);
        let back = dequantize(q.min, q.scale, &q.bytes);
        assert!(back.iter().all(|&v| v == 0.42));
    }

    /// An empty tensor round-trips to empty — biasless layers exist.
    #[test]
    fn an_empty_tensor_round_trips_to_empty() {
        let q = quantize(&[]);
        assert!(q.bytes.is_empty());
        assert!(dequantize(q.min, q.scale, &q.bytes).is_empty());
    }

    /// The code is monotone: a larger value never gets a smaller byte, so the
    /// ordering of the weights survives the round trip.
    #[test]
    fn the_code_preserves_order() {
        let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let q = quantize(&values);
        for pair in q.bytes.windows(2) {
            assert!(pair[1] >= pair[0], "quantisation reordered the weights");
        }
    }

    /// Dequantisation through the table equals the affine formula on the host,
    /// value for value — the drift the table guards against is a shader's, and
    /// the two agree where no shader is involved.
    #[test]
    fn the_table_agrees_with_the_affine_formula_on_the_host() {
        let q = quantize(&[-0.5, -0.1, 0.0, 0.25, 0.5]);
        let table = dequantize(q.min, q.scale, &q.bytes);
        for (&byte, &got) in q.bytes.iter().zip(&table) {
            let formula = q.min + (byte as f32) * q.scale;
            assert_eq!(got, formula);
        }
    }
}
