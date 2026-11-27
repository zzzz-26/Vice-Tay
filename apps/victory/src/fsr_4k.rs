//! Series X SDR 4K profiles for the title's PC FidelityFX dispatch.
use anyhow::{Context, Result, ensure};
use victory_loader::{PreparedImage, resolution};

const RETAIL_BUILD: &str = "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";
const SDR_MODE: u32 = 0x0615_c100;
const STOCK_SDR_MODE: [u8; 16] = [0, 15, 0, 0, 112, 8, 0, 0, 60, 0, 0, 0, 1, 0, 0, 0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FsrPreset {
    Quality,
    Balanced,
    Performance,
}

impl FsrPreset {
    /// The profile `VICTORY_FSR_4K` (AMD FSR 4.1.2) or `VICTORY_DLSS_4K`
    /// (NVIDIA DLSS) names, if any.
    pub(super) fn requested(switch: &str) -> Result<Option<Self>> {
        match std::env::var(switch) {
            Err(std::env::VarError::NotPresent) => Ok(None),
            Ok(name) if name == "0" => Ok(None),
            Ok(name) => Self::from_name(&name.trim().to_ascii_lowercase())
                .map(Some)
                .with_context(|| {
                    format!("{switch}={name}: expected quality, balanced, performance or 0")
                }),
            Err(error) => Err(error.into()),
        }
    }

    pub(super) fn from_name(name: &str) -> Option<Self> {
        match name {
            "quality" => Some(Self::Quality),
            "balanced" => Some(Self::Balanced),
            "performance" => Some(Self::Performance),
            _ => None,
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Quality => "quality",
            Self::Balanced => "balanced",
            Self::Performance => "performance",
        }
    }

    // AMD SDK FFX_UPSCALE_QUALITY_MODE values and ratios.
    pub(super) fn mode(self) -> u8 {
        match self {
            Self::Quality => 1,
            Self::Balanced => 2,
            Self::Performance => 3,
        }
    }

    /// NGX `NVSDK_NGX_PerfQuality_Value` for the same profile (MaxPerf 0,
    /// Balanced 1, MaxQuality 2). DLSS renders at the size the title picks
    /// for FSR; this only names the mode its model was tuned for.
    pub(super) fn dlss_quality(self) -> i32 {
        match self {
            Self::Quality => 2,
            Self::Balanced => 1,
            Self::Performance => 0,
        }
    }

    pub(super) fn render_scale(self) -> f32 {
        match self {
            Self::Quality => 2.0 / 3.0,
            Self::Balanced => 1.0 / 1.7,
            Self::Performance => 0.5,
        }
    }

    pub(super) fn render_size(self) -> [u32; 2] {
        [3840.0, 2160.0].map(|size| (size * self.render_scale()) as u32)
    }

    /// The selector and render scale when a PC upscaler serves Series S.
    pub(super) fn series_s_patches(self) -> [(&'static resolution::Site, Vec<u8>); 2] {
        [
            (
                &resolution::FFX_SCALE,
                resolution::ffx_scale_code(self.render_scale()),
            ),
            (&resolution::PICKER, resolution::picker_code(7)),
        ]
    }
}

pub(super) fn prepare_fsr_4k(image: &mut PreparedImage, preset: FsrPreset) -> Result<()> {
    ensure!(
        image.original_hash.to_hex().as_str() == RETAIL_BUILD,
        "FSR at 4K is unavailable for this retail game build"
    );
    ensure!(
        *field::<16>(image, SDR_MODE)? == STOCK_SDR_MODE,
        "Unexpected Series X SDR mode"
    );
    ensure!(
        *field::<4>(image, 0x0615_d95c)? == (2.0f32 / 3.0).to_le_bytes(),
        "Unexpected Series X FSR scene scale"
    );
    ensure!(
        *field::<8>(image, 0x007e_9421)? == [0xb8, 6, 0, 0, 0, 0x0f, 0x45, 0xc1],
        "Unexpected Series X temporal renderer selection"
    );
    ensure!(
        *field::<14>(image, 0x028e_f69a)?
            == [
                0x0f, 0xb6, 0xc0, 0x41, 0x89, 0x86, 0x90, 0, 0, 0, 0x84, 0xc0, 0x75, 0x0a
            ],
        "Unexpected Series X FSR render size selection"
    );
    const NATIVE_LOAD: u32 = 0x028e_f6a8;
    const QUALITY_LOAD: u32 = 0x028e_f6b2;
    if preset != FsrPreset::Quality {
        ensure!(
            *field::<10>(image, NATIVE_LOAD)?
                == [0xc5, 0xfa, 0x10, 0x05, 0x28, 0x78, 0x6d, 0x03, 0xeb, 0x08]
                && *field::<8>(image, QUALITY_LOAD)?
                    == [0xc5, 0xfa, 0x10, 0x05, 0xa2, 0xe2, 0x86, 0x03],
            "Unexpected Series X FSR scale instructions"
        );
    }
    // The original selector has already run its capability check. An
    // explicit FSR session uses renderer 7 every time it queries this.
    *field::<8>(image, 0x007e_9421)? = [0xb8, 7, 0, 0, 0, 0x90, 0x90, 0x90];
    // Record the SDK mode and jump to the scaled-scene branch.
    *field::<14>(image, 0x028e_f69a)? = [
        0xb8,
        preset.mode(),
        0,
        0,
        0,
        0x41,
        0x89,
        0x86,
        0x90,
        0,
        0,
        0,
        0xeb,
        0x0a,
    ];
    if preset != FsrPreset::Quality {
        // The jump makes the native-scale instruction unreachable. Reuse
        // its RIP operand bytes for a private literal, leaving the constants
        // used by other passes unchanged.
        let local_scale = NATIVE_LOAD + 4;
        *field::<4>(image, local_scale)? = preset.render_scale().to_le_bytes();
        let displacement = local_scale as i32 - (QUALITY_LOAD as i32 + 8);
        *field::<4>(image, QUALITY_LOAD + 4)? = displacement.to_le_bytes();
    }
    tracing::info!(
        renderer = 7,
        preset = preset.name(),
        quality = preset.mode(),
        expected_render = ?preset.render_size(),
        output = ?[3840, 2160],
        "Selected Series X FSR with the stock 4K frontend"
    );
    Ok(())
}

fn field<const N: usize>(image: &mut PreparedImage, rva: u32) -> Result<&mut [u8; N]> {
    let offset = image
        .info
        .rva_to_file_offset(rva)
        .context("Display profile field missing")? as usize;
    image
        .bytes
        .get_mut(offset..offset + N)
        .context("Incomplete display profile field")?
        .try_into()
        .context("Invalid display profile field length")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_name_their_render_sizes_and_dlss_modes() {
        for (name, size, dlss) in [
            ("quality", [2560, 1440], 2),
            ("balanced", [2258, 1270], 1),
            ("performance", [1920, 1080], 0),
        ] {
            let preset = FsrPreset::from_name(name).unwrap();
            assert_eq!(preset.name(), name);
            assert_eq!(preset.render_size(), size);
            assert_eq!(preset.dlss_quality(), dlss);
        }
        assert_eq!(FsrPreset::from_name("ultra"), None);
    }

    #[test]
    fn series_s_performance_uses_half_size_and_selects_fidelityfx() {
        let [(scale_site, scale_code), (picker_site, picker_code)] =
            FsrPreset::Performance.series_s_patches();
        assert_eq!(scale_site.at, resolution::FFX_SCALE.at);
        assert_eq!(&scale_code[0x64..0x68], &0.5f32.to_le_bytes());
        assert_eq!(picker_site.at, resolution::PICKER.at);
        assert_eq!(picker_code, [0xb8, 7, 0, 0, 0, 0xc3]);
    }

    #[test]
    fn fsr_presets_leave_the_stock_scanout_and_settings_viewport() {
        let Some(path) = std::env::var_os("VICTORY_DISPLAY_PROFILE_TEST_IMAGE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        for (preset, size) in [
            (FsrPreset::Quality, [2560, 1440]),
            (FsrPreset::Balanced, [2258, 1270]),
            (FsrPreset::Performance, [1920, 1080]),
        ] {
            let mut image = victory_loader::prepare(&bytes).unwrap();
            let scanout = *field::<5>(&mut image, 0x0247_ad4a).unwrap();
            let settings = *field::<8>(&mut image, 0x0580_48d4).unwrap();
            prepare_fsr_4k(&mut image, preset).unwrap();
            assert_eq!(*field::<16>(&mut image, SDR_MODE).unwrap(), STOCK_SDR_MODE);
            assert_eq!(*field::<5>(&mut image, 0x0247_ad4a).unwrap(), scanout);
            assert_eq!(*field::<8>(&mut image, 0x0580_48d4).unwrap(), settings);
            assert_eq!(
                *field::<4>(&mut image, 0x007e_9422).unwrap(),
                7u32.to_le_bytes()
            );
            assert_eq!(
                *field::<4>(&mut image, 0x028e_f69b).unwrap(),
                u32::from(preset.mode()).to_le_bytes()
            );
            assert_eq!(preset.render_size(), size);
            let disp = i32::from_le_bytes(*field::<4>(&mut image, 0x028e_f6b6).unwrap());
            let reference = 0x028e_f6ba_u32.wrapping_add_signed(disp);
            let scale = f32::from_le_bytes(*field::<4>(&mut image, reference).unwrap());
            assert_eq!([3840.0, 2160.0].map(|w| (w * scale) as u32), size);
        }
    }
}
