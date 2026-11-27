//! The Windows D3D12 loader reads both exports from the process executable.

use std::ffi::c_char;

use anyhow::{Context, Result, ensure};

#[unsafe(no_mangle)]
static D3D12SDKVersion: u32 = 620;

#[repr(transparent)]
struct SdkPath(*const c_char);

// SAFETY: the exported pointer refers to an immutable static C string.
unsafe impl Sync for SdkPath {}

#[unsafe(no_mangle)]
static D3D12SDKPath: SdkPath = SdkPath(c".\\D3D12\\".as_ptr());

pub fn check_runtime() -> Result<()> {
    let executable = std::env::current_exe().context("locate victory.exe for Agility SDK")?;
    let runtime = executable.with_file_name("D3D12").join("D3D12Core.dll");
    ensure!(
        runtime.is_file(),
        "Agility SDK build requires {} from Microsoft.Direct3D.D3D12 1.620.2 (x64). See Getting started in README.md.",
        runtime.display()
    );
    tracing::info!(path = %runtime.display(), "Agility SDK 620 requested");
    Ok(())
}
