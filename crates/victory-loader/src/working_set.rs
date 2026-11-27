//! A hard ceiling on this process's working set (`VICTORY_WORKING_SET_MB`;
//! unset or 0, the default: none).
//!
//! On a PC with less RAM than the title asks for (16 GB: the Series S
//! profile commits 15 to 18 GB of private memory in the world, the RTX 4090
//! test PC holds 23.5 GB), Windows otherwise takes the pages it needs from
//! everything that runs, the desktop included, and the PC stalls on paging
//! before the game does. With a ceiling, the pages past it are the game's
//! own: Windows trims them to its standby and compressed lists, from where
//! they come back without a disk read while they last. Measured on a 16 GB
//! laptop (docs/laptop-16gb.md): with no ceiling, free RAM ran out before
//! the world loaded; with 4,500 MB the world ran for minutes.
//!
//! The ceiling is set once, before the title starts. Windows can require
//! the process's increase-working-set privilege, which ordinary user
//! accounts typically have.

/// The ceiling asked for, in bytes: `VICTORY_WORKING_SET_MB`, from 1,024 MB
/// (smaller would keep only the layer's own code in memory) up; `None`
/// for none (unset, 0, or not a number).
pub fn asked() -> Option<u64> {
    let spec = std::env::var("VICTORY_WORKING_SET_MB").ok()?;
    let ceiling = parse_mb(&spec);
    if ceiling.is_none() && spec.trim() != "0" {
        eprintln!(
            "VIctory: VICTORY_WORKING_SET_MB={spec}: expected megabytes, 1024 or more (0: none)"
        );
    }
    ceiling
}

/// [`asked`] for a setting: megabytes from 1,024 up, as bytes; `None` for
/// fewer, for a number that is not one, and for one whose bytes would not
/// fit in an address (`usize`) or in `u64`.
pub fn parse_mb(spec: &str) -> Option<u64> {
    let mb: u64 = spec.trim().parse().ok()?;
    if mb < 1024 {
        return None;
    }
    let bytes = mb.checked_mul(1 << 20)?;
    usize::try_from(bytes).ok()?;
    Some(bytes)
}

/// Sets the ceiling [`asked`] for, if any, and says what it did.
#[cfg(windows)]
pub fn apply_asked() {
    use windows_sys::Win32::System::Memory::{
        QUOTA_LIMITS_HARDWS_MAX_ENABLE, QUOTA_LIMITS_HARDWS_MIN_DISABLE, SetProcessWorkingSetSizeEx,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let Some(ceiling) = asked() else {
        return;
    };
    // Request a soft minimum so Windows can trim under RAM pressure.
    const FLOOR: usize = 64 << 20;
    // SAFETY: the pseudo-handle of this process, which needs no closing.
    let set = unsafe {
        SetProcessWorkingSetSizeEx(
            GetCurrentProcess(),
            FLOOR,
            ceiling as usize, // fits: `parse_mb` checks it
            QUOTA_LIMITS_HARDWS_MAX_ENABLE | QUOTA_LIMITS_HARDWS_MIN_DISABLE,
        )
    } != 0;
    let error = if set {
        0
    } else {
        std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
    };
    tracing::info!(
        ceiling_mb = ceiling >> 20,
        set,
        error,
        "working set ceiling (VICTORY_WORKING_SET_MB): past it Windows trims the game's own pages"
    );
    eprintln!(
        "VIctory: working set ceiling {} MB: {}",
        ceiling >> 20,
        if set {
            "set".to_string()
        } else {
            format!("not set (error {error})")
        }
    );
}

#[cfg(test)]
mod tests {
    use super::parse_mb;

    #[test]
    fn the_working_set_ceiling_is_read_in_megabytes_from_1024() {
        assert_eq!(parse_mb("4500"), Some(4500 << 20));
        assert_eq!(parse_mb(" 1024 "), Some(1024 << 20));
        assert_eq!(parse_mb("1023"), None);
        assert_eq!(parse_mb("0"), None);
        assert_eq!(parse_mb("4.5"), None);
        assert_eq!(parse_mb(""), None);
        // Bytes that would not fit: refused, not wrapped around.
        assert_eq!(parse_mb(&u64::MAX.to_string()), None);
        assert_eq!(parse_mb(&((u64::MAX >> 20) + 1).to_string()), None);
        assert_eq!(
            parse_mb(&(u64::MAX >> 20).to_string()),
            Some((u64::MAX >> 20) << 20)
        );
        assert_eq!(parse_mb("18446744073709551616"), None);
    }
}
