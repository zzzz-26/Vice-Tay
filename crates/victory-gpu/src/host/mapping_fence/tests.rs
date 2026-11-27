use super::*;
use std::cell::{Cell, RefCell};
use windows::Win32::Foundation::{E_FAIL, E_OUTOFMEMORY, ERROR_INVALID_HANDLE};
use windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_RESET;

#[test]
fn removal_sentinel_is_never_a_completed_mapping() {
    assert!(!reached(4, 3, || panic!("healthy fence")).unwrap());
    assert!(reached(4, 4, || panic!("healthy fence")).unwrap());
    assert_eq!(
        reached(4, u64::MAX, || Err(Error::from_hresult(
            DXGI_ERROR_DEVICE_RESET
        )))
        .unwrap_err()
        .code(),
        DXGI_ERROR_DEVICE_RESET
    );
    assert_eq!(
        reached(4, u64::MAX, || Ok(())).unwrap_err().code(),
        DXGI_ERROR_DEVICE_REMOVED
    );
}

#[test]
fn event_registration_and_native_failed_wait_are_reported() {
    let err = wait_until_complete(
        || Ok(false),
        || Err(Error::from_hresult(E_OUTOFMEMORY)),
        || panic!("no event registered"),
    );
    assert_eq!(err.unwrap_err().code(), E_OUTOFMEMORY);
    // This is a real WAIT_FAILED, not a signalled event (and cannot block).
    let err = wait_until_complete(|| Ok(false), || Ok(()), || wait_event(HANDLE::default()));
    assert_eq!(
        err.unwrap_err().code(),
        windows::core::HRESULT::from_win32(ERROR_INVALID_HANDLE.0)
    );
}

#[test]
fn a_stale_wake_cannot_complete_the_next_mapping() {
    let completed = Cell::new(0);
    let arms = Cell::new(0);
    wait_until_complete(
        || reached(2, completed.get(), || panic!("healthy fence")),
        || {
            arms.set(arms.get() + 1);
            Ok(())
        },
        || {
            completed.set(completed.get() + 1);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(completed.get(), 2);
    assert_eq!(arms.get(), 1);
    let err = wait_until_complete(
        || reached(3, completed.get(), || Ok(())),
        || Ok(()),
        || {
            completed.set(u64::MAX);
            Ok(())
        },
    );
    assert_eq!(err.unwrap_err().code(), DXGI_ERROR_DEVICE_REMOVED);
}

#[test]
fn an_unconfirmed_submission_is_not_reexecuted_and_cannot_cover_new_mappings() {
    let mut progress = Progress::default();
    let trace = RefCell::new(Vec::new());
    let submit = || trace.borrow_mut().push("submit");
    let failed = |value, recovering| {
        assert_eq!(value, 1);
        trace
            .borrow_mut()
            .push(if recovering { "recover" } else { "check" });
        Err(Error::from_hresult(E_FAIL))
    };
    assert_eq!(progress.run(submit, failed).unwrap_err().code(), E_FAIL);
    assert_eq!(progress.run(submit, failed).unwrap_err().code(), E_FAIL);
    progress
        .run(submit, |value, recovering| {
            assert_eq!(value, if recovering { 1 } else { 2 });
            trace.borrow_mut().push(if recovering {
                "confirmed old"
            } else {
                "confirmed new"
            });
            Ok(())
        })
        .unwrap();
    assert_eq!(
        trace.into_inner(),
        [
            "submit",
            "check",
            "recover",
            "confirmed old",
            "submit",
            "confirmed new"
        ]
    );
}

#[test]
fn native_mapping_markers_complete_on_the_host() {
    let Some(host) = super::super::host() else {
        return;
    };
    assert!(
        super::super::wait_resident(),
        "mapping fence test needs residency enabled"
    );
    assert!(host.shared_buffer(64 << 10).is_some());
    assert!(host.shared_buffer(128 << 10).is_some());
    assert!(
        host.log_messages().is_none(),
        "D3D12 reported validation errors"
    );
}
