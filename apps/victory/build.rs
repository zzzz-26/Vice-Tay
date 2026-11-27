fn main() {
    println!("cargo:rerun-if-env-changed=VICTORY_AGILITY_SDK");
    println!("cargo:rustc-check-cfg=cfg(agility_sdk)");
    if std::env::var("VICTORY_AGILITY_SDK").as_deref() != Ok("1") {
        return;
    }

    assert!(
        std::env::var("TARGET").as_deref() == Ok("x86_64-pc-windows-msvc"),
        "VICTORY_AGILITY_SDK=1 requires the x86_64 Windows MSVC target"
    );
    println!("cargo:rustc-cfg=agility_sdk");
    println!("cargo:rustc-link-arg-bin=victory=/EXPORT:D3D12SDKVersion,DATA");
    println!("cargo:rustc-link-arg-bin=victory=/EXPORT:D3D12SDKPath,DATA");
}
