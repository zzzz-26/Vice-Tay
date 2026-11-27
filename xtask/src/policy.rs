//! Rule #1, statically: the repository only ever contains small text files
//! that we wrote. See docs/asset-policy.md.
//!
//! This is an allowlist, not a blocklist: a file passes only if its name is
//! on the list, it is not too big, it is text, and it does not smuggle data
//! inside text (long hex/base64 runs).

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

// (The translator's one file is 570 KiB of source.)
const MAX_FILE_SIZE: usize = 640 * 1024;
const MAX_ENCODED_RUN: usize = 512;

const ALLOWED_EXTENSIONS: &[&str] = &[
    "rs", "toml", "md", "txt", "lock", "yml", "yaml", "json", "hlsl", "hlsli", "ps1", "cmd", "sh",
];
const ALLOWED_NAMES: &[&str] = &[
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".victory-root",
    "pre-commit",
    "LICENSE",
    "LICENSE-MIT",
    "LICENSE-APACHE",
];
// These text files still go through every content check below.
const ALLOWED_TEXT_PATHS: &[&str] = &[
    "docs/.nojekyll",
    "docs/index.html",
    "linux/wineserver/view-tree.patch",
];

/// Paths explicitly exempted, with a reason. Keep this empty if at all possible.
const EXCEPTIONS: &[(&str, &str)] = &[(
    "crates/victory-formats/src/bcpack-tables.bin",
    "the BCPack format's constant tables: nothing decodes without them and they cannot be derived (owner's decision, 2026-11-23; docs/asset-policy.md)",
)];

/// Signatures of formats that must never be committed.
const FORBIDDEN_MAGIC: &[(&[u8], &str)] = &[
    (b"MZ", "PE executable"),
    (b"DXBC", "shader container"),
    (b"DDS ", "DDS texture"),
    (b"RIFF", "RIFF (audio/video)"),
    (b"BC\xC0\xDE", "LLVM bitcode"),
    (b"Microsoft C/C++ MSF", "PDB"),
    (b"!<arch>", "library archive"),
    (b"PK\x03\x04", "zip"),
    (b"7z\xBC\xAF", "7-zip"),
    (b"MSCF", "cabinet"),
    (b"\xD0\xCF\x11\xE0", "MSI/OLE"),
];

pub fn run(root: &Path, staged: bool) -> Result<()> {
    let files = list_files(root, staged)?;
    let mut violations = Vec::new();
    for path in &files {
        if EXCEPTIONS.iter().any(|(p, _)| p == path) {
            continue;
        }
        let content = if staged {
            let out = Command::new("git")
                .args(["show", &format!(":{path}")])
                .current_dir(root)
                .output()?;
            if !out.status.success() {
                continue; // deleted in the index
            }
            out.stdout
        } else {
            match std::fs::read(root.join(path)) {
                Ok(c) => c,
                Err(_) => continue,
            }
        };
        for problem in check(path, &content) {
            violations.push(format!("{path}: {problem}"));
        }
    }
    if violations.is_empty() {
        println!("asset policy: {} files ok", files.len());
        return Ok(());
    }
    for v in &violations {
        eprintln!("POLICY: {v}");
    }
    bail!(
        "{} asset-policy violation(s). Nothing from the game may be committed; \
         see docs/asset-policy.md",
        violations.len()
    )
}

pub(crate) fn list_files(root: &Path, staged: bool) -> Result<Vec<String>> {
    let git = |args: &[&str]| -> Result<Option<Vec<String>>> {
        let out = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .context("running git")?;
        if !out.status.success() {
            return Ok(None);
        }
        Ok(Some(
            out.stdout
                .split(|&b| b == 0)
                .filter(|s| !s.is_empty())
                .map(|s| String::from_utf8_lossy(s).replace('\\', "/"))
                .collect(),
        ))
    };
    if staged {
        return git(&[
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
        ])?
        .context("--staged needs a git repository");
    }
    if let Some(mut tracked) = git(&[
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ])? {
        tracked.sort();
        tracked.dedup();
        return Ok(tracked);
    }
    // Not a git repository: walk everything except build output.
    Ok(walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !matches!(e.file_name().to_str(), Some("target" | ".git")))
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            e.path()
                .strip_prefix(root)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
        })
        .collect())
}

/// All problems with one file. Pure, for testing.
pub fn check(path: &str, content: &[u8]) -> Vec<String> {
    let mut problems = Vec::new();
    let name = path.rsplit('/').next().unwrap_or(path);
    let ext = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    let allowed = ALLOWED_NAMES.contains(&name)
        || ALLOWED_TEXT_PATHS.contains(&path)
        || ext
            .as_deref()
            .is_some_and(|e| ALLOWED_EXTENSIONS.contains(&e));
    if !allowed {
        problems.push("file type not on the allowlist".to_owned());
    }
    if content.len() > MAX_FILE_SIZE {
        problems.push(format!(
            "{} bytes exceeds the {MAX_FILE_SIZE}-byte limit",
            content.len()
        ));
    }
    for (magic, what) in FORBIDDEN_MAGIC {
        if content.starts_with(magic) {
            problems.push(format!("looks like a {what}"));
        }
    }
    if content.contains(&0) {
        problems.push("binary content (NUL bytes)".to_owned());
    } else if let Some(run) = longest_encoded_run(content)
        && run > MAX_ENCODED_RUN
    {
        problems.push(format!("{run}-character hex/base64 run (embedded data?)"));
    } else {
        let digits = longest_hex_literal_run(content);
        if digits > MAX_ENCODED_RUN {
            problems.push(format!(
                "{digits} hex digits in one run of \\x/0x literals (embedded data?)"
            ));
        }
    }
    problems
}

/// Longest run of characters that could be hex or base64 payload.
fn longest_encoded_run(content: &[u8]) -> Option<usize> {
    let is_payload = |b: &u8| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=');
    content.split(|b| !is_payload(b)).map(<[u8]>::len).max()
}

/// Most hex digits in `\xNN` escapes or `0x` literals with only commas,
/// whitespace or backslashes between them: a byte string or `xxd -i` array,
/// which `longest_encoded_run` sees as many 3- or 4-character runs.
/// shortcut: decimal arrays (`[77, 90, ...]`) still pass, add a literal count if one shows up.
fn longest_hex_literal_run(content: &[u8]) -> usize {
    let (mut i, mut run, mut longest) = (0, 0, 0);
    while i < content.len() {
        match &content[i..] {
            [b'\\' | b'0', b'x' | b'X', rest @ ..] => {
                let len = rest
                    .iter()
                    .take_while(|b| b.is_ascii_hexdigit() || **b == b'_')
                    .count();
                run += rest[..len].iter().filter(|b| b.is_ascii_hexdigit()).count();
                longest = longest.max(run);
                // Skip a type suffix such as `u8` or `u32`.
                let suffix = rest[len..]
                    .iter()
                    .take_while(|b| b.is_ascii_alphanumeric())
                    .count();
                i += 2 + len + suffix;
            }
            [b, ..] if matches!(b, b',' | b'\\') || b.is_ascii_whitespace() => i += 1,
            _ => {
                run = 0;
                i += 1;
            }
        }
    }
    longest
}

#[cfg(test)]
mod tests {
    use super::{MAX_FILE_SIZE, check};

    /// Enforces the policy wherever tests run, even in clones without the hook.
    #[test]
    fn repository_passes_policy() {
        super::run(&crate::workspace_root(), false).unwrap();
    }

    #[test]
    fn accepts_source() {
        assert!(check("crates/x/src/lib.rs", b"fn main() {}\n").is_empty());
        assert!(check(".gitignore", b"/target/\n").is_empty());
        assert!(check(".githooks/pre-commit", b"#!/bin/sh\n").is_empty());
    }

    #[test]
    fn allowed_text_paths_keep_the_content_checks() {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg"/>"#;
        for path in super::ALLOWED_TEXT_PATHS {
            assert!(check(path, b"original documentation").is_empty());
            assert!(!check(path, b"DDS \0").is_empty());
        }
        assert!(!check("docs/charts/profile-frame-rate.svg", svg).is_empty());
        assert!(!check("docs/unrelated.js", b"let data = 0;").is_empty());
    }

    #[test]
    fn rejects_game_like_files() {
        assert!(!check("assets/tex.dds", b"DDS \x7c\0\0\0").is_empty());
        assert!(
            !check("notes.txt", b"MZ\x90\0").is_empty(),
            "magic wins over extension"
        );
        assert!(
            !check("data.bin", b"hello").is_empty(),
            "extension not allowed"
        );
    }

    #[test]
    fn file_size_is_checked_independently_of_embedded_data() {
        let mut content = vec![b' '; MAX_FILE_SIZE];
        assert!(check("a.rs", &content).is_empty());
        content.push(b' ');
        assert_eq!(
            check("a.rs", &content),
            [format!("{} bytes exceeds the {MAX_FILE_SIZE}-byte limit", content.len())]
        );
    }

    #[test]
    fn rejects_data_smuggled_into_text() {
        let blob = format!("const DATA: &str = \"{}\";", "A1".repeat(400));
        assert!(!check("src/blob.rs", blob.as_bytes()).is_empty());
    }

    /// Synthetic bytes, never game data.
    fn synthetic(n: u32) -> impl Iterator<Item = u32> {
        (0..n).map(|i| (i * 37 + 11) % 256)
    }

    #[test]
    fn rejects_data_smuggled_as_an_escaped_byte_string() {
        // 4096 bytes as one unbroken `\xNN` hex run, the shape a byte-string
        // literal of copied data takes.
        let esc: String = synthetic(4096).map(|b| format!("\\x{b:02x}")).collect();
        let src = format!("pub const DATA: &[u8] = b\"{esc}\";\n");
        assert!(
            !check("src/data.rs", src.as_bytes()).is_empty(),
            "{} bytes of \\x-escaped hex passed",
            src.len()
        );
    }

    #[test]
    fn rejects_data_smuggled_as_a_hex_byte_array() {
        // `xxd -i` style: 4096 bytes as hex literals.
        let lits: Vec<String> = synthetic(4096).map(|b| format!("0x{b:02x}")).collect();
        let src = format!("pub const DATA: [u8; 4096] = [{}];\n", lits.join(", "));
        assert!(
            !check("src/data.rs", src.as_bytes()).is_empty(),
            "{} bytes of hex array passed",
            src.len()
        );
        // A short constant table, like the 16-entry ones in the tree, stays legal.
        let table: Vec<String> = synthetic(16).map(|w| format!("0x{w:08x}_u32")).collect();
        let src = format!(
            "const T: [u32; 16] = [\n    {},\n];\n",
            table.join(",\n    ")
        );
        assert!(check("src/table.rs", src.as_bytes()).is_empty());
    }
}
