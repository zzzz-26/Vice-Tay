//! File discovery shared by `shader-info` and `shaders`.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// Pretranslation and its completion marker must see the same regular files.
/// Resolve an explicitly supplied link, but do not follow links inside a root.
pub(crate) fn precompile_entries(root: &Path, shader_cache: &Path) -> impl Iterator<Item = walkdir::Result<walkdir::DirEntry>> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_owned());
    let cache = shader_cache.canonicalize().unwrap_or_else(|_| shader_cache.to_owned());
    // A cache below the input can also hold new pipelines, images and logs.
    // If the game directory doubles as the cache root, exclude just its shaders.
    let exclude = match cache.parent() {
        Some(parent) if parent != root && parent.starts_with(&root) => parent.to_owned(),
        _ => cache,
    };
    WalkDir::new(root).sort_by_file_name().into_iter().filter_entry(move |entry| entry.path() != exclude)
}

pub(crate) struct FileScan {
    limit: u64,
    verbose: bool,
    files_read: usize,
    walk_errors: usize,
    metadata_errors: usize,
    read_errors: usize,
    over_limit: usize,
}

impl FileScan {
    pub(crate) fn new(limit: u64, verbose: bool) -> Self {
        Self {
            limit,
            verbose,
            files_read: 0,
            walk_errors: 0,
            metadata_errors: 0,
            read_errors: 0,
            over_limit: 0,
        }
    }

    pub(crate) fn files(&mut self, root: &Path) -> impl Iterator<Item = (PathBuf, Vec<u8>)> {
        self.read_entries(root, WalkDir::new(root))
    }

    pub(crate) fn read_entries(
        &mut self,
        root: &Path,
        entries: impl IntoIterator<Item = walkdir::Result<walkdir::DirEntry>>,
    ) -> impl Iterator<Item = (PathBuf, Vec<u8>)> {
        entries.into_iter().filter_map(move |entry| {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    // A failed walk may hide an entire subtree of unknown size.
                    self.walk_errors += 1;
                    self.detail(format_args!("walking {}: {error}", root.display()));
                    return None;
                }
            };
            if !entry.file_type().is_file() {
                return None;
            }
            let data = self.read_file(entry.path(), entry.metadata().map_err(io::Error::from))?;
            Some((entry.into_path(), data))
        })
    }

    pub(crate) fn had_io_errors(&self) -> bool {
        self.walk_errors + self.metadata_errors + self.read_errors != 0
    }

    fn read_file(&mut self, path: &Path, metadata: io::Result<fs::Metadata>) -> Option<Vec<u8>> {
        let size = match metadata {
            Ok(metadata) => metadata.len(),
            Err(error) => {
                self.metadata_errors += 1;
                self.detail(format_args!("reading metadata for {}: {error}", path.display()));
                return None;
            }
        };
        if size > self.limit {
            self.over_limit += 1;
            self.detail(format_args!("skipping large file {} ({size} bytes)", path.display()));
            return None;
        }
        match fs::read(path) {
            Ok(data) => {
                self.files_read += 1;
                Some(data)
            }
            Err(error) => {
                self.read_errors += 1;
                self.detail(format_args!("reading {}: {error}", path.display()));
                None
            }
        }
    }

    fn detail(&self, message: fmt::Arguments<'_>) {
        if self.verbose {
            eprintln!("{message}");
        } else {
            tracing::debug!("{message}");
        }
    }
}

impl fmt::Display for FileScan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let errors = self.walk_errors + self.metadata_errors + self.read_errors;
        let status = match (errors > 0, self.over_limit > 0) {
            (false, false) => "complete",
            (false, true) => "size-limited",
            (true, false) => "incomplete (I/O errors)",
            (true, true) => "incomplete (I/O errors and size limit)",
        };
        write!(
            f,
            "file scan {status}: files read {}, skipped {} (walk errors {}, metadata errors {}, \
             read errors {}, over {} bytes: {})",
            self.files_read, errors + self.over_limit, self.walk_errors, self.metadata_errors,
            self.read_errors, self.limit, self.over_limit,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn summary(scan: &FileScan) -> String {
        let text = scan.to_string();
        println!("{text}");
        text
    }

    #[test]
    fn reads_regular_files_including_empty_files_and_the_limit_itself() {
        let root = tempdir().unwrap();
        let nested = root.path().join("nested");
        fs::create_dir(&nested).unwrap();
        let empty = root.path().join("empty.bin");
        let full = nested.join("full.bin");
        fs::write(&empty, []).unwrap();
        fs::write(&full, b"abcd").unwrap();

        let mut scan = FileScan::new(4, false);
        let files: std::collections::BTreeMap<_, _> = scan.files(root.path()).collect();
        assert_eq!(files[&full], b"abcd");
        assert!(files[&empty].is_empty());
        assert_eq!(
            summary(&scan),
            "file scan complete: files read 2, skipped 0 (walk errors 0, metadata errors 0, \
             read errors 0, over 4 bytes: 0)"
        );
    }

    #[test]
    fn empty_directory_counts_no_files_or_skips() {
        let root = tempdir().unwrap();
        let mut scan = FileScan::new(4, false);
        assert_eq!(scan.files(root.path()).count(), 0);
        assert_eq!(
            summary(&scan),
            "file scan complete: files read 0, skipped 0 (walk errors 0, metadata errors 0, \
             read errors 0, over 4 bytes: 0)"
        );
    }

    #[test]
    fn missing_root_counts_one_walk_error_and_later_roots_are_read() {
        let root = tempdir().unwrap();
        let file = root.path().join("ok.bin");
        fs::write(&file, b"ok").unwrap();
        let mut scan = FileScan::new(4, true);
        assert_eq!(scan.files(&root.path().join("missing")).count(), 0);
        assert_eq!(scan.files(&file).count(), 1);
        assert_eq!(
            summary(&scan),
            "file scan incomplete (I/O errors): files read 1, skipped 1 (walk errors 1, \
             metadata errors 0, read errors 0, over 4 bytes: 0)"
        );
    }

    #[test]
    fn failed_metadata_skips_even_a_readable_file() {
        let root = tempdir().unwrap();
        let file = root.path().join("readable.bin");
        fs::write(&file, b"ok").unwrap();
        let mut scan = FileScan::new(4, true);
        let denied = io::Error::from(io::ErrorKind::PermissionDenied);
        assert!(scan.read_file(&file, Err(denied)).is_none());
        assert_eq!(
            summary(&scan),
            "file scan incomplete (I/O errors): files read 0, skipped 1 (walk errors 0, \
             metadata errors 1, read errors 0, over 4 bytes: 0)"
        );
    }

    #[test]
    fn file_removed_after_metadata_counts_one_read_error() {
        let root = tempdir().unwrap();
        let file = root.path().join("removed.bin");
        fs::write(&file, b"ok").unwrap();
        let metadata = fs::metadata(&file).unwrap();
        fs::remove_file(&file).unwrap();
        let mut scan = FileScan::new(4, true);
        assert!(scan.read_file(&file, Ok(metadata)).is_none());
        assert_eq!(
            summary(&scan),
            "file scan incomplete (I/O errors): files read 0, skipped 1 (walk errors 0, \
             metadata errors 0, read errors 1, over 4 bytes: 0)"
        );
    }

    #[test]
    fn size_skips_are_distinct_from_io_errors_even_in_the_same_scan() {
        let root = tempdir().unwrap();
        fs::write(root.path().join("large.bin"), b"12345").unwrap();
        let mut scan = FileScan::new(4, true);
        assert_eq!(scan.files(root.path()).count(), 0);
        assert_eq!(
            summary(&scan),
            "file scan size-limited: files read 0, skipped 1 (walk errors 0, metadata errors 0, \
             read errors 0, over 4 bytes: 1)"
        );
        assert_eq!(scan.files(&root.path().join("missing")).count(), 0);
        assert_eq!(
            summary(&scan),
            "file scan incomplete (I/O errors and size limit): files read 0, skipped 2 \
             (walk errors 1, metadata errors 0, read errors 0, over 4 bytes: 1)"
        );
    }
}
