//! Identifies the inputs to one completed shader pretranslation without reading the archives.

use std::path::{Path, PathBuf};

use crate::shader::{MAX_FILE, scan::precompile_entries};

pub(super) fn precompile_stamp(dir: &Path, key: &[u8; 32], paths: &[PathBuf]) -> Option<PathBuf> {
    let mut hash = blake3::Hasher::new();
    hash.update(key);
    for root in paths {
        let root = root.canonicalize().ok()?;
        hash_path(&mut hash, &root);
        // Match FileScan's regular files and size limit. Stable order keeps the
        // marker independent of the order returned by the filesystem.
        for entry in precompile_entries(&root, dir) {
            let entry = entry.ok()?;
            if !entry.file_type().is_file() {
                continue;
            }
            let metadata = entry.metadata().ok()?;
            if metadata.len() > MAX_FILE {
                continue;
            }
            hash_path(&mut hash, entry.path());
            hash.update(&metadata.len().to_le_bytes());
            let modified = metadata
                .modified()
                .ok()?
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?;
            hash.update(&modified.as_nanos().to_le_bytes());
            // An installer may preserve the write timestamp when replacing or
            // modifying an archive. The filesystem also records the change itself.
            #[cfg(windows)]
            hash.update(&change_time(entry.path())?.to_le_bytes());
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                hash.update(&metadata.ctime().to_le_bytes());
                hash.update(&metadata.ctime_nsec().to_le_bytes());
                hash.update(&metadata.ino().to_le_bytes());
            }
        }
    }
    Some(dir.join(format!("precompiled-v2-{}", hash.finalize().to_hex())))
}

fn hash_path(hash: &mut blake3::Hasher, path: &Path) {
    let bytes = path.as_os_str().as_encoded_bytes();
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

#[cfg(windows)]
fn change_time(path: &Path) -> Option<i64> {
    use std::os::windows::{fs::OpenOptionsExt, io::AsRawHandle};
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_BASIC_INFO, FILE_READ_ATTRIBUTES, FileBasicInfo, GetFileInformationByHandleEx,
    };
    let file = std::fs::OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .open(path)
        .ok()?;
    let mut info = FILE_BASIC_INFO::default();
    // SAFETY: a live file handle and a writable FILE_BASIC_INFO of the requested size.
    let ok = unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle(),
            FileBasicInfo,
            (&raw mut info).cast(),
            size_of::<FILE_BASIC_INFO>() as u32,
        )
    };
    (ok != 0 && info.ChangeTime != 0).then_some(info.ChangeTime)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_old_key_or_other_source_cannot_complete_the_current_scan() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        let first = dir.join("first");
        let second = dir.join("second");
        std::fs::write(&first, b"first source").unwrap();
        std::fs::write(&second, b"second source").unwrap();

        let old = precompile_stamp(dir, &[1; 32], std::slice::from_ref(&first)).unwrap();
        std::fs::write(&old, []).unwrap();
        let current = precompile_stamp(dir, &[2; 32], std::slice::from_ref(&first)).unwrap();
        assert!(!current.is_file());
        std::fs::write(&current, []).unwrap();
        assert!(
            precompile_stamp(dir, &[2; 32], std::slice::from_ref(&first))
                .unwrap()
                .is_file()
        );
        assert!(
            !precompile_stamp(dir, &[2; 32], &[second])
                .unwrap()
                .is_file()
        );
        assert!(precompile_stamp(dir, &[2; 32], &[dir.join("missing")]).is_none());
    }

    #[test]
    fn tracks_nested_and_explicit_archives_when_an_update_preserves_size_and_write_time() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("game");
        let sub = root.join("packs");
        std::fs::create_dir_all(&sub).unwrap();
        let file = sub.join("shaders.bin");
        std::fs::write(&file, b"original").unwrap();
        let stamp =
            |path: &Path| precompile_stamp(temp.path(), &[2; 32], &[path.to_owned()]).unwrap();
        let before_dir = stamp(&root);
        let before_file = stamp(&file);
        assert_eq!(stamp(&root), before_dir);
        let write_time = file.metadata().unwrap().modified().unwrap();
        // The change-time clock need not advance between back-to-back writes.
        std::thread::sleep(std::time::Duration::from_millis(30));
        std::fs::write(&file, b"modified").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&file)
            .unwrap()
            .set_modified(write_time)
            .unwrap();
        assert_eq!(file.metadata().unwrap().modified().unwrap(), write_time);
        assert_ne!(stamp(&file), before_file);
        assert_ne!(stamp(&root), before_dir);
    }

    #[test]
    fn nested_files_can_appear_and_disappear_between_launches() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("game");
        let sub = root.join("packs");
        std::fs::create_dir_all(&sub).unwrap();
        let stamp =
            || precompile_stamp(temp.path(), &[2; 32], std::slice::from_ref(&root)).unwrap();
        let empty = stamp();
        let file = sub.join("shaders.bin");
        std::fs::write(&file, b"shader").unwrap();
        assert_ne!(stamp(), empty);
        std::fs::remove_file(file).unwrap();
        assert_eq!(stamp(), empty);
    }

    #[test]
    fn writing_the_cache_inside_the_game_does_not_invalidate_or_feed_the_scan() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("game");
        let output = root.join("derived");
        let cache = output.join("shaders");
        std::fs::create_dir_all(&cache).unwrap();
        let archive = root.join("input.bin");
        std::fs::write(&archive, b"input").unwrap();
        let stamp = || precompile_stamp(&cache, &[2; 32], std::slice::from_ref(&root)).unwrap();
        let before = stamp();
        std::fs::write(&before, []).unwrap();
        std::fs::write(cache.join("compiled"), b"shader output").unwrap();
        std::fs::write(output.join("pipeline"), b"pipeline output").unwrap();
        assert_eq!(stamp(), before);
        let mut scan = crate::shader::scan::FileScan::new(MAX_FILE, false);
        let read: Vec<_> = scan
            .read_entries(&root, precompile_entries(&root, &cache))
            .collect();
        assert_eq!(read, [(archive.canonicalize().unwrap(), b"input".to_vec())]);
    }

    #[cfg(unix)]
    #[test]
    fn explicit_links_select_the_target_and_links_inside_the_directory_are_skipped() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("game");
        std::fs::create_dir(&root).unwrap();
        let archive = root.join("input.bin");
        let outside = temp.path().join("other.bin");
        std::fs::write(&archive, b"input").unwrap();
        std::fs::write(&outside, b"other").unwrap();
        let alias = temp.path().join("game-link");
        let archive_alias = temp.path().join("archive-link");
        symlink(&root, &alias).unwrap();
        symlink(&archive, &archive_alias).unwrap();
        symlink(&outside, root.join("internal-link")).unwrap();
        let cache = temp.path().join("shaders");
        std::fs::create_dir(&cache).unwrap();
        let stamp = |p: &Path| precompile_stamp(&cache, &[2; 32], &[p.to_owned()]).unwrap();
        assert_eq!(stamp(&root), stamp(&alias));
        assert_eq!(stamp(&archive), stamp(&archive_alias));
        for source in [&alias, &archive_alias] {
            let mut scan = crate::shader::scan::FileScan::new(MAX_FILE, false);
            let read: Vec<_> = scan
                .read_entries(source, precompile_entries(source, &cache))
                .collect();
            assert_eq!(read, [(archive.clone(), b"input".to_vec())]);
        }
    }
}
