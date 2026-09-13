use std::path::{Component, Path, PathBuf};

/// Resolve an LLM-provided relative path inside a sandbox root.
///
/// The tool arguments (file paths) are decided by the LLM, whose context is
/// the analyzed repository's source — a malicious repo can inject a prompt
/// instructing the agent to read arbitrary files (e.g. `~/.ssh/id_rsa`) and
/// send them to the remote provider. This helper guarantees the resolved
/// path cannot escape `root`:
/// - absolute paths are rejected;
/// - `..` components are rejected;
/// - symlinks are resolved (`canonicalize`) and must stay under `root`.
pub fn resolve_path_within(root: &Path, rel: &str) -> anyhow::Result<PathBuf> {
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        anyhow::bail!("absolute paths are not allowed: {}", rel);
    }
    if rel_path
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        anyhow::bail!("path traversal ('..') is not allowed: {}", rel);
    }

    let joined = root.join(rel_path);

    // If the target exists, verify via symlink-resolved path that it stays
    // inside the root. NOTE: we validate the canonical path but return the
    // *lexical* join, so callers keep the same path shape as before
    // (relative when `root` is relative) — returning a canonical absolute
    // path would leak local paths into docs and into the LLM context.
    if let Ok(canon) = joined.canonicalize() {
        let root_canon = root
            .canonicalize()
            .unwrap_or_else(|_| root.to_path_buf());
        if !canon.starts_with(&root_canon) {
            anyhow::bail!(
                "path escapes project root: {} (resolved to {})",
                rel,
                canon.display()
            );
        }
    }

    Ok(joined)
}

/// Check if a file is a test file
pub fn is_test_file(path: &Path) -> bool {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();

    let path_str = path.to_string_lossy().to_lowercase();

    // Path-based checks (support different path separators)
    if path_str.contains("/test/")
        || path_str.contains("\\test\\")
        || path_str.contains("/tests/")
        || path_str.contains("\\tests\\")
        || path_str.contains("/__tests__/")
        || path_str.contains("\\__tests__\\")
        || path_str.contains("/spec/")
        || path_str.contains("\\spec\\")
        || path_str.contains("/specs/")
        || path_str.contains("\\specs\\")
        || path_str.starts_with("test/")
        || path_str.starts_with("test\\")
        || path_str.starts_with("tests/")
        || path_str.starts_with("tests\\")
        || path_str.starts_with("__tests__/")
        || path_str.starts_with("__tests__\\")
        || path_str.starts_with("spec/")
        || path_str.starts_with("spec\\")
        || path_str.starts_with("specs/")
        || path_str.starts_with("specs\\")
    {
        return true;
    }

    // Filename-based checks
    // Python test files
    if file_name.starts_with("test_") || file_name.ends_with("_test.py") {
        return true;
    }

    // JavaScript/TypeScript test files
    if file_name.ends_with(".test.js")
        || file_name.ends_with(".spec.js")
        || file_name.ends_with(".test.ts")
        || file_name.ends_with(".spec.ts")
        || file_name.ends_with(".test.jsx")
        || file_name.ends_with(".spec.jsx")
        || file_name.ends_with(".test.tsx")
        || file_name.ends_with(".spec.tsx")
    {
        return true;
    }

    // Java test files
    if file_name.ends_with("test.java") || file_name.ends_with("tests.java") {
        return true;
    }

    // C# test files
    if file_name.ends_with("test.cs") 
        || file_name.ends_with("tests.cs")
        || file_name.ends_with(".test.cs")
        || file_name.ends_with(".tests.cs") {
        return true;
    }

    // Rust test files
    if file_name.ends_with("_test.rs") || file_name.ends_with("_tests.rs") {
        return true;
    }

    // Go test files
    if file_name.ends_with("_test.go") {
        return true;
    }

    // C/C++ test files
    if file_name.ends_with("_test.c")
        || file_name.ends_with("_test.cpp")
        || file_name.ends_with("_test.cc")
        || file_name.ends_with("test.c")
        || file_name.ends_with("test.cpp")
        || file_name.ends_with("test.cc")
    {
        return true;
    }

    // Generic test filename patterns
    if file_name.contains("test")
        && (file_name.starts_with("test")
            || file_name.ends_with("test")
            || file_name.contains("_test_")
            || file_name.contains(".test.")
            || file_name.contains("-test-")
            || file_name.contains("-test.")
            || file_name.contains(".spec.")
            || file_name.contains("_spec_")
            || file_name.contains("-spec-")
            || file_name.contains("-spec."))
    {
        return true;
    }

    false
}

/// Check if a directory is a test directory
pub fn is_test_directory(dir_name: &str) -> bool {
    let name_lower = dir_name.to_lowercase();

    // Common test directory names
    matches!(
        name_lower.as_str(),
        "test"
            | "tests"
            | "__tests__"
            | "spec"
            | "specs"
            | "testing"
            | "test_data"
            | "testdata"
            | "fixtures"
            | "e2e"
            | "integration"
            | "unit"
            | "acceptance"
    ) || name_lower.ends_with("_test")
        || name_lower.ends_with("_tests")
        || name_lower.ends_with("-test")
        || name_lower.ends_with("-tests")
}

/// Check if a file path is a binary file
pub fn is_binary_file_path(path: &Path) -> bool {
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        let ext_lower = extension.to_lowercase();
        matches!(
            ext_lower.as_str(),
            // Image files
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "ico" | "svg" | "webp" |
            // Audio files
            "mp3" | "wav" | "flac" | "aac" | "ogg" | "m4a" |
            // Video files
            "mp4" | "avi" | "mkv" | "mov" | "wmv" | "flv" | "webm" |
            // Compressed files
            "zip" | "rar" | "7z" | "tar" | "gz" | "bz2" | "xz" |
            // Executable files
            "exe" | "dll" | "so" | "dylib" | "bin" |
            // Document files
            "pdf" | "doc" | "docx" | "xls" | "xlsx" | "ppt" | "pptx" |
            // Font files
            "ttf" | "otf" | "woff" | "woff2" |
            // Other binary files
            "db" | "sqlite" | "sqlite3" | "dat" | "cache" |
            "archive"
        )
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_path_within_allows_normal_relative_paths() {
        let root = Path::new("/tmp/nonexistent-root");
        let resolved = resolve_path_within(root, "src/main.rs").unwrap();
        assert_eq!(resolved, Path::new("/tmp/nonexistent-root/src/main.rs"));
    }

    #[test]
    fn test_resolve_path_within_rejects_absolute_paths() {
        let root = Path::new("/tmp/root");
        assert!(resolve_path_within(root, "/etc/passwd").is_err());
    }

    #[test]
    fn test_resolve_path_within_rejects_parent_traversal() {
        let root = Path::new("/tmp/root");
        assert!(resolve_path_within(root, "../../etc/passwd").is_err());
        assert!(resolve_path_within(root, "src/../../../.ssh/id_rsa").is_err());
    }

    #[test]
    fn test_resolve_path_within_rejects_symlink_escape() {
        let dir = std::env::temp_dir().join("litho_sandbox_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("project")).unwrap();
        std::fs::write(dir.join("project/secret.txt"), "secret").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(dir.join("project/secret.txt"), dir.join("project/link.txt"))
            .unwrap();

        let root = dir.join("project");
        // Normal file inside root resolves fine
        assert!(resolve_path_within(&root, "secret.txt").is_ok());
        // Root-internal symlink is allowed
        #[cfg(unix)]
        assert!(resolve_path_within(&root, "link.txt").is_ok());

        // Symlink pointing outside root must be rejected
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc", root.join("etc-link")).unwrap();
            assert!(resolve_path_within(&root, "etc-link/passwd").is_err());
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
