//! Relative-path link helpers for the agent-facing output set.
//!
//! All source paths in research data are project-root-relative. Pages live under
//! `<OUT_DIR>/.agent-content/...`. To make a source file clickable we compute the
//! relative path from a page's directory to the absolute source path using
//! `pathdiff`. When the two roots are unrelated (e.g. different volumes) and
//! `pathdiff` cannot produce a relative path, we fall back to a non-link code span.

use std::path::{Path, PathBuf};

/// Compute a URL-style relative href from `from_dir` to the absolute `to` path.
///
/// `from_dir` is the directory containing the linking document. Both paths are
/// normalized to absolute form first so `pathdiff` can reason about `..` hops.
/// Returns `None` when no relative path exists (cross-root); callers should fall
/// back to a plain code span in that case.
pub fn relative_href(from_dir: &Path, to: &Path) -> Option<String> {
    let from_abs = absolutize(from_dir);
    let to_abs = absolutize(to);
    let rel = pathdiff::diff_paths(&to_abs, &from_abs)?;
    Some(path_to_href(&rel))
}

/// Build a source link (href) for a project-relative source path, given the
/// absolute project root and the page's directory.
pub fn source_href(project_root: &Path, from_dir: &Path, project_rel_source: &str) -> Option<String> {
    let cleaned = normalize_separators(project_rel_source);
    if cleaned.is_empty() || cleaned == "." {
        return None;
    }
    let abs_source = project_root.join(&cleaned);
    relative_href(from_dir, &abs_source)
}

/// Convert a `Path` to a forward-slash href string, suitable for Markdown links.
///
/// Spaces and parentheses are percent-encoded so the link survives CommonMark
/// parsing (a raw space or `(`/`)` would break the `[](...)` syntax or mis-route
/// the destination).
fn path_to_href(path: &Path) -> String {
    let mut s = normalize_separators(&path.to_string_lossy());
    if s.is_empty() {
        s = ".".to_string();
    }
    s.replace(' ', "%20").replace('(', "%28").replace(')', "%29")
}

/// Normalize backslashes to forward slashes and trim surrounding whitespace.
pub fn normalize_separators(raw: &str) -> String {
    raw.trim().replace('\\', "/")
}

/// Make a path absolute against the current directory without requiring it to
/// exist (so output paths not yet written still resolve).
fn absolutize(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(path),
        Err(_) => path.to_path_buf(),
    }
}

/// Produce a filesystem- and URL-safe slug from an arbitrary name.
///
/// Lowercases, collapses non-alphanumeric runs to a single `-`, trims leading and
/// trailing `-`, and falls back to `item` when the result would be empty. Unlike
/// the research `slugify_id`, this never returns a name that could collide with a
/// reserved directory such as `diagrams`.
pub fn slugify(name: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in name.to_lowercase().chars() {
        if ch.is_alphanumeric() {
            out.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "item".to_string()
    } else {
        trimmed
    }
}

/// Slug uniqueness guard. Names are slugged and de-duplicated within a namespace
/// (a single output directory) by appending `-2`, `-3`, ... on collision.
#[derive(Debug, Default)]
pub struct SlugRegistry {
    used: std::collections::HashMap<String, usize>,
}

impl SlugRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a unique slug for `name` within this registry's namespace.
    pub fn allocate(&mut self, name: &str) -> String {
        let base = slugify(name);
        let counter = self.used.entry(base.clone()).or_insert(0);
        *counter += 1;
        if *counter == 1 {
            base
        } else {
            format!("{}-{}", base, counter)
        }
    }
}

/// Clamp a single path component to `max` bytes without splitting a UTF-8 char.
pub fn clamp_bytes(name: &str, max: usize) -> String {
    if name.len() <= max {
        return name.to_string();
    }
    let mut end = max;
    while end > 0 && !name.is_char_boundary(end) {
        end -= 1;
    }
    name[..end].to_string()
}

/// Clamp a single path component to the common filesystem NAME_MAX (255 bytes).
pub fn clamp_component(name: &str) -> String {
    clamp_bytes(name, 255)
}

/// Append the Markdown page suffix to a source basename, preserving its real
/// name for human review (`ApiKeyService.java` -> `ApiKeyService.java.md`).
///
/// A source file literally named `index` would otherwise produce `index.md` and
/// shadow a directory hub page, so it becomes `index-file.md`. The component is
/// clamped below the filesystem NAME_MAX, leaving room for the `.md` suffix.
pub fn verbatim_leaf(basename: &str) -> String {
    let trimmed = basename.trim();
    let base = if trimmed.is_empty() { "file" } else { trimmed };
    // Reserve room for ".md" (3 bytes) under NAME_MAX.
    let base = clamp_bytes(base, 252);
    let candidate = format!("{}.md", base);
    if candidate.eq_ignore_ascii_case("index.md") {
        format!("{}-file.md", base)
    } else {
        candidate
    }
}

/// Exact-name uniqueness guard for one output directory.
///
/// Unlike [`SlugRegistry`] this preserves the desired name verbatim (source file
/// and directory names must stay recognizable) and only appends `-2`, `-3`, ...
/// when a name is already taken. Comparison is case-insensitive so it also
/// guards case-insensitive filesystems (macOS APFS).
#[derive(Debug, Default)]
pub struct NameRegistry {
    used: std::collections::HashMap<String, usize>,
}

impl NameRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `name` as taken without returning a value (used to reserve
    /// `index.md` for the directory hub page).
    pub fn reserve(&mut self, name: &str) {
        self.used.entry(name.to_lowercase()).or_insert(1);
    }

    /// Allocate a collision-free name in this directory's namespace.
    pub fn allocate(&mut self, desired: &str) -> String {
        let key = desired.to_lowercase();
        let counter = {
            let c = self.used.entry(key).or_insert(0);
            *c += 1;
            *c
        };
        let name = if counter == 1 {
            desired.to_string()
        } else {
            match desired.strip_suffix(".md") {
                Some(stem) => format!("{}-{}.md", stem, counter),
                None => format!("{}-{}", desired, counter),
            }
        };
        // Record the generated name too, so a later desired name that equals it
        // (e.g. a literal `foo-2` after a collision produced `foo-2`) cannot
        // alias to the same output path.
        self.used.entry(name.to_lowercase()).or_insert(1);
        name
    }
}

/// Escape text for safe inclusion inside a double-quoted Mermaid node label.
///
/// Mermaid labels are quoted with `"`; embedded quotes would terminate the label
/// early, so they are replaced. Newlines and backslashes are also neutralized.
pub fn escape_mermaid_label(raw: &str) -> String {
    raw.replace('\\', "/")
        .replace('"', "'")
        .replace(['\n', '\r'], " ")
        .trim()
        .to_string()
}

/// Derive a valid Mermaid node id from a name.
///
/// Mermaid node ids must be alphanumeric/underscore and must not start with a
/// digit; human-facing text goes in the quoted label instead. This is distinct
/// from `slugify` (which is for filenames) and guarantees a syntactically valid
/// identifier even for names full of spaces, `&`, `/`, or `,`.
pub fn mermaid_node_id(raw: &str, registry: &mut SlugRegistry) -> String {
    let mut id = String::new();
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            id.push(ch.to_ascii_lowercase());
        } else if !id.ends_with('_') {
            id.push('_');
        }
    }
    let mut id = id.trim_matches('_').to_string();
    if id.is_empty() {
        id = "n".to_string();
    }
    if id.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        id = format!("n_{}", id);
    }
    // Reuse the registry so repeated names map to a stable id within a diagram.
    let unique = registry.allocate(&id);
    unique.replace('-', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_href_siblings() {
        let from = Path::new("/root/.agent-content/tree");
        let to = Path::new("/root/src/main.rs");
        let href = relative_href(from, to).unwrap();
        assert_eq!(href, "../../src/main.rs");
    }

    #[test]
    fn source_href_project_relative() {
        let project = Path::new("/proj");
        let from = Path::new("/proj/litho.docs/.agent-content");
        let href = source_href(project, from, "core/cache/Cache.rs").unwrap();
        // from .agent-content up to /proj is two hops, then into core/...
        assert_eq!(href, "../../core/cache/Cache.rs");
    }

    #[test]
    fn slugify_collapses_and_fallback() {
        assert_eq!(slugify("AI Framework & Assistants"), "ai-framework-assistants");
        assert_eq!(slugify("!!!"), "item");
    }

    #[test]
    fn slug_registry_dedupes() {
        let mut reg = SlugRegistry::new();
        assert_eq!(reg.allocate("Cache"), "cache");
        assert_eq!(reg.allocate("Cache"), "cache-2");
        assert_eq!(reg.allocate("cache"), "cache-3");
    }

    #[test]
    fn mermaid_ids_are_valid_and_stable() {
        let mut reg = SlugRegistry::new();
        let a = mermaid_node_id("Identity, Auth & Access", &mut reg);
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
        assert!(!a.chars().next().unwrap().is_ascii_digit());
        // Distinct names yield distinct ids.
        let b = mermaid_node_id("Data Pipeline/ETL", &mut reg);
        assert_ne!(a, b);
    }

    #[test]
    fn escape_label_removes_quotes() {
        assert_eq!(escape_mermaid_label(r#"Say "hi""#), "Say 'hi'");
    }

    #[test]
    fn verbatim_leaf_preserves_name_and_avoids_index() {
        assert_eq!(verbatim_leaf("ApiKeyService.java"), "ApiKeyService.java.md");
        assert_eq!(verbatim_leaf("README.md"), "README.md.md");
        assert_eq!(verbatim_leaf("index"), "index-file.md");
        assert_eq!(verbatim_leaf(""), "file.md");
    }

    #[test]
    fn name_registry_is_case_insensitive() {
        let mut reg = NameRegistry::new();
        reg.reserve("index.md");
        assert_eq!(reg.allocate("ApiKey.java.md"), "ApiKey.java.md");
        // Same name different case collides on case-insensitive filesystems.
        assert_eq!(reg.allocate("apikey.java.md"), "apikey.java-2.md");
        // Reserved index.md cascades to -2.
        assert_eq!(reg.allocate("index.md"), "index-2.md");
    }

    #[test]
    fn name_registry_does_not_alias_generated_names() {
        let mut reg = NameRegistry::new();
        assert_eq!(reg.allocate("Foo"), "Foo");
        // Case-insensitive collision on "Foo".
        assert_eq!(reg.allocate("foo"), "foo-2");
        // A literal "foo-2" must not reuse the name just generated.
        assert_eq!(reg.allocate("foo-2"), "foo-2-2");
    }

    #[test]
    fn clamp_component_respects_name_max() {
        let long = "a".repeat(300);
        assert_eq!(clamp_component(&long).len(), 255);
        // Verbatim leaf keeps room for the .md suffix.
        let leaf = verbatim_leaf(&long);
        assert!(leaf.len() <= 255);
        assert!(leaf.ends_with(".md"));
    }
}
