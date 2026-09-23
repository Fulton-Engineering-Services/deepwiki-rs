//! Unified exclusion rules shared by the structure walker, the LLM
//! file-explorer tool, and the `explain` subcommand.
//!
//! Historically the two consumers interpreted the same config keys
//! differently:
//!
//! - the structure walker (`StructureExtractor`) matched `excluded_dirs`
//!   against directory basenames with exact, case-insensitive equality;
//! - the file explorer (`AgentToolFileExplorer`) substring-matched
//!   `excluded_dirs` against the **full, project-joined path**.
//!
//! That divergence caused real bugs: a `.litho` entry made the explorer
//! ignore every path when the scan root was the shadow tree under
//! `.litho/tree/repo`, and entries like `build` / `.git` blindsided
//! unrelated paths such as `build.gradle.kts` and `.github/`.
//!
//! Everything here matches against paths **relative to the project root**
//! with one shared rule set, for all consumers:
//!
//! - `excluded_dirs` entries without `/` match any single path component
//!   (basename) exactly, or as a glob if they contain `* ? [ ]`;
//! - entries containing `/` are root-anchored path patterns
//!   (`ui/api-types/src/generated`, `docs/**/*.md`, `**/generated`);
//! - `excluded_files` follows the same shape: basename rules for entries
//!   without `/`, path rules for entries with `/`, with real glob semantics
//!   (the old code stripped `*` and substring-matched, so `batch_*.log`
//!   never matched anything);
//! - hidden-directory, test-directory, hidden-file, test-file, binary-file
//!   and extension rules are applied identically by both consumers.

use crate::config::Config;
use crate::utils::file_utils::{is_binary_file_path, is_test_directory, is_test_file};
use glob::{MatchOptions, Pattern};
use std::path::{Component, Path, PathBuf};

/// Inputs are pre-lowercased at match time, so patterns compile with
/// case-sensitive matching; separators are literal so `*` never crosses `/`
/// in path patterns (use `**` for that).
const MATCH_OPTIONS: MatchOptions = MatchOptions {
    case_sensitive: true,
    require_literal_separator: true,
    require_literal_leading_dot: false,
};

/// Identifies which configuration rule excluded a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExclusionKind {
    ExcludedDir,
    TestDir,
    HiddenDir,
    ExcludedFile,
    ExcludedExtension,
    NotIncludedExtension,
    TestFile,
    HiddenFile,
    BinaryFile,
    Untracked,
    Oversize,
}

impl ExclusionKind {
    pub fn label(self) -> &'static str {
        use ExclusionKind::*;
        match self {
            ExcludedDir => "excluded_dirs",
            TestDir => "test-directory rule",
            HiddenDir => "hidden-directory rule",
            ExcludedFile => "excluded_files",
            ExcludedExtension => "excluded_extensions",
            NotIncludedExtension => "included_extensions",
            TestFile => "test-file rule",
            HiddenFile => "hidden-file rule",
            BinaryFile => "binary-file rule",
            Untracked => "git_tracked_only",
            Oversize => "max_file_size",
        }
    }
}

/// A single reason why a path is excluded from analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct Exclusion {
    pub kind: ExclusionKind,
    pub detail: String,
}

impl Exclusion {
    pub fn new(kind: ExclusionKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for Exclusion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind.label(), self.detail)
    }
}

/// One compiled entry from `excluded_dirs` / `excluded_files`.
#[derive(Clone, Debug)]
struct CompiledRule {
    /// The config entry as written (for messages).
    display: String,
    matcher: Matcher,
}

#[derive(Clone, Debug)]
enum Matcher {
    /// Literal, compared for equality (already lowercased).
    Exact(String),
    /// Glob compiled from the lowercased entry. `alternates` holds variants
    /// that make `a/**/b` also match `a/b` and `**/x` also match `x`, so
    /// `**` matches zero-length subdirectory runs like gitignore.
    Glob {
        pattern: Pattern,
        alternates: Vec<Pattern>,
    },
}

fn has_glob_chars(s: &str) -> bool {
    s.contains(['*', '?', '[', ']'])
}

/// Build variants of a path pattern that match zero-length `**` segments.
fn glob_alternates(lower: &str) -> Vec<Pattern> {
    let mut candidates: Vec<String> = Vec::new();
    if lower.contains("/**/") {
        candidates.push(lower.replace("/**/", "/"));
    }
    if let Some(rest) = lower.strip_prefix("**/") {
        candidates.push(rest.to_string());
    }
    candidates
        .iter()
        .filter_map(|c| Pattern::new(c).ok())
        .collect()
}

impl CompiledRule {
    fn parse(entry: &str) -> Option<Self> {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            return None;
        }
        let lower = trimmed.to_lowercase();
        let matcher = if has_glob_chars(&lower) {
            match Pattern::new(&lower) {
                Ok(pattern) => Matcher::Glob {
                    pattern,
                    alternates: glob_alternates(&lower),
                },
                // Invalid glob: degrade to a literal so a typo in the config
                // can never panic or silently over-match.
                Err(_) => Matcher::Exact(lower),
            }
        } else {
            Matcher::Exact(lower)
        };
        Some(Self {
            display: trimmed.to_string(),
            matcher,
        })
    }

    fn matches_component(&self, component_lower: &str) -> bool {
        match &self.matcher {
            Matcher::Exact(lit) => component_lower == lit,
            Matcher::Glob {
                pattern,
                alternates,
            } => {
                pattern.matches_with(component_lower, MATCH_OPTIONS)
                    || alternates
                        .iter()
                        .any(|p| p.matches_with(component_lower, MATCH_OPTIONS))
            }
        }
    }

    fn matches_path(&self, rel_lower: &str) -> bool {
        match &self.matcher {
            Matcher::Exact(lit) => rel_lower == lit,
            Matcher::Glob {
                pattern,
                alternates,
            } => {
                pattern.matches_path_with(Path::new(rel_lower), MATCH_OPTIONS)
                    || alternates
                        .iter()
                        .any(|p| p.matches_path_with(Path::new(rel_lower), MATCH_OPTIONS))
            }
        }
    }
}

/// Compiled exclusion rules, built once from a [`Config`] and shared by all
/// path consumers.
#[derive(Clone, Debug)]
pub struct ExclusionFilter {
    /// `excluded_dirs` entries without `/`: match any single component.
    dir_component_rules: Vec<CompiledRule>,
    /// `excluded_dirs` entries with `/`: root-anchored path patterns.
    dir_path_rules: Vec<CompiledRule>,
    /// `excluded_files` entries without `/`: match the file basename.
    file_component_rules: Vec<CompiledRule>,
    /// `excluded_files` entries with `/`: root-anchored path patterns.
    file_path_rules: Vec<CompiledRule>,
    excluded_extensions: Vec<String>,
    included_extensions: Vec<String>,
    include_tests: bool,
    include_hidden: bool,
}

impl ExclusionFilter {
    pub fn new(config: &Config) -> Self {
        let (mut dir_comp, mut dir_path) = (Vec::new(), Vec::new());
        for entry in &config.excluded_dirs {
            let Some(rule) = CompiledRule::parse(entry) else {
                continue;
            };
            if entry.contains('/') {
                dir_path.push(rule);
            } else {
                dir_comp.push(rule);
            }
        }
        let (mut file_comp, mut file_path) = (Vec::new(), Vec::new());
        for entry in &config.excluded_files {
            let Some(rule) = CompiledRule::parse(entry) else {
                continue;
            };
            if entry.contains('/') {
                file_path.push(rule);
            } else {
                file_comp.push(rule);
            }
        }
        Self {
            dir_component_rules: dir_comp,
            dir_path_rules: dir_path,
            file_component_rules: file_comp,
            file_path_rules: file_path,
            excluded_extensions: config
                .excluded_extensions
                .iter()
                .map(|e| e.to_lowercase())
                .collect(),
            included_extensions: config
                .included_extensions
                .iter()
                .map(|e| e.to_lowercase())
                .collect(),
            include_tests: config.include_tests,
            include_hidden: config.include_hidden,
        }
    }

    /// All rules that exclude this directory itself (not its ancestors).
    /// The structure walker uses this because it only descends into
    /// directories it has already accepted; the explorer and `explain`
    /// walk full chains via [`ExclusionFilter::dir_exclusions`].
    pub fn dir_component_exclusions(&self, rel_dir: &Path) -> Vec<Exclusion> {
        let components = split_components(rel_dir);
        let Some(last_orig) = components.last() else {
            return Vec::new();
        };
        let last_lower = last_orig.to_lowercase();
        let rel_lower = normalize_rel(rel_dir);
        let rel_display = components.join("/");

        let mut out = Vec::new();

        for rule in &self.dir_component_rules {
            if rule.matches_component(&last_lower) {
                out.push(Exclusion::new(
                    ExclusionKind::ExcludedDir,
                    format!(
                        "entry \"{}\" matched directory \"{}\"",
                        rule.display, rel_display
                    ),
                ));
            }
        }
        for rule in &self.dir_path_rules {
            if rule.matches_path(&rel_lower) {
                out.push(Exclusion::new(
                    ExclusionKind::ExcludedDir,
                    format!(
                        "entry \"{}\" matched path \"{}\"",
                        rule.display, rel_display
                    ),
                ));
            }
        }
        if !self.include_tests && is_test_directory(last_orig) {
            out.push(Exclusion::new(
                ExclusionKind::TestDir,
                format!(
                    "directory \"{}\" matches test-directory patterns (include_tests = false)",
                    last_orig
                ),
            ));
        }
        if !self.include_hidden && last_orig.starts_with('.') {
            out.push(Exclusion::new(
                ExclusionKind::HiddenDir,
                format!(
                    "directory \"{}\" is hidden (include_hidden = false)",
                    last_orig
                ),
            ));
        }
        out
    }

    /// All rules that exclude this directory or any of its ancestors,
    /// ordered root-first.
    pub fn dir_exclusions(&self, rel_dir: &Path) -> Vec<Exclusion> {
        let mut out = Vec::new();
        for prefix in ancestor_prefixes(rel_dir) {
            out.extend(self.dir_component_exclusions(&prefix));
        }
        out
    }

    /// File-level rules only (patterns, extensions, test/hidden/binary) for
    /// the file at `rel_file` — does not inspect ancestor directories.
    pub fn file_rules(&self, rel_file: &Path) -> Vec<Exclusion> {
        let components = split_components(rel_file);
        let Some(last_orig) = components.last() else {
            return Vec::new();
        };
        let last_lower = last_orig.to_lowercase();
        let rel_lower = normalize_rel(rel_file);
        let rel_display = components.join("/");

        let mut out = Vec::new();

        for rule in &self.file_component_rules {
            if rule.matches_component(&last_lower) {
                out.push(Exclusion::new(
                    ExclusionKind::ExcludedFile,
                    format!("entry \"{}\" matched \"{}\"", rule.display, last_orig),
                ));
            }
        }
        for rule in &self.file_path_rules {
            if rule.matches_path(&rel_lower) {
                out.push(Exclusion::new(
                    ExclusionKind::ExcludedFile,
                    format!(
                        "entry \"{}\" matched path \"{}\"",
                        rule.display, rel_display
                    ),
                ));
            }
        }

        let extension_lower = Path::new(last_orig)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        if let Some(ref ext) = extension_lower
            && self.excluded_extensions.iter().any(|e| e == ext)
        {
            out.push(Exclusion::new(
                ExclusionKind::ExcludedExtension,
                format!("extension \"{}\" is excluded", ext),
            ));
        }

        if !self.included_extensions.is_empty() {
            match &extension_lower {
                Some(ext) if self.included_extensions.iter().any(|e| e == ext) => {}
                Some(ext) => out.push(Exclusion::new(
                    ExclusionKind::NotIncludedExtension,
                    format!(
                        "extension \"{}\" is not in included_extensions ({})",
                        ext,
                        self.included_extensions.join(", ")
                    ),
                )),
                None => out.push(Exclusion::new(
                    ExclusionKind::NotIncludedExtension,
                    format!(
                        "file has no extension but included_extensions is set ({})",
                        self.included_extensions.join(", ")
                    ),
                )),
            }
        }

        if !self.include_tests && is_test_file(Path::new(rel_display.as_str())) {
            out.push(Exclusion::new(
                ExclusionKind::TestFile,
                format!(
                    "\"{}\" matches test-file patterns (include_tests = false)",
                    rel_display
                ),
            ));
        }
        if !self.include_hidden && last_orig.starts_with('.') {
            out.push(Exclusion::new(
                ExclusionKind::HiddenFile,
                format!("file \"{}\" is hidden (include_hidden = false)", last_orig),
            ));
        }
        if is_binary_file_path(Path::new(last_orig)) {
            out.push(Exclusion::new(
                ExclusionKind::BinaryFile,
                format!("\"{}\" has a binary-file extension", last_orig),
            ));
        }

        out
    }

    /// All rules that exclude this file: ancestor-directory rules
    /// (root-first) followed by file-level rules. This is the single source
    /// of truth for "is this path in scope?" used by the structure walker,
    /// the file explorer, and `explain`.
    pub fn file_exclusions(&self, rel_file: &Path) -> Vec<Exclusion> {
        let mut out = Vec::new();
        let components = split_components(rel_file);
        // Ancestors = all components except the file name itself.
        for n in 1..components.len() {
            let prefix = PathBuf::from(components[..n].join("/"));
            out.extend(self.dir_component_exclusions(&prefix));
        }
        out.extend(self.file_rules(rel_file));
        out
    }
}

/// Split a relative path into components, skipping root/prefix components.
fn split_components(rel: &Path) -> Vec<&str> {
    rel.components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .collect()
}

/// Lowercased, forward-slash form of a relative path.
fn normalize_rel(rel: &Path) -> String {
    rel.to_string_lossy().replace('\\', "/").to_lowercase()
}

/// All ancestor prefixes of `rel` including `rel` itself, root-first.
/// For `a/b/c` this yields `a`, `a/b`, `a/b/c`.
fn ancestor_prefixes(rel: &Path) -> Vec<PathBuf> {
    let components = split_components(rel);
    (1..=components.len())
        .map(|n| PathBuf::from(components[..n].join("/")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bare config with all exclusion lists emptied, so each test only
    /// exercises the entries it declares.
    fn base_config() -> Config {
        let mut config = Config::default();
        config.excluded_dirs.clear();
        config.excluded_files.clear();
        config.excluded_extensions.clear();
        config.included_extensions.clear();
        config
    }

    fn has_kinds(exclusions: &[Exclusion], kinds: &[ExclusionKind]) -> bool {
        kinds
            .iter()
            .all(|k| exclusions.iter().any(|e| e.kind == *k))
    }

    #[test]
    fn basename_dir_rule_matches_component_at_any_depth() {
        let mut config = base_config();
        config.excluded_dirs = vec!["generated".into()];
        let filter = ExclusionFilter::new(&config);

        let hits = filter.file_exclusions(Path::new("ui/api-types/src/generated/api.ts"));
        assert!(has_kinds(&hits, &[ExclusionKind::ExcludedDir]));
        // Path-scoped form must NOT fire (entry has no '/').
        assert_eq!(
            hits.iter()
                .filter(|e| e.kind == ExclusionKind::ExcludedDir)
                .count(),
            1
        );
    }

    #[test]
    fn basename_dir_rule_is_exact_not_substring() {
        // Regression: the old explorer substring-matched the full path, so
        // entry "build" ignored every path containing those 5 letters.
        let mut config = base_config();
        config.excluded_dirs = vec!["build".into()];
        let filter = ExclusionFilter::new(&config);

        assert!(
            filter
                .file_exclusions(Path::new("build.gradle.kts"))
                .is_empty(),
            "build.gradle.kts must not match entry \"build\""
        );
        assert!(
            filter
                .file_exclusions(Path::new("build-logic/src/plugin.kts"))
                .is_empty(),
            "build-logic must not match entry \"build\""
        );
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("scripts/build/lib.sh")),
            &[ExclusionKind::ExcludedDir]
        ));
    }

    #[test]
    fn git_entry_does_not_match_github_but_hidden_rule_does() {
        let mut config = base_config();
        config.excluded_dirs = vec![".git".into()];
        config.include_hidden = true;
        let filter = ExclusionFilter::new(&config);
        assert!(
            filter
                .file_exclusions(Path::new(".github/workflows/ci.yml"))
                .is_empty(),
            ".github must not match entry \".git\" when hidden dirs are included"
        );

        config.include_hidden = false;
        let filter = ExclusionFilter::new(&config);
        assert!(has_kinds(
            &filter.file_exclusions(Path::new(".github/workflows/ci.yml")),
            &[ExclusionKind::HiddenDir]
        ));
    }

    #[test]
    fn shadow_root_prefix_never_leaks_into_matching() {
        // Regression: the explorer used to substring-match FULL paths, so
        // with scan root .litho/tree/repo an entry ".litho" ignored
        // everything. Rules now only ever see project-relative paths.
        let mut config = base_config();
        config.excluded_dirs = vec![".litho".into()];
        let filter = ExclusionFilter::new(&config);
        assert!(
            filter.file_exclusions(Path::new("ui/foo.ts")).is_empty(),
            "relative path without .litho component must not match entry \".litho\""
        );
        assert!(has_kinds(
            &filter.file_exclusions(Path::new(".litho/cache/x.json")),
            &[ExclusionKind::HiddenDir]
        ));
    }

    #[test]
    fn path_scoped_dir_entry_only_matches_from_root() {
        let mut config = base_config();
        config.excluded_dirs = vec!["ui/api-types/src/generated".into()];
        let filter = ExclusionFilter::new(&config);

        assert!(has_kinds(
            &filter.file_exclusions(Path::new("ui/api-types/src/generated/api.ts")),
            &[ExclusionKind::ExcludedDir]
        ));
        assert!(
            filter
                .file_exclusions(Path::new("core/generated/api.ts"))
                .is_empty(),
            "a root-anchored path entry must not match the same basename elsewhere"
        );
    }

    #[test]
    fn dir_rule_glob_matches_any_depth() {
        let mut config = base_config();
        config.excluded_dirs = vec!["**/generated".into()];
        let filter = ExclusionFilter::new(&config);
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("core/generated/x.ts")),
            &[ExclusionKind::ExcludedDir]
        ));
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("generated/x.ts")),
            &[ExclusionKind::ExcludedDir],
        ));
    }

    #[test]
    fn file_globs_are_real_globs() {
        // Regression: the old code stripped '*' and substring-matched, so
        // "batch_*.log" could never match "batch_run.log".
        let mut config = base_config();
        config.excluded_files = vec!["batch_*.log".into(), "*.md".into()];
        let filter = ExclusionFilter::new(&config);

        assert!(has_kinds(
            &filter.file_exclusions(Path::new("logs/batch_run.log")),
            &[ExclusionKind::ExcludedFile]
        ));
        assert!(
            filter.file_exclusions(Path::new("logs/run.log")).is_empty(),
            "\"batch_*.log\" must not match \"run.log\""
        );
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("docs/README.md")),
            &[ExclusionKind::ExcludedFile]
        ));
    }

    #[test]
    fn file_path_glob_handles_zero_length_double_star() {
        let mut config = base_config();
        config.excluded_files = vec!["docs/**/*.md".into()];
        let filter = ExclusionFilter::new(&config);

        assert!(has_kinds(
            &filter.file_exclusions(Path::new("docs/guide.md")),
            &[ExclusionKind::ExcludedFile],
            // zero directories between docs/ and the file
        ));
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("docs/a/b/c.md")),
            &[ExclusionKind::ExcludedFile]
        ));
        assert!(
            filter.file_exclusions(Path::new("src/x.md")).is_empty(),
            "path glob must be root-anchored"
        );
    }

    #[test]
    fn directory_and_hidden_rules_apply_to_dir_chains() {
        let mut config = base_config();
        config.excluded_dirs = vec![".design".into()];
        let filter = ExclusionFilter::new(&config);

        // dir_exclusions covers ancestors + self for directory targets.
        let hits = filter.dir_exclusions(Path::new(".design/finalized"));
        assert!(has_kinds(&hits, &[ExclusionKind::HiddenDir]));
    }

    #[test]
    fn test_dir_and_test_file_rules_fire() {
        let config = base_config(); // include_tests = false by default
        let filter = ExclusionFilter::new(&config);

        assert!(has_kinds(
            &filter.file_exclusions(Path::new("core/core-integration-tests/mod.rs")),
            &[ExclusionKind::TestDir]
        ));
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("src/app.component.spec.ts")),
            &[ExclusionKind::TestFile]
        ));
        // Test detection must use the RELATIVE path: an absolute prefix
        // containing /test/ must not contaminate matching (old bug).
        assert!(filter.file_exclusions(Path::new("src/main.ts")).is_empty());
    }

    #[test]
    fn extension_rules_and_included_whitelist() {
        let mut config = base_config();
        config.excluded_extensions = vec!["svg".into()];
        let filter = ExclusionFilter::new(&config);
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("assets/icon.svg")),
            &[ExclusionKind::ExcludedExtension]
        ));

        config.excluded_extensions.clear();
        config.included_extensions = vec!["rs".into()];
        let filter = ExclusionFilter::new(&config);
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("src/lib.ts")),
            &[ExclusionKind::NotIncludedExtension]
        ));
        assert!(filter.file_exclusions(Path::new("src/main.rs")).is_empty());
        // No extension + whitelist set => excluded (structure walker parity).
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("Makefile")),
            &[ExclusionKind::NotIncludedExtension]
        ));
    }

    #[test]
    fn hidden_file_and_binary_rules() {
        let config = base_config(); // include_hidden = false
        let filter = ExclusionFilter::new(&config);
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("app/.env")),
            &[ExclusionKind::HiddenFile]
        ));
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("core/libs/libnative.so")),
            &[ExclusionKind::BinaryFile]
        ));
    }

    #[test]
    fn matching_is_case_insensitive() {
        let mut config = base_config();
        config.excluded_dirs = vec!["GENERATED".into()];
        let filter = ExclusionFilter::new(&config);
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("ui/Generated/api.ts")),
            &[ExclusionKind::ExcludedDir]
        ));
    }

    #[test]
    fn empty_entries_are_ignored() {
        let mut config = base_config();
        config.excluded_dirs = vec!["".into(), "   ".into()];
        config.excluded_files = vec!["".into()];
        let filter = ExclusionFilter::new(&config);
        assert!(filter.file_exclusions(Path::new("src/main.ts")).is_empty());
    }

    #[test]
    fn invalid_glob_degrades_to_literal() {
        let mut config = base_config();
        config.excluded_files = vec!["[unclosed".into()];
        let filter = ExclusionFilter::new(&config);
        assert!(
            filter.file_exclusions(Path::new("src/main.ts")).is_empty(),
            "invalid glob must not match everything"
        );
        assert!(has_kinds(
            &filter.file_exclusions(Path::new("src/[unclosed")),
            &[ExclusionKind::ExcludedFile]
        ));
    }

    #[test]
    fn ancestors_reported_root_first() {
        let mut config = base_config();
        config.excluded_dirs = vec!["ui".into(), "generated".into()];
        config.include_hidden = false;
        let filter = ExclusionFilter::new(&config);

        let hits = filter.file_exclusions(Path::new("ui/.hidden/generated/x.ts"));
        let kinds: Vec<ExclusionKind> = hits.iter().map(|e| e.kind).collect();
        // Root-first: ui (excluded_dirs), then .hidden (hidden rule),
        // then generated (excluded_dirs), then file-level rules.
        assert_eq!(kinds[0], ExclusionKind::ExcludedDir);
        assert_eq!(kinds[1], ExclusionKind::HiddenDir);
        assert_eq!(kinds[2], ExclusionKind::ExcludedDir);
    }
}
