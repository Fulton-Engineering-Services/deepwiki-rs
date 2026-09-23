//! `deepwiki-rs explain <path>` — show which exclusion rules apply to a path.
//!
//! This is the debugging counterpart to the exclusion config: every rule
//! that fires is printed with the config entry responsible, then a verdict.
//! Both consumers (structure walker and LLM file-explorer) share the same
//! filter (`utils::exclusion`), so what is printed here is what the scan
//! actually applies — with the two consumer-specific extras (git gate for
//! the walker, size cap for the explorer) shown separately.

use crate::cli::Args;
use crate::config::Config;
use crate::utils::exclusion::{Exclusion, ExclusionFilter, ExclusionKind};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Result of the git-tracked lookup for the target path.
enum GitNote {
    /// Not inside a git work tree (or git is unavailable) — the
    /// `git_tracked_only` rule is inactive for this run.
    Inactive(String),
    File {
        tracked: bool,
    },
    Dir {
        tracked_files: usize,
    },
}

pub fn run(args: &Args, target: &Path, config_arg: Option<PathBuf>) -> Result<()> {
    let (config, config_source) = load_config(config_arg.as_ref())?;
    let root = args.project_path.clone();

    // Resolve the input to a project-relative path. Relative inputs are
    // interpreted against the project root (which is "." unless -p says
    // otherwise), absolute inputs must live inside the root.
    let rel: PathBuf = if target.is_absolute() {
        let canon_root = root.canonicalize().unwrap_or_else(|_| root.clone());
        let canon_target = target
            .canonicalize()
            .unwrap_or_else(|_| target.to_path_buf());
        match canon_target.strip_prefix(&canon_root) {
            Ok(stripped) => stripped.to_path_buf(),
            Err(_) => anyhow::bail!(
                "path {} is outside the project root {}",
                target.display(),
                root.display()
            ),
        }
    } else {
        target.to_path_buf()
    };

    let abs = root.join(&rel);
    let exists = abs.exists();
    let is_dir = abs.is_dir();

    let filter = ExclusionFilter::new(&config);
    let explanation = evaluate(&filter, &root, &rel, is_dir, config.max_file_size);
    let git = git_note(&root, &rel, is_dir);

    print_report(
        &rel,
        &root,
        &config_source,
        is_dir,
        exists,
        &explanation,
        &git,
    );
    Ok(())
}

/// Everything `evaluate` decides from config + path shape alone (git is
/// looked up separately because it needs the filesystem).
struct Evaluation {
    /// Per directory component from root to leaf: (component path, rules).
    dir_rows: Vec<(PathBuf, Vec<Exclusion>)>,
    /// File-level rules (files only).
    file_rows: Vec<Exclusion>,
    /// Size-cap rule (files only, when the file exists).
    size_row: Option<Exclusion>,
}

impl Evaluation {
    /// All matching rules in evaluation order (root-first).
    fn all_exclusions(&self, git_untracked: bool) -> Vec<Exclusion> {
        let mut out: Vec<Exclusion> = self
            .dir_rows
            .iter()
            .flat_map(|(_, rules)| rules.iter().cloned())
            .collect();
        out.extend(self.file_rows.iter().cloned());
        if let Some(size) = &self.size_row {
            out.push(size.clone());
        }
        if git_untracked {
            out.push(Exclusion::new(
                ExclusionKind::Untracked,
                "not tracked by git (git_tracked_only = true; structure walker only)".to_string(),
            ));
        }
        out
    }
}

fn evaluate(
    filter: &ExclusionFilter,
    root: &Path,
    rel: &Path,
    is_dir: bool,
    max_file_size: u64,
) -> Evaluation {
    let components: Vec<PathBuf> = rel
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(PathBuf::from(s)),
            _ => None,
        })
        .collect();

    // Directory rows: for a directory target, every component including the
    // leaf; for a file target, only the ancestors.
    let dir_count = if is_dir {
        components.len()
    } else {
        components.len().saturating_sub(1)
    };
    let mut dir_rows = Vec::new();
    for n in 1..=dir_count {
        let prefix: PathBuf = components[..n].iter().collect();
        dir_rows.push((prefix.clone(), filter.dir_component_exclusions(&prefix)));
    }

    let file_rows = if is_dir {
        Vec::new()
    } else {
        filter.file_rules(rel)
    };

    let size_row = if is_dir {
        None
    } else if let Ok(metadata) = std::fs::metadata(root.join(rel)) {
        if metadata.len() > max_file_size {
            Some(Exclusion::new(
                ExclusionKind::Oversize,
                format!(
                    "{} bytes exceeds max_file_size {}",
                    metadata.len(),
                    max_file_size
                ),
            ))
        } else {
            None
        }
    } else {
        None
    };

    Evaluation {
        dir_rows,
        file_rows,
        size_row,
    }
}

fn load_config(explicit: Option<&PathBuf>) -> Result<(Config, String)> {
    if let Some(path) = explicit {
        let config = Config::from_file(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        return Ok((config, path.display().to_string()));
    }
    let default = PathBuf::from("litho.toml");
    if default.exists() {
        let config = Config::from_file(&default)
            .with_context(|| format!("failed to read {}", default.display()))?;
        Ok((config, default.display().to_string()))
    } else {
        Ok((
            Config::default(),
            "<built-in defaults (no litho.toml found)>".to_string(),
        ))
    }
}

fn git_note(root: &Path, rel: &Path, is_dir: bool) -> GitNote {
    let in_repo = Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(root)
        .output();
    match in_repo {
        Err(e) => return GitNote::Inactive(format!("git unavailable: {}", e)),
        Ok(out) if !out.status.success() => {
            return GitNote::Inactive("not a git repository".to_string());
        }
        _ => {}
    }

    if is_dir {
        let out = Command::new("git")
            .args(["ls-files", "--"])
            .arg(rel)
            .current_dir(root)
            .output();
        match out {
            Ok(out) if out.status.success() => {
                let tracked_files = String::from_utf8_lossy(&out.stdout).lines().count();
                GitNote::Dir { tracked_files }
            }
            _ => GitNote::Inactive("git ls-files failed".to_string()),
        }
    } else {
        let out = Command::new("git")
            .args(["ls-files", "--error-unmatch", "--"])
            .arg(rel)
            .current_dir(root)
            .output();
        match out {
            Ok(out) => GitNote::File {
                tracked: out.status.success(),
            },
            Err(e) => GitNote::Inactive(format!("git unavailable: {}", e)),
        }
    }
}

fn print_report(
    rel: &Path,
    root: &Path,
    config_source: &str,
    is_dir: bool,
    exists: bool,
    eval: &Evaluation,
    git: &GitNote,
) {
    let rel_display = rel.display();
    let target_kind = if is_dir { "directory" } else { "file" };

    println!("path           : {}", rel_display);
    println!("project root   : {}", root.display());
    println!("config         : {}", config_source);
    println!("target         : {}", target_kind);
    if !exists {
        println!("note           : path does not exist on disk — rules are evaluated statically");
    }
    println!();

    if !eval.dir_rows.is_empty() {
        println!("directories (root -> leaf):");
        for (component, rules) in &eval.dir_rows {
            if rules.is_empty() {
                println!("  ok   {}", component.display());
            } else {
                for rule in rules {
                    println!("  EXCL {}  —  {}", component.display(), rule);
                }
            }
        }
        println!();
    }

    if is_dir {
        println!("file rules     : n/a (target is a directory)");
        println!();
    } else {
        println!("file rules ({}):", rel_display);
        if eval.file_rows.is_empty() {
            println!("  ok   no excluded_files / extension / test / hidden / binary rule matched");
        } else {
            for rule in &eval.file_rows {
                println!("  EXCL {}", rule);
            }
        }
        match &eval.size_row {
            Some(rule) => println!("  EXCL {}", rule),
            None => {
                if exists {
                    println!("  ok   max_file_size");
                } else {
                    println!("  --   max_file_size (skipped: path does not exist)");
                }
            }
        }
        println!();
    }

    // git gate: structure-walker only
    println!("other checks:");
    match git {
        GitNote::Inactive(reason) => {
            println!("  ok   git_tracked_only — inactive ({})", reason);
        }
        GitNote::File { tracked: true } => {
            println!("  ok   git_tracked_only — tracked by git (structure walker only)");
        }
        GitNote::File { tracked: false } => {
            println!("  EXCL git_tracked_only: not tracked by git (structure walker only)");
        }
        GitNote::Dir { tracked_files } if *tracked_files > 0 => {
            println!(
                "  ok   git_tracked_only — {} tracked files under this directory",
                tracked_files
            );
        }
        GitNote::Dir { .. } => {
            println!(
                "  !!   git_tracked_only — no tracked files under this directory \
                 (the structure walker skips its files)"
            );
        }
    }
    println!("  ok   max_file_size on read — enforced by the file-explorer tool");
    println!();

    let git_untracked = matches!(git, GitNote::File { tracked: false });
    let matches: Vec<Exclusion> = eval.all_exclusions(git_untracked);
    if matches.is_empty() {
        println!("verdict: INCLUDED — no exclusion rule matched");
    } else {
        println!(
            "verdict: EXCLUDED — {} rule{} matched",
            matches.len(),
            if matches.len() == 1 { "" } else { "s" }
        );
        println!("  first: {}", matches[0]);
        if matches.len() > 1 {
            for rule in &matches[1..] {
                println!("  also: {}", rule);
            }
        }
        println!();
        println!(
            "note: the structure walker and the LLM file-explorer both apply these \
             rules; the git gate runs in the walker only, the size cap in both."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::exclusion::ExclusionKind;

    fn bare_config() -> Config {
        let mut config = Config::default();
        config.excluded_dirs.clear();
        config.excluded_files.clear();
        config.excluded_extensions.clear();
        config.included_extensions.clear();
        config
    }

    #[test]
    fn evaluate_reports_first_matching_dir_rule_root_first() {
        let mut config = bare_config();
        config.excluded_dirs = vec!["generated".into()];
        let filter = ExclusionFilter::new(&config);

        let ev = evaluate(
            &filter,
            Path::new("."),
            Path::new("ui/api-types/src/generated/api.ts"),
            false,
            config.max_file_size,
        );
        assert_eq!(ev.dir_rows.len(), 4);
        // The first three components are clean; the leaf fires.
        assert!(ev.dir_rows[..3].iter().all(|(_, r)| r.is_empty()));
        assert_eq!(ev.dir_rows[3].1[0].kind, ExclusionKind::ExcludedDir);

        let all = ev.all_exclusions(false);
        assert_eq!(all[0].kind, ExclusionKind::ExcludedDir);
    }

    #[test]
    fn evaluate_clean_file_has_no_matches() {
        let config = bare_config();
        let filter = ExclusionFilter::new(&config);
        let ev = evaluate(
            &filter,
            Path::new("."),
            Path::new("src/main.ts"),
            false,
            config.max_file_size,
        );
        assert!(ev.all_exclusions(false).is_empty());
    }

    #[test]
    fn evaluate_directory_target_has_no_file_rows() {
        let mut config = bare_config();
        config.excluded_dirs = vec![".design".into()];
        config.include_hidden = false;
        let filter = ExclusionFilter::new(&config);

        let ev = evaluate(
            &filter,
            Path::new("."),
            Path::new(".design/finalized"),
            true,
            config.max_file_size,
        );
        assert!(ev.file_rows.is_empty());
        let all = ev.all_exclusions(false);
        assert!(all.iter().any(|e| e.kind == ExclusionKind::HiddenDir));
    }

    #[test]
    fn evaluate_file_rule_fires_without_dir_rules() {
        let mut config = bare_config();
        config.excluded_files = vec!["batch_*.log".into()];
        let filter = ExclusionFilter::new(&config);

        let ev = evaluate(
            &filter,
            Path::new("."),
            Path::new("logs/batch_run.log"),
            false,
            config.max_file_size,
        );
        let all = ev.all_exclusions(false);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].kind, ExclusionKind::ExcludedFile);
    }

    #[test]
    fn git_untracked_flips_verdict_input() {
        let config = bare_config();
        let filter = ExclusionFilter::new(&config);
        let ev = evaluate(
            &filter,
            Path::new("."),
            Path::new("src/main.ts"),
            false,
            config.max_file_size,
        );
        assert!(ev.all_exclusions(false).is_empty());
        let with_git = ev.all_exclusions(true);
        assert_eq!(with_git.len(), 1);
        assert_eq!(with_git[0].kind, ExclusionKind::Untracked);
    }
}
