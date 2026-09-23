use std::path::{Path, PathBuf};

use crate::types::{CodeAndDirectoryInsights, DirectoryDossier};

/// Return a copy of the global directory insights containing only directories
/// that live under one of `root_paths`.
pub fn filter_insights_by_paths(
    insights: &CodeAndDirectoryInsights,
    root_paths: &[PathBuf],
    project_path: &Path,
) -> CodeAndDirectoryInsights {
    let normalized_roots: Vec<String> = root_paths.iter().map(|p| normalize(p)).collect();

    let dirs: Vec<DirectoryDossier> = insights
        .directory_insights
        .iter()
        .filter(|d| is_dossier_in_scope(&d.path, project_path, &normalized_roots))
        .cloned()
        .collect();

    CodeAndDirectoryInsights {
        file_insights: Vec::new(), // legacy field, keep empty
        directory_insights: dirs,
    }
}

fn is_dossier_in_scope(
    dossier_path: &Path,
    project_path: &Path,
    normalized_roots: &[String],
) -> bool {
    let absolute = normalize(dossier_path);
    if is_under_any(&absolute, normalized_roots) {
        return true;
    }

    // If the dossier path is absolute while the area roots are project-relative,
    // also test the path relative to the project root.
    if let Ok(relative) = dossier_path.strip_prefix(project_path) {
        let relative = normalize(relative);
        if relative != absolute && is_under_any(&relative, normalized_roots) {
            return true;
        }
    }

    false
}

fn normalize(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

fn is_under_any(path: &str, roots: &[String]) -> bool {
    roots
        .iter()
        .any(|root| path == root || path.starts_with(&format!("{}/", root)))
}
