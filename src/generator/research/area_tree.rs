use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::types::DirectoryDossier;

/// A mapped area in the project, produced by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct MappedArea {
    pub name: String,
    pub description: String,
    pub root_paths: Vec<String>,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct AreaMappingReport {
    pub areas: Vec<MappedArea>,
}

/// A node in the hierarchical area tree.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct AreaNode {
    pub id: String,
    pub name: String,
    pub description: String,
    pub root_paths: Vec<PathBuf>,
    pub children: Vec<AreaNode>,
    #[serde(skip)]
    #[schemars(skip)]
    pub dossier_paths: Vec<PathBuf>,
}

/// Hierarchical area tree built from directory dossiers.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct AreaTree {
    pub root: AreaNode,
}

impl AreaTree {
    /// Build an area tree from a mapping report and the directory dossiers.
    ///
    /// Each mapped area becomes a child of the root node. Dossiers are assigned
    /// to the first area whose root path is a prefix of the dossier path;
    /// dossiers that do not fall under any area are kept at the root.
    pub fn from_dossiers(
        project_path: &Path,
        dossiers: &[DirectoryDossier],
        report: &AreaMappingReport,
    ) -> Self {
        let project_path = normalize_path(project_path);

        let root_name = project_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "project".to_string());

        let mut root = AreaNode {
            id: "root".to_string(),
            name: root_name,
            description: "Root of the project".to_string(),
            root_paths: vec![project_path.clone()],
            ..Default::default()
        };

        // Pre-normalize every dossier path once so coverage checks are cheap.
        let normalized_dossiers: Vec<(PathBuf, String)> = dossiers
            .iter()
            .filter_map(|d| {
                let rel = relative_to_project(&project_path, &d.path)?;
                let rel_str = path_to_normalized_string(&rel);
                Some((rel.clone(), rel_str))
            })
            .collect();

        let mut assigned = vec![false; normalized_dossiers.len()];

        for area in &report.areas {
            let mut node = AreaNode {
                id: slugify_id(&area.name),
                name: area.name.clone(),
                description: area.description.clone(),
                ..Default::default()
            };

            let mut roots: Vec<(PathBuf, String)> = Vec::new();
            for raw in &area.root_paths {
                if let Some(rel) = relative_to_project_str(&project_path, raw) {
                    roots.push((rel.clone(), path_to_normalized_string(&rel)));
                }
            }
            node.root_paths = roots.iter().map(|(p, _)| p.clone()).collect();

            for (idx, (dossier_path, dossier_str)) in normalized_dossiers.iter().enumerate() {
                if assigned[idx] {
                    continue;
                }
                for (_, root_str) in &roots {
                    if path_is_under(dossier_str, root_str) {
                        node.dossier_paths.push(dossier_path.clone());
                        assigned[idx] = true;
                        break;
                    }
                }
            }

            root.children.push(node);
        }

        // Any dossiers not covered by an area stay at the root.
        for (idx, (path, _)) in normalized_dossiers.iter().enumerate() {
            if !assigned[idx] {
                root.dossier_paths.push(path.clone());
            }
        }

        AreaTree { root }
    }

    /// Return all leaf nodes (nodes with no children) in tree order.
    pub fn leaves(&self) -> Vec<&AreaNode> {
        let mut out = Vec::new();
        Self::collect_leaves(&self.root, &mut out);
        out
    }

    fn collect_leaves<'a>(node: &'a AreaNode, out: &mut Vec<&'a AreaNode>) {
        if node.children.is_empty() {
            out.push(node);
        } else {
            for child in &node.children {
                Self::collect_leaves(child, out);
            }
        }
    }

    /// Return every node at the given depth. Depth 0 is the root.
    pub fn nodes_at_depth(&self, depth: usize) -> Vec<&AreaNode> {
        let mut out = Vec::new();
        Self::collect_at_depth(&self.root, depth, 0, &mut out);
        out
    }

    fn collect_at_depth<'a>(
        node: &'a AreaNode,
        target: usize,
        current: usize,
        out: &mut Vec<&'a AreaNode>,
    ) {
        if current == target {
            out.push(node);
            return;
        }
        for child in &node.children {
            Self::collect_at_depth(child, target, current + 1, out);
        }
    }

    /// Total number of dossiers in the subtree rooted at `node`.
    #[allow(dead_code)]
    pub fn dossier_count(&self, node: &AreaNode) -> usize {
        Self::subtree_count(node)
    }

    /// Render a human-readable tree view for LLM prompts.
    #[allow(dead_code)]
    pub fn to_index_content(&self) -> String {
        let mut lines = Vec::new();
        Self::render_node(&self.root, 0, &mut lines);
        lines.join("\n")
    }

    fn render_node(node: &AreaNode, depth: usize, lines: &mut Vec<String>) {
        let indent = "  ".repeat(depth);
        let roots = node
            .root_paths
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!("{}- {}: {}", indent, node.name, node.description));
        if !roots.is_empty() {
            lines.push(format!("{}  roots: [{}]", indent, roots));
        }
        lines.push(format!(
            "{}  dossiers: {}",
            indent,
            Self::subtree_count(node)
        ));
        for child in &node.children {
            Self::render_node(child, depth + 1, lines);
        }
    }

    fn subtree_count(node: &AreaNode) -> usize {
        let mut count = node.dossier_paths.len();
        for child in &node.children {
            count += Self::subtree_count(child);
        }
        count
    }
}

/// Convert a name into a URL-friendly id.
///
/// Lowercases the input, replaces runs of non-alphanumeric characters with a
/// single `-`, and guarantees a non-empty result by falling back to `area`.
pub fn slugify_id(name: &str) -> String {
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
    let out = out.trim_matches('-').to_string();
    if out.is_empty() {
        "area".to_string()
    } else {
        out
    }
}

fn normalize_path(path: &Path) -> PathBuf {
    PathBuf::from(path_to_normalized_string(path))
}

fn path_to_normalized_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Make `path` relative to `project_path` using normalized separator comparison.
fn relative_to_project(project_path: &Path, path: &Path) -> Option<PathBuf> {
    relative_to_project_str(project_path, &path.to_string_lossy())
}

fn relative_to_project_str(project_path: &Path, raw: &str) -> Option<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let raw_path = PathBuf::from(trimmed.replace('\\', "/"));
    if raw_path
        .components()
        .any(|c| c == std::path::Component::ParentDir)
    {
        return None;
    }
    let project_is_current = project_path.as_os_str().is_empty() || project_path.as_os_str() == ".";
    if project_is_current {
        if raw_path.is_absolute() {
            return None;
        }
        return Some(empty_to_dot(&raw_path));
    }
    if let Ok(rel) = raw_path.strip_prefix(project_path) {
        return Some(empty_to_dot(rel));
    }
    if !raw_path.is_absolute() {
        let candidate = project_path.join(&raw_path);
        if candidate.starts_with(project_path)
            && let Ok(rel) = candidate.strip_prefix(project_path)
        {
            return Some(empty_to_dot(rel));
        }
    }
    None
}

fn empty_to_dot(path: &Path) -> PathBuf {
    if path.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        path.to_path_buf()
    }
}

#[allow(dead_code)]
fn path_is_under(child: &str, root: &str) -> bool {
    if root == "." {
        return true;
    }
    if child == root {
        return true;
    }
    if child.starts_with(root) {
        let remainder = &child[root.len()..];
        return remainder.starts_with('/');
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DirectoryDossier, DirectoryPurpose};

    fn dossier(path: &str) -> DirectoryDossier {
        DirectoryDossier {
            path: PathBuf::from(path),
            name: Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            purpose: DirectoryPurpose::Core,
            file_count: 1,
            subdirectory_count: 0,
            importance_score: 0.5,
            summary: "summary".to_string(),
            key_files: vec![],
            file_insights: vec![],
        }
    }

    fn mapping(name: &str, roots: &[&str]) -> MappedArea {
        MappedArea {
            name: name.to_string(),
            description: format!("The {} area", name),
            root_paths: roots.iter().map(|s| s.to_string()).collect(),
            rationale: "mapped by LLM".to_string(),
        }
    }

    #[test]
    fn test_tree_from_dossiers_assigns_areas_and_root() {
        let project_path = Path::new("example-project");
        let dossiers = vec![
            dossier("example-project/src/core"),
            dossier("example-project/src/api"),
            dossier("example-project/tests"),
            dossier("example-project/docs"),
        ];
        let report = AreaMappingReport {
            areas: vec![
                mapping("Core", &["src/core"]),
                mapping("API", &["src/api"]),
                mapping("Docs", &["docs"]),
            ],
        };

        let tree = AreaTree::from_dossiers(project_path, &dossiers, &report);

        assert_eq!(tree.root.name, "example-project");
        assert_eq!(tree.root.children.len(), 3);
        assert_eq!(tree.leaves().len(), 3);

        let root_dossiers: Vec<_> = tree
            .root
            .dossier_paths
            .iter()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .collect();
        assert_eq!(root_dossiers, vec!["tests"]);

        let core = tree
            .nodes_at_depth(1)
            .into_iter()
            .find(|n| n.name == "Core")
            .expect("Core area missing");
        assert_eq!(core.dossier_paths.len(), 1);
        assert_eq!(
            core.dossier_paths[0].to_string_lossy().replace('\\', "/"),
            "src/core"
        );

        assert_eq!(tree.dossier_count(&tree.root), 4);
        assert_eq!(tree.dossier_count(core), 1);
    }

    #[test]
    fn test_nodes_at_depth() {
        let project_path = Path::new("example-project");
        let dossiers = vec![dossier("example-project/src/core")];
        let report = AreaMappingReport {
            areas: vec![mapping("Core", &["src/core"])],
        };
        let tree = AreaTree::from_dossiers(project_path, &dossiers, &report);

        assert_eq!(tree.nodes_at_depth(0).len(), 1);
        assert_eq!(tree.nodes_at_depth(1).len(), 1);
        assert!(tree.nodes_at_depth(2).is_empty());
    }

    #[test]
    fn test_to_index_content_includes_names_and_counts() {
        let project_path = Path::new("example-project");
        let dossiers = vec![dossier("example-project/src/core")];
        let report = AreaMappingReport {
            areas: vec![mapping("Core", &["src/core"])],
        };
        let tree = AreaTree::from_dossiers(project_path, &dossiers, &report);
        let content = tree.to_index_content();

        assert!(content.contains("example-project"));
        assert!(content.contains("Core"));
        assert!(content.contains("dossiers: 1"));
    }

    #[test]
    fn test_slugify_id() {
        assert_eq!(slugify_id("Core API"), "core-api");
        assert_eq!(slugify_id("Hello--World!!!"), "hello-world");
        assert_eq!(slugify_id("  "), "area");
        assert_eq!(slugify_id("Module_2.0"), "module-2-0");
    }

    #[test]
    fn test_path_is_under() {
        assert!(path_is_under("src/core/lib.rs", "src/core"));
        assert!(path_is_under("src/core", "src/core"));
        assert!(!path_is_under("src/core2", "src/core"));
        assert!(path_is_under("anything", "."));
    }
}
