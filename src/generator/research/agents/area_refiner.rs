use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use crate::generator::agent_executor::{AgentExecuteParams, extract};
use crate::generator::context::GeneratorContext;
use crate::generator::preprocess::memory::{MemoryScope as PreprocessMemoryScope, ScopedKeys};
use crate::generator::research::area_tree::{AreaMappingReport, AreaNode, AreaTree, MappedArea};
use crate::generator::research::memory::MemoryScope;
use crate::types::{CodeAndDirectoryInsights, DirectoryDossier};

const AREA_TREE_MEMORY_KEY: &str = "AreaTree";

/// Recursive area refiner.
///
/// Walks the coarse `AreaTree` produced by `AreaMapperAgent` and, for any node
/// that still contains too many dossiers, asks an LLM to split it into smaller
/// sub-areas. The tree is mutated in place and stored back to memory.
#[derive(Default)]
pub struct AreaRefinerAgent;

impl AreaRefinerAgent {
    pub async fn refine_tree(context: &GeneratorContext, tree: &mut AreaTree) -> Result<()> {
        let insights: CodeAndDirectoryInsights = context
            .get_from_memory::<CodeAndDirectoryInsights>(
                PreprocessMemoryScope::PREPROCESS,
                ScopedKeys::CODE_INSIGHTS,
            )
            .await
            .ok_or_else(|| anyhow!("CodeAndDirectoryInsights not found in memory"))?;

        let dossiers = insights.directory_insights;
        let dossier_map: HashMap<PathBuf, DirectoryDossier> = dossiers
            .into_iter()
            .map(|d| (normalize_path(&d.path), d))
            .collect();

        let project_path = context.config.project_path.clone();
        let max_depth = context.config.macro_scan.max_depth;
        let scope_max_dossiers = context.config.macro_scan.scope_max_dossiers;

        println!(
            "🔧 Refining area tree (max_depth={}, scope_max_dossiers={})...",
            max_depth, scope_max_dossiers
        );

        // Take ownership of the root so the recursive helper can consume and
        // rebuild nodes without borrow conflicts.
        let dummy_root = AreaNode {
            id: "__refiner_dummy__".to_string(),
            name: String::new(),
            description: String::new(),
            root_paths: Vec::new(),
            children: Vec::new(),
            dossier_paths: Vec::new(),
        };
        let root = std::mem::replace(&mut tree.root, dummy_root);

        let refined_root = Self::refine_node(
            context,
            root,
            0,
            &dossier_map,
            &project_path,
            max_depth,
            scope_max_dossiers,
        )
        .await?;

        tree.root = refined_root;

        // Store the refined tree back to memory.
        context
            .store_to_memory(MemoryScope::STUDIES_RESEARCH, AREA_TREE_MEMORY_KEY, tree)
            .await?;

        println!("🔧 Area tree refinement complete.");
        Ok(())
    }

    fn refine_node<'a>(
        context: &'a GeneratorContext,
        node: AreaNode,
        depth: usize,
        dossier_map: &'a HashMap<PathBuf, DirectoryDossier>,
        project_path: &'a Path,
        max_depth: usize,
        scope_max_dossiers: usize,
    ) -> Pin<Box<dyn Future<Output = Result<AreaNode>> + Send + 'a>> {
        Box::pin(async move {
            // Base case: depth limit reached or node is small enough.
            if depth >= max_depth || node.dossier_paths.len() <= scope_max_dossiers {
                return Ok(node);
            }

            let index_content = build_node_index(&node, dossier_map, project_path);
            if index_content.is_empty() {
                println!(
                    "⚠️  No dossier index could be built for area '{}'; skipping refinement.",
                    node.name
                );
                return Ok(node);
            }

            let system_prompt = r#"You are a professional software architecture analyst specializing in subsystem decomposition.

Your task is to split one coarse area into a small set of more focused sub-areas.

You MUST output strict JSON only (no markdown, no code fences, no prose outside JSON).
Return exactly this structure:

{
  "areas": [
    {
      "name": "string",
      "description": "string",
      "root_paths": ["string"],
      "rationale": "string"
    }
  ]
}

Rules:
- Always include the top-level "areas" key.
- Produce between 2 and 8 sub-areas.
- Each sub-area must have a clear functional identity.
- "root_paths" must list directory paths (relative to the project root) that belong to this sub-area.
- Every directory in the input should be assigned to exactly one sub-area.
- Use concise names and one-paragraph descriptions."#
                .to_string();

            let user_prompt = format!(
                "## Refinement Task\n\nSplit the area '{}' into focused sub-areas.\n\n### Area Description\n{}\n\n### Directory Dossier Index\n{}\n\n### Requirements\n- Produce 2–8 sub-areas.\n- Group by functional/business concern, not by file extension.\n- Every directory in the index must belong to exactly one sub-area.",
                node.name, node.description, index_content
            );

            let language_instruction = context.config.target_language.prompt_instruction();
            let system_prompt = format!("{}\n\n{}", system_prompt, language_instruction);
            let user_prompt = format!("{}\n\n{}", user_prompt, language_instruction);

            let params = AgentExecuteParams {
                prompt_sys: system_prompt,
                prompt_user: user_prompt,
                cache_scope: format!("{}/AreaRefiner/{}", MemoryScope::STUDIES_RESEARCH, node.id),
                log_tag: format!("AreaRefiner/{}", node.name),
                progress: None,
            };

            println!(
                "🔧 Refining area '{}' (depth={}, dossiers={})...",
                node.name,
                depth,
                node.dossier_paths.len()
            );

            let sub_area_report: AreaMappingReport = match extract(context, params).await {
                Ok(report) => report,
                Err(e) => {
                    println!(
                        "⚠️  Failed to refine area '{}': {}. Keeping node unrefined.",
                        node.name, e
                    );
                    return Ok(node);
                }
            };

            // Partition this node's dossiers among the returned sub-areas.
            let children = partition_dossiers_to_children(
                &node.dossier_paths,
                &sub_area_report.areas,
                &node.id,
                project_path,
            );

            if children.is_empty() {
                println!(
                    "⚠️  Area '{}' refinement produced no children; keeping node unrefined.",
                    node.name
                );
                return Ok(node);
            }

            // Recursively refine each new child.
            let mut refined_children = Vec::with_capacity(children.len());
            for child in children {
                let refined_child = Self::refine_node(
                    context,
                    child,
                    depth + 1,
                    dossier_map,
                    project_path,
                    max_depth,
                    scope_max_dossiers,
                )
                .await?;
                refined_children.push(refined_child);
            }

            Ok(AreaNode {
                children: refined_children,
                ..node
            })
        })
    }
}

/// Build an importance-sorted index of the dossiers that belong to `node`.
fn build_node_index(
    node: &AreaNode,
    dossier_map: &HashMap<PathBuf, DirectoryDossier>,
    project_path: &Path,
) -> String {
    let max_summary_chars: usize = 280;
    let max_index_chars: usize = 80_000;

    let mut dossiers: Vec<&DirectoryDossier> = node
        .dossier_paths
        .iter()
        .filter_map(|path| dossier_map.get(&normalize_path(path)))
        .collect();

    let total_dossiers = dossiers.len();

    dossiers.sort_by(|a, b| {
        b.importance_score
            .partial_cmp(&a.importance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut lines: Vec<String> = Vec::with_capacity(dossiers.len());
    let mut total_len: usize = 0;

    for dossier in &dossiers {
        let relative_path = dossier
            .path
            .strip_prefix(project_path)
            .unwrap_or(&dossier.path)
            .to_string_lossy()
            .replace('\\', "/");

        let summary = crate::utils::truncate_at_char_boundary(&dossier.summary, max_summary_chars);
        let line = format!(
            "- {} | purpose: {:?} | importance: {:.2} | summary: {}",
            relative_path, dossier.purpose, dossier.importance_score, summary
        );

        total_len += line.len() + 1;
        if total_len > max_index_chars && !lines.is_empty() {
            let remaining = total_dossiers - lines.len();
            lines.push(format!(
                "\n[{} additional dossiers omitted to stay within context budget]",
                remaining
            ));
            break;
        }
        lines.push(line);
    }

    lines.join("\n")
}

/// Assign each dossier path to the most specific matching sub-area.
fn partition_dossiers_to_children(
    node_dossiers: &[PathBuf],
    sub_areas: &[MappedArea],
    parent_id: &str,
    project_path: &Path,
) -> Vec<AreaNode> {
    let mut buckets: Vec<Vec<PathBuf>> = vec![Vec::new(); sub_areas.len()];

    for dossier_path in node_dossiers {
        let normalized = normalize_path(dossier_path);
        let dossier_full_str = normalized.to_string_lossy().replace('\\', "/");
        let dossier_str = normalized
            .strip_prefix(project_path)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| dossier_full_str.clone());

        let mut best_idx: Option<usize> = None;
        let mut best_len: usize = 0;

        for (idx, area) in sub_areas.iter().enumerate() {
            for root in &area.root_paths {
                let root_str = PathBuf::from(root).to_string_lossy().replace('\\', "/");

                if (dossier_str == root_str
                    || dossier_str.starts_with(&format!("{}/", root_str))
                    || dossier_full_str == root_str
                    || dossier_full_str.starts_with(&format!("{}/", root_str)))
                    && root_str.len() > best_len
                {
                    best_len = root_str.len();
                    best_idx = Some(idx);
                }
            }
        }

        let idx = best_idx.unwrap_or(0);
        buckets[idx].push(normalized);
    }

    sub_areas
        .iter()
        .enumerate()
        .filter_map(|(idx, area)| {
            let dossiers = &buckets[idx];
            if dossiers.is_empty() {
                // Skip empty sub-areas: the LLM may have invented a path that
                // does not match any real dossier.
                return None;
            }

            let root_paths: Vec<PathBuf> = area.root_paths.iter().map(PathBuf::from).collect();

            let slug = slugify(&area.name);
            let id = format!("{}_{}", parent_id, slug);

            Some(AreaNode {
                id,
                name: area.name.clone(),
                description: area.description.clone(),
                root_paths,
                children: Vec::new(),
                dossier_paths: dossiers.clone(),
            })
        })
        .collect()
}

fn normalize_path(path: &Path) -> PathBuf {
    path.to_string_lossy().replace('\\', "/").into()
}

fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.to_lowercase().chars() {
        if ch.is_alphanumeric() {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_end_matches('_').to_string()
}
