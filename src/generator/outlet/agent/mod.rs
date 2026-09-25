//! Agent-focused output set: short, interlinked, progressively-discoverable
//! Markdown written to `<OUT_DIR>/.agent-content/`.
//!
//! This is a parallel output to the human-oriented docs. It is generated
//! deterministically from research already held in memory (no LLM calls): a
//! recursive area tree down to tightly-scoped per-file pages, short topic pages,
//! and dedicated Mermaid diagram files at each level. Every page links directly
//! into the codebase via relative paths so an agent can go from the map straight
//! to a source file without pulling a 1k-line human report into context.
//!
//! The set is written *after* the human docs (see `workflow.rs`); `DiskOutlet`
//! wipes the output directory at its start, so agent content must be written
//! last to survive. A final `MermaidFixer` pass runs over `.agent-content/`
//! because the human fix pass executes before this directory exists.

pub mod diagrams;
pub mod links;
pub mod model;
pub mod render;

use anyhow::Result;
use std::fs;
use std::path::Path;

use crate::generator::context::GeneratorContext;
use crate::generator::outlet::MermaidFixer;
use crate::generator::outlet::agent::model::{
    AgentOptions, AgentSite, OwnedKeyModule, ResearchBundle, build_site,
};
use crate::generator::outlet::agent::render::{RenderContext, render_diagram, render_page};
use crate::generator::research::area_tree::AreaTree;
use crate::generator::research::memory::{MemoryRetriever, MemoryScope};
use crate::generator::research::types::{
    AgentType as ResearchAgentType, BoundaryAnalysisReport, DatabaseOverviewReport,
    DomainModulesReport, KeyModuleReport, SystemContextReport,
};
use crate::generator::preprocess::memory::{MemoryScope as PreprocessScope, ScopedKeys};
use crate::types::CodeAndDirectoryInsights;
use crate::types::FileInsight;

/// Directory name for the agent set, nested under the output path.
pub const AGENT_DIR: &str = ".agent-content";

/// Writes the agent-focused markdown set. Implements the shared [`Outlet`] trait
/// so it can be invoked from the workflow like the other outlets.
pub struct AgentContentOutlet {
    options: AgentOptions,
}

impl Default for AgentContentOutlet {
    fn default() -> Self {
        Self::new(AgentOptions::default())
    }
}

impl AgentContentOutlet {
    pub fn new(options: AgentOptions) -> Self {
        Self { options }
    }

    /// Build the outlet's options from the generator config.
    pub fn from_context(context: &GeneratorContext) -> Self {
        let opts = AgentOptions {
            diagrams: context.config.agent_content_diagrams,
            ..AgentOptions::default()
        };
        Self::new(opts)
    }
}

impl crate::generator::outlet::Outlet for AgentContentOutlet {
    async fn save(&self, context: &GeneratorContext) -> Result<()> {
        let agent_root = context.config.output_path.join(AGENT_DIR);
        println!("\n🤖 Generating agent-focused content set...");

        // Collect research from memory into a single bundle.
        let bundle = collect_bundle(context).await;

        // Build the page + diagram tree.
        let site = build_site(&bundle, &self.options);

        // Reset the agent directory so stale files from prior runs disappear.
        if agent_root.exists() {
            fs::remove_dir_all(&agent_root)?;
        }
        fs::create_dir_all(&agent_root)?;

        // Write every page and diagram.
        write_site(&site, context, &agent_root)?;

        // Fix Mermaid syntax within the agent set only.
        if self.options.diagrams && !site.diagrams.is_empty() {
            MermaidFixer::fix_mermaid_charts(context, &agent_root).await?;
        }

        println!(
            "✅ Agent content for '{}': {} page(s), {} diagram(s) → {}",
            site.project_name,
            site.pages.len(),
            site.diagrams.len(),
            agent_root.display()
        );
        Ok(())
    }
}

/// Load all research artifacts the agent set needs from memory.
async fn collect_bundle(context: &GeneratorContext) -> ResearchBundle {
    let mut bundle = ResearchBundle::default();

    if let Some(v) = context.get_research(&ResearchAgentType::SystemContextResearcher.to_string()).await
        && let Ok(r) = serde_json::from_value::<SystemContextReport>(v.clone())
    {
        bundle.project_name = if r.project_name.trim().is_empty() {
            context.config.get_project_name()
        } else {
            r.project_name.clone()
        };
        bundle.system_context = Some(r);
    }
    if bundle.project_name.is_empty() {
        bundle.project_name = context.config.get_project_name();
    }

    if let Some(v) = context.get_research(&ResearchAgentType::DomainModulesDetector.to_string()).await
        && let Ok(r) = serde_json::from_value::<DomainModulesReport>(v)
    {
        bundle.domain_modules = Some(r);
    }

    if let Some(v) = context.get_research(&ResearchAgentType::BoundaryAnalyzer.to_string()).await
        && let Ok(r) = serde_json::from_value::<BoundaryAnalysisReport>(v)
    {
        bundle.boundary = Some(r);
    }

    if let Some(v) = context.get_research(&ResearchAgentType::DatabaseOverviewAnalyzer.to_string()).await
        && let Ok(r) = serde_json::from_value::<DatabaseOverviewReport>(v)
    {
        bundle.database = Some(r);
    }

    // The refined (arbitrary-depth) area tree is stored back under "AreaTree".
    if let Some(tree) = context
        .get_from_memory::<AreaTree>(MemoryScope::STUDIES_RESEARCH, "AreaTree")
        .await
    {
        bundle.area_tree = Some(tree);
    }

    // Key modules: per-area traceability keys carry the area id; the merged
    // report list is the fallback. Key format is `KeyModulesInsight_<area.id>_*`.
    bundle.modules = collect_modules(context).await;

    // File insights for tightly-scoped per-file pages.
    bundle.file_insights = collect_file_insights(context).await;

    bundle
}

/// Gather key-module reports tagged with their owning area id where available.
async fn collect_modules(context: &GeneratorContext) -> Vec<OwnedKeyModule> {
    let mut out: Vec<OwnedKeyModule> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Per-area traceability keys: `KeyModulesInsight_<area.id>_<domain.name>`.
    if let Some(keys) = Some(context.list_memory_keys(MemoryScope::STUDIES_RESEARCH).await) {
        for key in keys {
            let Some(rest) = key.strip_prefix("KeyModulesInsight_") else {
                continue;
            };
            // area.id is slugified (no underscores); the first `_` ends it.
            let Some((area_id, _)) = rest.split_once('_') else {
                continue;
            };
            if area_id.is_empty() {
                continue;
            }
            if let Some(report) = context
                .get_from_memory::<KeyModuleReport>(MemoryScope::STUDIES_RESEARCH, &key)
                .await
            {
                let mark = report.domain_name.clone();
                if seen.insert(mark) {
                    out.push(OwnedKeyModule { area_id: Some(area_id.to_string()), report });
                }
            }
        }
    }

    // Merged report list (non-macro-scan path, and any not covered above).
    if let Some(v) = context.get_research(&ResearchAgentType::KeyModulesInsight.to_string()).await
        && let Ok(reports) = serde_json::from_value::<Vec<KeyModuleReport>>(v)
    {
        for report in reports {
            if seen.insert(report.domain_name.clone()) {
                out.push(OwnedKeyModule { area_id: None, report });
            }
        }
    }

    out
}

/// Gather per-file insights from preprocessing for tight file pages.
async fn collect_file_insights(context: &GeneratorContext) -> Vec<FileInsight> {
    let Some(insights) = context
        .get_from_memory::<CodeAndDirectoryInsights>(PreprocessScope::PREPROCESS, ScopedKeys::CODE_INSIGHTS)
        .await
    else {
        return Vec::new();
    };
    insights
        .directory_insights
        .iter()
        .flat_map(|d| d.file_insights.iter().cloned())
        .collect()
}

/// Write all pages and diagram files under the agent root.
fn write_site(site: &AgentSite, context: &GeneratorContext, agent_root: &Path) -> Result<()> {
    let project_root = context.config.project_path.clone();
    let rctx = RenderContext {
        project_root: &project_root,
        agent_root,
    };

    for page in &site.pages {
        let path = agent_root.join(&page.rel_path);
        write_file(&path, &render_page(page, &rctx))?;
    }

    for diagram in &site.diagrams {
        let path = agent_root.join(&diagram.rel_path);
        write_file(&path, &render_diagram(diagram, &rctx))?;
    }

    Ok(())
}

/// Write a single file, creating parent directories as needed.
fn write_file(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::outlet::agent::model::build_site;

    #[test]
    fn outlet_is_constructible_and_writes_directory_constant() {
        assert_eq!(AGENT_DIR, ".agent-content");
        let outlet = AgentContentOutlet::default();
        assert!(outlet.options.diagrams);
    }

    #[test]
    fn build_and_write_roundtrip() {
        use crate::generator::outlet::agent::model::{
            AgentPage, PageKind, ResearchBundle,
        };
        use crate::generator::research::area_tree::{AreaNode, AreaTree};
        use std::path::PathBuf;

        let bundle = ResearchBundle {
            project_name: "proj".into(),
            area_tree: Some(AreaTree {
                root: AreaNode {
                    id: "root".into(),
                    name: "proj".into(),
                    description: "root".into(),
                    root_paths: vec![PathBuf::from(".")],
                    children: vec![AreaNode {
                        id: "core".into(),
                        name: "Core".into(),
                        description: "core".into(),
                        root_paths: vec![PathBuf::from("core")],
                        children: Vec::new(),
                        dossier_paths: Vec::new(),
                    }],
                    dossier_paths: Vec::new(),
                },
            }),
            ..Default::default()
        };
        let site = build_site(&bundle, &AgentOptions::default());
        // Index + topics + one area page at minimum.
        assert!(site.pages.iter().any(|p: &AgentPage| p.rel_path == "index.md"));
        assert!(site.pages.iter().any(|p: &AgentPage| p.kind == PageKind::Area));
    }
}

