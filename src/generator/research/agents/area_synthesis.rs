use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::generator::agent_executor::{AgentExecuteParams, extract};
use crate::generator::compose::memory::MemoryScope as DocumentationMemoryScope;
use crate::generator::context::GeneratorContext;
use crate::generator::research::area_tree::AreaTree;
use crate::generator::research::memory::{MemoryRetriever, MemoryScope};
use crate::generator::research::types::{AgentType, KeyModuleReport};
use crate::utils::threads::do_parallel_with_limit;

/// Integration report produced by the LLM for one internal area node.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct AreaIntegrationReport {
    pub summary: String,
    pub cross_area_dependencies: Vec<String>,
    pub interfaces: Vec<String>,
    pub architecture_role: String,
}

/// Synthesizes integration documentation by walking the area tree bottom-up.
#[derive(Default)]
pub struct AreaSynthesisAgent;

#[derive(Debug, Clone)]
struct ChildTask {
    id: String,
    name: String,
    is_leaf: bool,
    description: String,
}

#[derive(Debug, Clone)]
struct SynthesisTask {
    id: String,
    name: String,
    description: String,
    children: Vec<ChildTask>,
}

impl AreaSynthesisAgent {
    pub async fn synthesize(
        context: &GeneratorContext,
        tree: &AreaTree,
    ) -> Result<Vec<(String, String)>> {
        let max_depth = compute_max_depth(tree);
        if max_depth == 0 {
            println!("ℹ️  Area tree has no depth; nothing to synthesize.");
            return Ok(Vec::new());
        }

        let mut results: Vec<(String, String)> = Vec::new();
        let max_parallels = context.config.llm.max_parallels;

        // Process internal nodes from the deepest level up to (but not including)
        // the root level.
        for depth in (1..max_depth).rev() {
            let nodes = tree.nodes_at_depth(depth);
            let internal_nodes: Vec<_> = nodes
                .into_iter()
                .filter(|n| !n.children.is_empty())
                .collect();

            if internal_nodes.is_empty() {
                continue;
            }

            println!(
                "🧩 Synthesizing {} area integration(s) at depth {}...",
                internal_nodes.len(),
                depth
            );

            let futures: Vec<_> = internal_nodes
                .into_iter()
                .map(|node| {
                    let ctx = context.clone();
                    let task = SynthesisTask {
                        id: node.id.clone(),
                        name: node.name.clone(),
                        description: node.description.clone(),
                        children: node
                            .children
                            .iter()
                            .map(|c| ChildTask {
                                id: c.id.clone(),
                                name: c.name.clone(),
                                is_leaf: c.children.is_empty(),
                                description: c.description.clone(),
                            })
                            .collect(),
                    };
                    async move { (task.id.clone(), synthesize_node(&ctx, task).await) }
                })
                .collect();

            let node_results = do_parallel_with_limit(futures, max_parallels).await;
            for (node_id, node_result) in node_results {
                match node_result {
                    Ok(markdown) => {
                        let key = format!("AreaIntegration_{}", node_id);
                        context
                            .store_to_memory(
                                DocumentationMemoryScope::DOCUMENTATION,
                                &key,
                                &markdown,
                            )
                            .await?;
                        results.push((key, markdown));
                    }
                    Err(e) => {
                        println!(
                            "⚠️  Area synthesis failed for node '{}': {}. Skipping.",
                            node_id, e
                        );
                    }
                }
            }
        }

        // Finally, synthesize the root integration across its immediate children.
        let root = &tree.root;
        if !root.children.is_empty() {
            println!("🧩 Synthesizing top-level area integration...");
            let root_task = SynthesisTask {
                id: "root".to_string(),
                name: root.name.clone(),
                description: root.description.clone(),
                children: root
                    .children
                    .iter()
                    .map(|c| ChildTask {
                        id: c.id.clone(),
                        name: c.name.clone(),
                        is_leaf: c.children.is_empty(),
                        description: c.description.clone(),
                    })
                    .collect(),
            };

            match synthesize_node(context, root_task).await {
                Ok(markdown) => {
                    let key = "AreaIntegration_root".to_string();
                    context
                        .store_to_memory(DocumentationMemoryScope::DOCUMENTATION, &key, &markdown)
                        .await?;
                    results.push((key, markdown));
                }
                Err(e) => {
                    println!("⚠️  Top-level area synthesis failed: {}. Skipping.", e);
                }
            }
        }

        println!(
            "🧩 Area synthesis produced {} integration document(s).",
            results.len()
        );
        Ok(results)
    }
}

async fn synthesize_node(context: &GeneratorContext, task: SynthesisTask) -> Result<String> {
    let system_prompt = r#"You are a professional software architecture writer. Synthesize the integration of several sub-areas or modules into a coherent architectural overview.

You MUST output strict JSON only (no markdown, no code fences, no prose outside JSON).
Return exactly this structure:

{
  "summary": "string",
  "cross_area_dependencies": ["string"],
  "interfaces": ["string"],
  "architecture_role": "string"
}

Rules:
- "summary" is a concise paragraph describing how the child areas/modules fit together.
- "cross_area_dependencies" lists specific dependencies or data/control flows between children.
- "interfaces" lists the integration surfaces (APIs, events, shared libraries, databases, etc.).
- "architecture_role" explains the role this combined area plays in the overall system.
- Use plain strings and string arrays only."#
        .to_string();

    let child_summaries = gather_child_summaries(context, &task).await?;
    let system_context = load_system_context(context).await;

    let mut user_prompt = format!(
        "## Area Integration Task\n\nSynthesize the integration of area '{}'.\n\n### Area Description\n{}\n\n### Child Areas / Modules\n{}",
        task.name, task.description, child_summaries
    );

    if let Some(ctx) = system_context {
        user_prompt.push_str("\n\n### System-Wide Context\n");
        user_prompt.push_str(&ctx);
    }

    user_prompt.push_str(
        "\n\n### Requirements\n\
        - Describe how the child areas fit together.\n\
        - Identify cross-area dependencies and interfaces.\n\
        - Explain the architectural role of this area in the overall system.",
    );

    let language_instruction = context.config.target_language.prompt_instruction();
    let system_prompt = format!("{}\n\n{}", system_prompt, language_instruction);
    let user_prompt = format!("{}\n\n{}", user_prompt, language_instruction);

    let params = AgentExecuteParams {
        prompt_sys: system_prompt,
        prompt_user: user_prompt,
        cache_scope: format!(
            "{}/AreaSynthesis/{}",
            DocumentationMemoryScope::DOCUMENTATION,
            task.id
        ),
        log_tag: format!("AreaSynthesis/{}", task.name),
        progress: None,
    };

    let report: AreaIntegrationReport = extract(context, params).await?;
    Ok(render_integration_markdown(&task.name, &report))
}

async fn gather_child_summaries(
    context: &GeneratorContext,
    task: &SynthesisTask,
) -> Result<String> {
    let research_keys = context
        .list_memory_keys(MemoryScope::STUDIES_RESEARCH)
        .await;
    let mut lines: Vec<String> = Vec::with_capacity(task.children.len());

    for child in &task.children {
        if child.is_leaf {
            let prefix = format!("KeyModulesInsight_{}_", child.id);
            let domain_keys: Vec<&String> = research_keys
                .iter()
                .filter(|k| k.starts_with(&prefix))
                .collect();

            if domain_keys.is_empty() {
                lines.push(format!("- **{}**: {}", child.name, child.description));
                continue;
            }

            let mut domain_lines: Vec<String> = Vec::with_capacity(domain_keys.len());
            for key in domain_keys {
                if let Some(report) = context
                    .get_from_memory::<KeyModuleReport>(MemoryScope::STUDIES_RESEARCH, key)
                    .await
                {
                    domain_lines.push(format!(
                        "  - {} ({}): {}",
                        report.domain_name, report.module_name, report.module_description
                    ));
                }
            }

            if domain_lines.is_empty() {
                lines.push(format!("- **{}**: {}", child.name, child.description));
            } else {
                lines.push(format!("- **{}**:", child.name));
                lines.extend(domain_lines);
            }
        } else {
            let integration_key = format!("AreaIntegration_{}", child.id);
            if let Some(markdown) = context
                .get_from_memory::<String>(
                    DocumentationMemoryScope::DOCUMENTATION,
                    &integration_key,
                )
                .await
            {
                let summary = first_paragraph(&markdown);
                lines.push(format!("- **{}** (sub-area): {}", child.name, summary));
            } else {
                lines.push(format!("- **{}**: {}", child.name, child.description));
            }
        }
    }

    Ok(lines.join("\n"))
}

async fn load_system_context(context: &GeneratorContext) -> Option<String> {
    let value = context
        .get_research(&AgentType::SystemContextResearcher.to_string())
        .await?;

    #[derive(Debug, Deserialize, Default)]
    struct BriefContext {
        #[serde(default)]
        project_name: String,
        #[serde(default)]
        project_description: String,
        #[serde(default)]
        business_value: String,
    }

    let brief: BriefContext = serde_json::from_value(value).unwrap_or_default();
    let mut parts = Vec::new();
    if !brief.project_name.is_empty() {
        parts.push(format!("Project: {}", brief.project_name));
    }
    if !brief.project_description.is_empty() {
        parts.push(format!("Description: {}", brief.project_description));
    }
    if !brief.business_value.is_empty() {
        parts.push(format!("Business value: {}", brief.business_value));
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn render_integration_markdown(area_name: &str, report: &AreaIntegrationReport) -> String {
    let mut md = format!("# {} Integration\n\n{}", area_name, report.summary);

    if !report.cross_area_dependencies.is_empty() {
        md.push_str("\n\n## Cross-Area Dependencies\n");
        for dep in &report.cross_area_dependencies {
            md.push_str(&format!("- {}\n", dep));
        }
    }

    if !report.interfaces.is_empty() {
        md.push_str("\n## Interfaces\n");
        for iface in &report.interfaces {
            md.push_str(&format!("- {}\n", iface));
        }
    }

    if !report.architecture_role.is_empty() {
        md.push_str("\n## Architecture Role\n");
        md.push_str(&report.architecture_role);
    }

    md
}

fn first_paragraph(markdown: &str) -> String {
    let trimmed = markdown.trim();
    if let Some(idx) = trimmed.find("\n\n") {
        trimmed[..idx].trim().to_string()
    } else {
        trimmed.to_string()
    }
}

fn compute_max_depth(tree: &AreaTree) -> usize {
    let mut depth = 0;
    loop {
        let nodes = tree.nodes_at_depth(depth);
        if nodes.is_empty() {
            return depth;
        }
        depth += 1;
    }
}
