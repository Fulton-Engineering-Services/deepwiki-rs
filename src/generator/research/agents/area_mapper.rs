use anyhow::{Result, anyhow};
use async_trait::async_trait;
use std::path::Path;

use crate::generator::agent_executor::{AgentExecuteParams, extract};
use crate::generator::context::GeneratorContext;
use crate::generator::preprocess::memory::{MemoryScope as PreprocessMemoryScope, ScopedKeys};
use crate::generator::research::area_tree::{AreaMappingReport, AreaTree};
use crate::generator::research::memory::MemoryScope;
use crate::generator::step_forward_agent::{
    AgentDataConfig, DataSource, FormatterConfig, LLMCallMode, PromptTemplate, StepForwardAgent,
};
use crate::types::{CodeAndDirectoryInsights, DirectoryDossier};

/// Coarse-grained area mapper for large codebases.
///
/// Consumes the full set of directory dossiers produced during preprocessing and
/// asks an LLM to group top-level directories into 5–25 functional/business
/// areas. The resulting `AreaMappingReport` is then turned into an `AreaTree`.
#[derive(Default)]
pub struct AreaMapperAgent;

const AREA_TREE_MEMORY_KEY: &str = "AreaTree";

#[async_trait]
impl StepForwardAgent for AreaMapperAgent {
    type Output = AreaTree;

    fn agent_type(&self) -> String {
        "AreaMapper".to_string()
    }

    fn memory_scope_key(&self) -> String {
        MemoryScope::STUDIES_RESEARCH.to_string()
    }

    fn data_config(&self) -> AgentDataConfig {
        AgentDataConfig {
            required_sources: vec![DataSource::CODE_INSIGHTS],
            optional_sources: vec![DataSource::PROJECT_STRUCTURE, DataSource::README_CONTENT],
        }
    }

    fn prompt_template(&self) -> PromptTemplate {
        PromptTemplate {
            system_prompt: r#"You are a professional software architecture analyst specializing in macro-scale codebase organization.

Your task is to group the top-level directories of a large project into a small set of coarse, functional/business areas.

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
- Each area must have a clear business/functional identity (e.g. "Core Platform Libraries", "User-Facing Web UI", "Infrastructure & Deployment").
- Do NOT group by file extension or programming language.
- "root_paths" must list the directory paths (relative to the project root) that belong to this area.
- Every important top-level directory should belong to exactly one area.
- Keep the number of areas between 5 and 25.
- The "rationale" field should explain why these directories belong together and what role the area plays in the system."#
                .to_string(),
            opening_instruction: "Based on the directory dossier index below, produce a coarse-grained functional area map for this project.".to_string(),
            closing_instruction: r#"
## Analysis Requirements
- Group directories by functional/business concern, not by file extension or technology stack.
- Aim for 5–25 top-level areas.
- Ensure each area has a concise name, a one-paragraph description, and a clear rationale.
- Every directory in the index should be assigned to exactly one area unless it is clearly irrelevant."#
                .to_string(),
            llm_call_mode: LLMCallMode::Extract,
            formatter_config: FormatterConfig {
                code_insights_limit: 0,
                include_source_code: false,
                enable_compression: true,
                ..FormatterConfig::default()
            },
        }
    }

    /// Custom execution: build a dossier index, extract the area mapping, and
    /// materialize an `AreaTree`.
    async fn execute(&self, context: &GeneratorContext) -> Result<Self::Output> {
        // 1. Validate required data sources.
        let config = self.data_config();
        for source in &config.required_sources {
            if let DataSource::MemoryData { scope, key } = source
                && !context.has_memory_data(scope, key).await
            {
                return Err(anyhow!(
                    "Required data source {}:{} is not available",
                    scope,
                    key
                ));
            }
        }

        // 2. Load dossiers.
        let insights: CodeAndDirectoryInsights = context
            .get_from_memory::<CodeAndDirectoryInsights>(
                PreprocessMemoryScope::PREPROCESS,
                ScopedKeys::CODE_INSIGHTS,
            )
            .await
            .ok_or_else(|| anyhow!("CodeAndDirectoryInsights not found in memory"))?;
        let dossiers = &insights.directory_insights;

        if dossiers.is_empty() {
            return Err(anyhow!("No directory dossiers available for area mapping"));
        }

        // 3. Build the dossier index used as the prompt body.
        let index_content = build_dossier_index(dossiers, &context.config.project_path);

        // 4. Call the LLM to extract the area mapping.
        let template = self.prompt_template();
        let language_instruction = context.config.target_language.prompt_instruction();

        let system_prompt = format!("{}\n\n{}", template.system_prompt, language_instruction);
        let user_prompt = format!(
            "{}\n\n## Directory Dossier Index\n{}\n\n{}",
            template.opening_instruction, index_content, template.closing_instruction
        );

        let params = AgentExecuteParams {
            prompt_sys: system_prompt,
            prompt_user: user_prompt,
            cache_scope: format!("{}/{}", self.memory_scope_key(), self.agent_type()),
            log_tag: "AreaMapper".to_string(),
            progress: None,
        };

        println!(
            "🗺️  Mapping coarse areas across {} dossiers...",
            dossiers.len()
        );
        let mapping_report: AreaMappingReport = extract(context, params).await?;

        // 5. Build the area tree.
        let tree = AreaTree::from_dossiers(&context.config.project_path, dossiers, &mapping_report);

        // 6. Print summary.
        println!(
            "🗺️  Mapped {} top-level areas from {} dossiers",
            mapping_report.areas.len(),
            dossiers.len()
        );

        // 7. Store and return.
        context
            .store_to_memory(&self.memory_scope_key(), AREA_TREE_MEMORY_KEY, &tree)
            .await?;

        Ok(tree)
    }
}

/// Build a compact, importance-sorted index of directory dossiers.
fn build_dossier_index(dossiers: &[DirectoryDossier], project_path: &Path) -> String {
    let max_index_chars: usize = 120_000;
    let max_summary_chars: usize = 280;

    let mut sorted: Vec<&DirectoryDossier> = dossiers.iter().collect();
    sorted.sort_by(|a, b| {
        b.importance_score
            .partial_cmp(&a.importance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_dossiers = sorted.len();
    let mut lines: Vec<String> = Vec::with_capacity(sorted.len());
    let mut total_len: usize = 0;

    for dossier in &sorted {
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
