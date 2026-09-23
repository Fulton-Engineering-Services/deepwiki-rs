use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::{Result, anyhow};

use crate::generator::agent_executor::{AgentExecuteParams, extract};
use crate::generator::compose::memory::MemoryScope as ComposeMemoryScope;
use crate::generator::context::GeneratorContext;
use crate::generator::preprocess::memory::{MemoryScope as PreprocessMemoryScope, ScopedKeys};
use crate::generator::research::agents::architecture_researcher::ArchitectureResearcher;
use crate::generator::research::agents::area_mapper::AreaMapperAgent;
use crate::generator::research::agents::area_refiner::AreaRefinerAgent;
use crate::generator::research::agents::area_synthesis::AreaSynthesisAgent;
use crate::generator::research::agents::boundary_analyzer::BoundaryAnalyzer;
use crate::generator::research::agents::database_overview_analyzer::DatabaseOverviewAnalyzer;
use crate::generator::research::agents::domain_modules_detector::DomainModulesDetector;
use crate::generator::research::agents::key_modules_insight::KeyModulesInsight;
use crate::generator::research::agents::system_context_researcher::SystemContextResearcher;
use crate::generator::research::agents::workflow_researcher::WorkflowResearcher;
use crate::generator::research::memory::MemoryScope as ResearchMemoryScope;
use crate::generator::research::types::{
    AgentType, DomainModule, DomainModulesReport, KeyModuleReport, SubModule,
};
use crate::generator::scoped_evidence_filter::filter_insights_by_paths;
use crate::generator::step_forward_agent::StepForwardAgent;
use crate::types::code::CodePurpose;
use crate::types::{CodeAndDirectoryInsights, DirectoryPurpose, FileInsight};
use crate::utils::threads::do_parallel_with_limit;

/// Owned snapshot of an `AreaNode` leaf used to pass area context into async
/// tasks without requiring `AreaNode` to be `Clone`.
struct LeafHandle {
    id: String,
    name: String,
    description: String,
    root_paths: Vec<PathBuf>,
    dossier_paths: Vec<PathBuf>,
}

/// Multi-agent research orchestrator
#[derive(Default)]
pub struct ResearchOrchestrator;

impl ResearchOrchestrator {
    /// Execute all agent analysis pipelines
    pub async fn execute_research_pipeline(&self, context: &GeneratorContext) -> Result<()> {
        println!("🚀 Starting Litho Studies Research investigation pipeline...");

        // First layer: Macro analysis (C1)
        self.execute_agent(&SystemContextResearcher, context)
            .await?;

        // Decide whether to use the hierarchical macro-scan path for large codebases.
        let all_insights = context
            .get_from_memory::<CodeAndDirectoryInsights>(
                PreprocessMemoryScope::PREPROCESS,
                ScopedKeys::CODE_INSIGHTS,
            )
            .await
            .unwrap_or_default();
        let dossier_count = all_insights.directory_insights.len();

        if context.config.macro_scan_enabled(dossier_count) {
            println!(
                "🗺️  Large codebase detected ({} dossiers). Activating hierarchical macro-scan...",
                dossier_count
            );
            self.execute_macro_scan_branch(context, &all_insights)
                .await?;
        } else {
            // Second layer: Meso analysis (C2)
            self.execute_agent(&DomainModulesDetector, context).await?;
            self.execute_agent(&ArchitectureResearcher, context).await?;
            self.execute_agent(&WorkflowResearcher, context).await?;

            // Third layer: Micro analysis (C3-C4)
            self.execute_agent(&KeyModulesInsight, context).await?;
        }

        // Boundary interface analysis
        self.execute_agent(&BoundaryAnalyzer::default(), context)
            .await?;

        // Database overview analysis (only if database files exist)
        if self.has_database_files(context).await {
            self.execute_agent(&DatabaseOverviewAnalyzer::default(), context)
                .await?;
        }

        println!("✓ Litho Studies Research pipeline execution completed");

        Ok(())
    }

    /// Hierarchical macro-scan branch:
    /// 1. Map coarse areas.
    /// 2. Refine boundaries.
    /// 3. Run scoped domain detection + key-modules analysis per leaf in parallel.
    /// 4. Merge results into the standard research keys.
    /// 5. Synthesize area integration docs.
    async fn execute_macro_scan_branch(
        &self,
        context: &GeneratorContext,
        all_insights: &CodeAndDirectoryInsights,
    ) -> Result<()> {
        // Stage A: coarse area mapping.
        let mut area_tree = AreaMapperAgent.execute(context).await?;
        context
            .store_to_memory(
                ResearchMemoryScope::STUDIES_RESEARCH,
                "AreaTree",
                serde_json::to_value(&area_tree)?,
            )
            .await?;

        // Stage B: DFS boundary refinement.
        AreaRefinerAgent::refine_tree(context, &mut area_tree).await?;

        let leaves = area_tree.leaves();
        if leaves.is_empty() {
            return Err(anyhow!("Area tree has no leaf areas after refinement"));
        }

        println!(
            "🍃 Found {} leaf areas to analyze in parallel",
            leaves.len()
        );

        // Stage C: per-leaf scoped pipeline.
        let max_parallels = context.config.llm.max_parallels;
        let leaf_futures: Vec<_> = leaves
            .iter()
            .map(|leaf| {
                let leaf_handle = LeafHandle {
                    id: leaf.id.clone(),
                    name: leaf.name.clone(),
                    description: leaf.description.clone(),
                    root_paths: leaf.root_paths.clone(),
                    dossier_paths: leaf.dossier_paths.clone(),
                };
                let context_clone = context.clone();
                let insights_clone = all_insights.clone();
                Box::pin(async move {
                    Self::process_leaf_area(&context_clone, &leaf_handle, &insights_clone).await
                })
            })
            .collect();

        let leaf_results = do_parallel_with_limit(leaf_futures, max_parallels).await;

        let mut merged_domain_modules = Vec::new();
        let mut merged_domain_relations = Vec::new();
        let mut merged_business_flows = Vec::new();
        let mut merged_key_reports = Vec::new();
        let mut area_summaries = Vec::new();
        let mut confidence_scores = Vec::new();
        let mut successful_areas = 0;

        for (leaf, result) in leaves.iter().zip(leaf_results) {
            match result {
                Ok((area_domain_report, area_key_reports)) => {
                    merged_domain_modules.extend(area_domain_report.domain_modules);
                    merged_domain_relations.extend(area_domain_report.domain_relations);
                    merged_business_flows.extend(area_domain_report.business_flows);
                    merged_key_reports.extend(area_key_reports);

                    if !area_domain_report.architecture_summary.is_empty() {
                        area_summaries.push(format!(
                            "### {}\n{}",
                            leaf.name, area_domain_report.architecture_summary
                        ));
                    }
                    confidence_scores.push(area_domain_report.confidence_score);
                    successful_areas += 1;
                }
                Err(e) => {
                    println!(
                        "⚠️  Area '{}' analysis failed and will be skipped: {}",
                        leaf.name, e
                    );
                }
            }
        }

        if successful_areas == 0 {
            return Err(anyhow!("All leaf-area analyses failed"));
        }

        // Stage D: merge per-area reports into the standard global research keys.
        let merged_summary = area_summaries.join("\n\n");
        let avg_confidence = if confidence_scores.is_empty() {
            0.0
        } else {
            confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64
        };

        let merged_domain_report = DomainModulesReport {
            domain_modules: merged_domain_modules,
            domain_relations: merged_domain_relations,
            business_flows: merged_business_flows,
            architecture_summary: merged_summary,
            confidence_score: avg_confidence,
        };

        context
            .store_to_memory(
                ResearchMemoryScope::STUDIES_RESEARCH,
                &AgentType::DomainModulesDetector.to_string(),
                serde_json::to_value(&merged_domain_report)?,
            )
            .await?;

        context
            .store_to_memory(
                ResearchMemoryScope::STUDIES_RESEARCH,
                &AgentType::KeyModulesInsight.to_string(),
                serde_json::to_value(&merged_key_reports)?,
            )
            .await?;

        println!(
            "✅ Merged macro-scan results: {} domains, {} key-module reports",
            merged_domain_report.domain_modules.len(),
            merged_key_reports.len()
        );

        // Stage E: bottom-up area synthesis into documentation.
        let area_docs = AreaSynthesisAgent::synthesize(context, &area_tree).await?;
        let mut integration_keys = Vec::with_capacity(area_docs.len());
        for (key, markdown) in area_docs {
            integration_keys.push(key.clone());
            context
                .store_to_memory(ComposeMemoryScope::DOCUMENTATION, &key, markdown)
                .await?;
        }
        context
            .store_to_memory(
                ComposeMemoryScope::DOCUMENTATION,
                "__area_integration_keys__",
                integration_keys,
            )
            .await?;

        // Continue with the remaining C2 agents using the merged reports.
        self.execute_agent(&ArchitectureResearcher, context).await?;
        self.execute_agent(&WorkflowResearcher, context).await?;

        Ok(())
    }

    /// Run the domain detector and key-modules analyst for a single leaf area.
    async fn process_leaf_area(
        context: &GeneratorContext,
        area: &LeafHandle,
        all_insights: &CodeAndDirectoryInsights,
    ) -> Result<(DomainModulesReport, Vec<KeyModuleReport>)> {
        println!(
            "🔍 Analyzing area '{}' ({} dossiers)...",
            area.name,
            area.dossier_paths.len()
        );

        let area_insights =
            filter_insights_by_paths(all_insights, &area.root_paths, &context.config.project_path);

        // Scoped domain detection for this area.
        let mut area_domain_report =
            Self::detect_domains_for_area(context, area, &area_insights).await?;

        // Prefix domain names with the area name to ensure global uniqueness.
        let _ = Self::prefix_area_domain_names(area.name.as_str(), &mut area_domain_report);

        // Store the per-area domain report for traceability.
        context
            .store_to_memory(
                ResearchMemoryScope::STUDIES_RESEARCH,
                &format!("DomainModulesDetector_{}", area.id),
                serde_json::to_value(&area_domain_report)?,
            )
            .await?;

        // Scoped key-modules analysis for each domain in this area.
        let max_parallels = context.config.llm.max_parallels;
        let area_id = area.id.clone();
        let area_name = area.name.clone();
        let domain_futures: Vec<_> = area_domain_report
            .domain_modules
            .iter()
            .map(|domain| {
                let domain = domain.clone();
                let context_clone = context.clone();
                let area_insights_clone = area_insights.clone();
                let area_id = area_id.clone();
                let area_name = area_name.clone();
                Box::pin(async move {
                    Self::analyze_domain_in_area(
                        &context_clone,
                        &area_id,
                        &area_name,
                        &domain,
                        &area_insights_clone,
                    )
                    .await
                })
            })
            .collect();

        let domain_results = do_parallel_with_limit(domain_futures, max_parallels).await;

        let mut area_key_reports = Vec::new();
        let mut successful_domains = 0;

        for (domain, result) in area_domain_report.domain_modules.iter().zip(domain_results) {
            match result {
                Ok(report) => {
                    // Area-specific traceability key.
                    context
                        .store_to_memory(
                            ResearchMemoryScope::STUDIES_RESEARCH,
                            &format!("KeyModulesInsight_{}_{}", area.id, domain.name),
                            serde_json::to_value(&report)?,
                        )
                        .await?;

                    // Standard key expected by the documentation composer.
                    context
                        .store_to_memory(
                            ResearchMemoryScope::STUDIES_RESEARCH,
                            &format!("{}_{}", AgentType::KeyModulesInsight, domain.name),
                            serde_json::to_value(&report)?,
                        )
                        .await?;

                    area_key_reports.push(report);
                    successful_domains += 1;
                }
                Err(e) => {
                    println!(
                        "⚠️  Key-modules analysis failed for area '{}', domain '{}': {}",
                        area.name, domain.name, e
                    );
                }
            }
        }

        if area_domain_report.domain_modules.is_empty() || successful_domains > 0 {
            println!("✅ Area '{}' analysis completed", area.name);
            Ok((area_domain_report, area_key_reports))
        } else {
            Err(anyhow!(
                "All key-modules analyses failed for area '{}'",
                area.name
            ))
        }
    }

    /// Run a scoped `DomainModulesDetector` extract call for one leaf area.
    async fn detect_domains_for_area(
        context: &GeneratorContext,
        area: &LeafHandle,
        area_insights: &CodeAndDirectoryInsights,
    ) -> Result<DomainModulesReport> {
        let template = DomainModulesDetector::default().prompt_template();

        let mut user_prompt = String::new();
        user_prompt.push_str(&template.opening_instruction);
        user_prompt.push('\n');
        user_prompt.push('\n');
        user_prompt.push_str(&Self::format_area_context(area, area_insights));
        user_prompt.push('\n');
        user_prompt.push_str("### Directory Dossiers in this Area\n");
        user_prompt.push_str(&Self::format_directory_dossiers(area_insights));
        user_prompt.push('\n');
        user_prompt.push_str(&template.closing_instruction);

        let language_instruction = context.config.target_language.prompt_instruction();
        let system_prompt = format!("{}\n\n{}", template.system_prompt, language_instruction);
        let user_prompt = format!("{}\n\n{}", user_prompt, language_instruction);

        let params = AgentExecuteParams {
            prompt_sys: system_prompt,
            prompt_user: user_prompt,
            cache_scope: format!(
                "{}/DomainModulesDetector_{}",
                ResearchMemoryScope::STUDIES_RESEARCH,
                area.id
            ),
            log_tag: format!("area domain detection: {}", area.name),
            progress: None,
        };

        extract(context, params).await
    }

    /// Run a scoped `KeyModulesInsight` extract call for one domain inside an area.
    async fn analyze_domain_in_area(
        context: &GeneratorContext,
        area_id: &str,
        area_name: &str,
        domain: &DomainModule,
        area_insights: &CodeAndDirectoryInsights,
    ) -> Result<KeyModuleReport> {
        let filtered_files = Self::filter_files_for_domain(domain, area_insights);
        let (system_prompt, user_prompt) = Self::build_key_module_prompt(domain, &filtered_files);

        let language_instruction = context.config.target_language.prompt_instruction();
        let system_prompt = format!("{}\n\n{}", system_prompt, language_instruction);
        let user_prompt = format!("{}\n\n{}", user_prompt, language_instruction);

        let params = AgentExecuteParams {
            prompt_sys: system_prompt,
            prompt_user: user_prompt,
            cache_scope: format!(
                "{}/KeyModulesInsight_{}_{}",
                ResearchMemoryScope::STUDIES_RESEARCH,
                area_id,
                domain.name
            ),
            log_tag: format!("key modules: {}", domain.name),
            progress: None,
        };

        let mut report: KeyModuleReport = extract(context, params).await?;
        report.domain_name = domain.name.clone();
        if report.module_name.is_empty() {
            report.module_name = format!("{} Core Module", domain.name);
        }

        println!(
            "✅ Domain analysis completed for '{}' in area '{}'",
            domain.name, area_name
        );
        Ok(report)
    }

    /// Prefix every domain name in an area report with the area name and return
    /// the old-to-new mapping so relation/flow references can be rewritten.
    fn prefix_area_domain_names(
        area_name: &str,
        report: &mut DomainModulesReport,
    ) -> HashMap<String, String> {
        let mut name_map = HashMap::new();

        for domain in &mut report.domain_modules {
            let old_name = domain.name.clone();
            let new_name = format!("{}/{}", area_name, old_name);
            domain.name = new_name.clone();
            name_map.insert(old_name, new_name);
        }

        for relation in &mut report.domain_relations {
            if let Some(new_name) = name_map.get(&relation.from_domain) {
                relation.from_domain = new_name.clone();
            }
            if let Some(new_name) = name_map.get(&relation.to_domain) {
                relation.to_domain = new_name.clone();
            }
        }

        for flow in &mut report.business_flows {
            for step in &mut flow.steps {
                if let Some(new_name) = name_map.get(&step.domain_module) {
                    step.domain_module = new_name.clone();
                }
            }
        }

        name_map
    }

    /// Collect file insights from the area-scoped insights that belong to a domain.
    fn filter_files_for_domain(
        domain: &DomainModule,
        area_insights: &CodeAndDirectoryInsights,
    ) -> Vec<FileInsight> {
        let mut domain_paths: HashSet<String> = HashSet::new();
        for path in &domain.code_paths {
            domain_paths.insert(path.replace('\\', "/"));
        }
        for sub in &domain.sub_modules {
            for path in &sub.code_paths {
                domain_paths.insert(path.replace('\\', "/"));
            }
        }

        if domain_paths.is_empty() {
            return Vec::new();
        }

        area_insights
            .directory_insights
            .iter()
            .flat_map(|d| d.file_insights.iter())
            .filter(|fi| {
                let file_path = fi.file_path.to_string_lossy().replace('\\', "/");
                domain_paths
                    .iter()
                    .any(|path| file_path.contains(path) || path.contains(&file_path))
            })
            .take(50)
            .cloned()
            .collect()
    }

    /// Build the same per-domain prompt that `KeyModulesInsight` uses.
    fn build_key_module_prompt(
        domain: &DomainModule,
        insights: &[FileInsight],
    ) -> (String, String) {
        let system_prompt =
            "Based on the provided domain and code insights, conduct in-depth analysis and return strict JSON only.

Output requirements (no markdown, no code fences, no prose outside JSON):
{
  \"domain_name\": \"string\",
  \"module_name\": \"string\",
  \"module_description\": \"string\",
  \"interaction\": \"string\",
  \"implementation\": \"string\",
  \"associated_files\": [\"string\"],
  \"flowchart_mermaid\": \"string\",
  \"sequence_diagram_mermaid\": \"string\"
}

Rules:
- Include all fields every time.
- Use plain strings for all textual fields.
- associated_files must be an array of strings.
- If uncertain, use empty strings/empty array.
- Mermaid fields must be valid mermaid text or empty string.
"
            .to_string();

        let user_prompt = format!(
            "## Domain Analysis Task\nAnalyze the core module technical details of the '{}' domain\n\n### Domain Information\n- Domain Name: {}\n- Domain Type: {}\n- Importance: {:.1}/10\n- Complexity: {:.1}/10\n- Description: {}\n\n### Submodule Overview\n{}\n\n### Related Code Insights\n{}\n",
            domain.name,
            domain.name,
            domain.domain_type,
            domain.importance,
            domain.complexity,
            domain.description,
            Self::format_sub_modules(&domain.sub_modules),
            Self::format_filtered_insights(insights)
        );

        (system_prompt, user_prompt)
    }

    fn format_sub_modules(sub_modules: &[SubModule]) -> String {
        if sub_modules.is_empty() {
            return "No submodule information available".to_string();
        }

        sub_modules
            .iter()
            .enumerate()
            .map(|(i, sub)| {
                format!(
                    "{}. **{}**\n   - Description: {}\n   - Importance: {:.1}/10\n   - Core Functions: {}\n   - Code Files: {}",
                    i + 1,
                    sub.name,
                    sub.description,
                    sub.importance,
                    sub.key_functions.join(", "),
                    sub.code_paths.join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn format_filtered_insights(insights: &[FileInsight]) -> String {
        if insights.is_empty() {
            return "No related code insights available".to_string();
        }

        insights
            .iter()
            .enumerate()
            .map(|(i, fi)| {
                format!(
                    "{}. File `{}`, Purpose: {:?}\n   Description: {}\n   Source Code\n```code\n{}```\n---\n",
                    i + 1,
                    fi.file_path.to_string_lossy(),
                    fi.code_purpose,
                    fi.summary,
                    fi.source_summary
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_area_context(area: &LeafHandle, insights: &CodeAndDirectoryInsights) -> String {
        let root_paths = area
            .root_paths
            .iter()
            .map(|p| format!("`{}`", p.display()))
            .collect::<Vec<_>>()
            .join(", ");
        let file_count: usize = insights
            .directory_insights
            .iter()
            .map(|d| d.file_insights.len())
            .sum();

        format!(
            "### Focus Area: {}\n{}\n\n- Root paths: {}\n- Dossiers in scope: {}\n- Files in scope: {}\n",
            area.name,
            area.description,
            root_paths,
            insights.directory_insights.len(),
            file_count
        )
    }

    fn format_directory_dossiers(insights: &CodeAndDirectoryInsights) -> String {
        if insights.directory_insights.is_empty() {
            return "No directory dossiers available for this area.".to_string();
        }

        let mut content = String::new();
        for dossier in &insights.directory_insights {
            content.push_str(&format!(
                "- `{}` (purpose: {:?}, importance: {:.1}): {}\n",
                dossier.path.display(),
                dossier.purpose,
                dossier.importance_score,
                dossier.summary
            ));
            if !dossier.key_files.is_empty() {
                content.push_str(&format!("  Key files: {}\n", dossier.key_files.join(", ")));
            }
        }
        content
    }

    /// Check if the project has database-related files
    async fn has_database_files(&self, context: &GeneratorContext) -> bool {
        if let Some(insights) = context
            .get_from_memory::<CodeAndDirectoryInsights>(
                PreprocessMemoryScope::PREPROCESS,
                ScopedKeys::CODE_INSIGHTS,
            )
            .await
        {
            insights.directory_insights.iter().any(|dossier| {
                dossier.purpose == DirectoryPurpose::Database
                    || dossier.name.to_lowercase().contains("database")
                    || dossier.name.to_lowercase().contains("db")
            }) || insights
                .directory_insights
                .iter()
                .flat_map(|d| d.file_insights.iter())
                .any(|fi| {
                    fi.code_purpose == CodePurpose::Database
                        || fi.file_path.to_string_lossy().ends_with(".sql")
                        || fi.file_path.to_string_lossy().ends_with(".sqlproj")
                })
        } else {
            false
        }
    }

    /// Execute a single agent
    async fn execute_agent<T>(&self, agent: &T, context: &GeneratorContext) -> Result<()>
    where
        T: StepForwardAgent + Send + Sync,
    {
        let agent_name = if let Some(agent_enum) = agent.agent_type_enum() {
            agent_enum.display_name(&context.config.target_language)
        } else {
            agent.agent_type()
        };

        println!("🤖 Executing {} agent analysis...", agent_name);

        agent.execute(context).await?;
        println!("✓ {} analysis completed", agent_name);
        Ok(())
    }
}
