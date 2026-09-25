use std::sync::Arc;
use std::time::Instant;

use crate::generator::compose::DocumentationComposer;
use crate::generator::outlet::{AgentContentOutlet, DiskOutlet, DocTree, Outlet, SummaryOutlet};
use crate::{
    cache::CacheManager,
    config::Config,
    generator::{
        context::GeneratorContext, preprocess::PreProcessAgent,
        research::orchestrator::ResearchOrchestrator, types::Generator,
    },
    llm::client::LLMClient,
    memory::Memory,
};
use anyhow::Result;
use tokio::sync::RwLock;

/// Memory scope and key definitions for workflow timing statistics
pub struct TimingScope;

impl TimingScope {
    /// Memory scope for timing statistics
    pub const TIMING: &'static str = "timing";
}

/// Memory key definitions for each workflow stage
pub struct TimingKeys;

impl TimingKeys {
    /// Preprocessing stage duration
    pub const PREPROCESS: &'static str = "preprocess";
    /// Research stage duration
    pub const RESEARCH: &'static str = "research";
    /// Document generation stage duration
    pub const COMPOSE: &'static str = "compose";
    /// Output stage duration
    pub const OUTPUT: &'static str = "output";
    /// Document generation time
    pub const DOCUMENT_GENERATION: &'static str = "document_generation";
    /// Total execution time
    pub const TOTAL_EXECUTION: &'static str = "total_execution";
}

pub async fn launch(c: &Config) -> Result<()> {
    let overall_start = Instant::now();

    let config = c.clone();

    // --force-regenerate: clear the LLM response cache before running
    if config.force_regenerate {
        let cache_dir = &config.cache.cache_dir;
        // Safety guard: refuse to wipe the filesystem root or a bare
        // single-component path (e.g. `--cache-dir /` or `.`).
        let is_safe_cache_dir =
            !cache_dir.as_os_str().is_empty() && cache_dir.components().count() >= 2;
        if !is_safe_cache_dir {
            anyhow::bail!(
                "refusing to clear suspicious cache dir '{}' via --force-regenerate",
                cache_dir.display()
            );
        }
        if cache_dir.exists() {
            println!(
                "=== Force regeneration: clearing LLM cache at {} ===",
                cache_dir.display()
            );
            std::fs::remove_dir_all(cache_dir)?;
        }
    }

    // Check mermaid-fixer availability at startup
    if !crate::generator::outlet::MermaidFixer::is_available().await {
        anyhow::bail!("mermaid-fixer is not installed. Run 'cargo install mermaid-fixer' to install it");
    }

    let llm_client = LLMClient::new(config.clone())?;
    let cache_manager = Arc::new(RwLock::new(CacheManager::new(
        config.cache.clone(),
        config.target_language.clone(),
    )));
    let memory = Arc::new(RwLock::new(Memory::new()));

    let context = GeneratorContext {
        llm_client,
        config,
        cache_manager,
        memory,
    };

    if context.config.cost_and_usage {
        let execution_id = format!(
            "{}_{}",
            chrono::Utc::now().format("%Y%m%d_%H%M%S"),
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        );
        crate::llm::client::usage_tracker::UsageTracker::global().begin_execution(
            execution_id,
            context.config.get_project_name(),
            context.config.llm.api_base_url.clone(),
            context.config.llm.provider.to_string(),
        );
        println!("💰 Cost & usage tracking enabled");
        if context.config.llm.provider != crate::config::LLMProvider::OpenAI {
            eprintln!(
                "⚠️  Warning: cost/usage capture is wired to the OpenAI-compatible (LiteLLM) transport; provider '{}' calls will not be captured",
                context.config.llm.provider
            );
        }
    }

    let pipeline_result: anyhow::Result<()> = async {
        // Sync external knowledge if configured
        if let Ok(syncer) = crate::integrations::KnowledgeSyncer::new(context.config.clone()) {
            if syncer.should_sync().unwrap_or(false) {
                println!("\n=== Syncing external knowledge sources ===");
                if let Err(e) = syncer.sync_all().await {
                    eprintln!("⚠️  Warning: Failed to sync external knowledge: {}", e);
                }
            } else {
                let lang = context.config.target_language.display_name();
                println!("ℹ️  External knowledge cache ({}) is up to date", lang);
            }
        }

        // Preprocessing stage
        if context.config.skip_preprocessing {
            println!("=== Skipping preprocessing (--skip-preprocessing) ===");
            println!("   ⚠️  Downstream stages read preprocessed insights from memory; if this is not a warm/partial run they may fail.");
        } else {
            let preprocess_start = Instant::now();
            let preprocess_agent = PreProcessAgent::new();
            preprocess_agent.execute(context.clone()).await?;
            let preprocess_time = preprocess_start.elapsed().as_secs_f64();
            context
                .store_to_memory(TimingScope::TIMING, TimingKeys::PREPROCESS, preprocess_time)
                .await?;
            println!(
                "=== Preprocessing completed, results stored to Memory (Duration: {:.2}s) ===",
                preprocess_time
            );
        }

        // Execute multi-agent research stage
        if context.config.skip_research {
            println!("=== Skipping research stage (--skip-research) ===");
        } else {
            let research_start = Instant::now();
            let research_orchestrator = ResearchOrchestrator::default();
            research_orchestrator
                .execute_research_pipeline(&context)
                .await?;
            let research_time = research_start.elapsed().as_secs_f64();
            context
                .store_to_memory(TimingScope::TIMING, TimingKeys::RESEARCH, research_time)
                .await?;
            println!("\n=== Project in-depth research completed (Duration: {:.2}s) ===", research_time);
        }

        // Execute document generation process
        if context.config.skip_documentation {
            println!("=== Skipping document generation (--skip-documentation) ===");
        } else {
            let compose_start = Instant::now();
            let mut doc_tree = DocTree::new(&context.config.target_language);
            let documentation_orchestrator = DocumentationComposer::default();
            documentation_orchestrator
                .execute(&context, &mut doc_tree)
                .await?;
            let compose_time = compose_start.elapsed().as_secs_f64();
            context
                .store_to_memory(TimingScope::TIMING, TimingKeys::COMPOSE, compose_time)
                .await?;
            println!("\n=== Document generation completed (Duration: {:.2}s) ===", compose_time);

            // Execute document storage
            let output_start = Instant::now();
            let outlet = DiskOutlet::new(doc_tree);
            outlet.save(&context).await?;

            // Generate and save summary report
            let summary_outlet = SummaryOutlet::new();
            summary_outlet.save(&context).await?;

            // Generate the agent-focused content set (written last so it
            // survives DiskOutlet's output-directory wipe at the start of save).
            if context.config.agent_content {
                let agent_outlet = AgentContentOutlet::from_context(&context);
                agent_outlet.save(&context).await?;
            }

            let output_time = output_start.elapsed().as_secs_f64();
            context
                .store_to_memory(TimingScope::TIMING, TimingKeys::OUTPUT, output_time)
                .await?;
            println!("\n=== Document storage completed (Duration: {:.2}s) ===", output_time);
        }

        Ok(())
    }
    .await;

    // Record total execution time
    let total_time = overall_start.elapsed().as_secs_f64();
    context
        .store_to_memory(TimingScope::TIMING, TimingKeys::TOTAL_EXECUTION, total_time)
        .await?;

    emit_cost_usage_report(&context).await;

    pipeline_result?;

    println!("\n🎉 All processes execution completed! Total duration: {:.2}s", total_time);

    Ok(())
}

/// Persist and render the cost & usage artifacts when tracking is enabled.
/// Called on both the success and failure paths so partial-run spend is
/// never discarded.
async fn emit_cost_usage_report(context: &GeneratorContext) {
    if !context.config.cost_and_usage {
        return;
    }
    let Some(report) =
        crate::llm::client::usage_tracker::UsageTracker::global().build_report()
    else {
        return;
    };
    if let Err(e) =
        crate::llm::client::usage_tracker::persist(&context.config.cost_usage_dir, &report)
    {
        eprintln!("\u{26a0}\u{fe0f}  Warning: failed to persist cost/usage records: {}", e);
    }
    let markdown = crate::llm::client::usage_tracker::render_markdown_report(&report);
    let out_path = context
        .config
        .output_path
        .join("__Litho_Cost_Usage_Report__.md");
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&out_path, markdown) {
        Ok(()) => println!(
            "\u{1f4b0} Cost & usage report: {} (total ${:.4}, {} calls)",
            out_path.display(),
            report.total_cost_usd,
            report.calls.len()
        ),
        Err(e) => eprintln!("\u{26a0}\u{fe0f}  Warning: failed to write cost/usage report: {}", e),
    }
}
