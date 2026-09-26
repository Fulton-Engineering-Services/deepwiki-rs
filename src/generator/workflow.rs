use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::generator::compose::DocumentationComposer;
use crate::generator::outlet::{AgentContentOutlet, DiskOutlet, DocTree, Outlet, SummaryOutlet};
use crate::generator::preprocess::memory::{MemoryScope as PreprocessScope, ScopedKeys};
use crate::generator::research::memory::MemoryScope as ResearchMemoryScope;
use crate::generator::research::types::AgentType as ResearchAgentType;
use crate::{
    cache::CacheManager,
    config::Config,
    generator::{
        context::GeneratorContext, preprocess::PreProcessAgent,
        research::orchestrator::ResearchOrchestrator, types::Generator,
    },
    llm::client::LLMClient,
    memory::{MEMORY_SNAPSHOT_FILE, MEMORY_SNAPSHOT_VERSION, Memory, MemorySnapshot},
};
use anyhow::{Context, Result, bail};
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

    // --only-agent-content: hydrate the persisted memory snapshot and rebuild
    // just the agent content set. Branches before the cache-clear so a
    // combined --force-regenerate can never wipe the warm LLM cache by
    // accident (it is rejected inside agent_content_only instead).
    if config.only_agent_content {
        return agent_content_only(config, overall_start).await;
    }

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

    // Tracks whether the research pipeline ran to completion, so a run that
    // dies mid-research never overwrites a previously good snapshot with a
    // partial one (set inside the async block below).
    let mut research_completed = false;

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
            research_completed = true;
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

    // Checkpoint research/preprocess artifacts for a later
    // --only-agent-content run. Best-effort and runs on failure paths too, so
    // a research stage that completed before a compose error is not thrown
    // away — while a research stage that itself failed leaves any existing
    // snapshot untouched (gated on research_completed below).
    persist_memory_snapshot(&context, research_completed).await;

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

/// `--only-agent-content`: hydrate `.litho/memory.json` and rebuild just the
/// agent content set. Runs no LLM inference — every artifact comes from the
/// snapshot written by a prior full run, and `AgentContentOutlet` renders
/// deterministically from memory (it resets `<output>/.agent-content/` itself,
/// so human docs are left untouched).
async fn agent_content_only(config: Config, overall_start: Instant) -> Result<()> {
    if config.force_regenerate {
        bail!(
            "--only-agent-content cannot be combined with --force-regenerate: agent content is \
             always rebuilt from the snapshot, and clearing the LLM cache would only force new \
             inference on the next full run"
        );
    }
    if !config.agent_content {
        bail!("--only-agent-content cannot be combined with --no-agent-content");
    }

    let snapshot_path = snapshot_path(&config);
    println!(
        "=== Agent-content-only mode: hydrating memory snapshot {} ===",
        snapshot_path.display()
    );
    let memory = load_memory_snapshot(&config).await?;

    // The outlet only shells out to mermaid-fixer for the diagram pass.
    if config.agent_content_diagrams
        && !crate::generator::outlet::MermaidFixer::is_available().await
    {
        bail!("mermaid-fixer is not installed. Run 'cargo install mermaid-fixer' to install it");
    }

    let llm_client = LLMClient::new(config.clone())?;
    let cache_manager = Arc::new(RwLock::new(CacheManager::new(
        config.cache.clone(),
        config.target_language.clone(),
    )));
    let context = GeneratorContext {
        llm_client,
        config,
        cache_manager,
        memory: Arc::new(RwLock::new(memory)),
    };

    validate_agent_memory(&context).await?;

    let agent_outlet = AgentContentOutlet::from_context(&context);
    agent_outlet.save(&context).await?;

    println!(
        "\n\u{1f389} Agent content regenerated from snapshot in {:.2}s (no LLM inference)",
        overall_start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Path of the persisted memory snapshot for a config.
fn snapshot_path(config: &Config) -> PathBuf {
    config.internal_path.join(MEMORY_SNAPSHOT_FILE)
}

/// Load and validate the memory snapshot, refusing stale or foreign ones:
/// the stored fingerprint must match the current project path, language,
/// models, analysis knobs, and git state.
async fn load_memory_snapshot(config: &Config) -> Result<Memory> {
    let path = snapshot_path(config);
    if !path.exists() {
        bail!(
            "no memory snapshot at {}; run a full pass first so --only-agent-content has \
             artifacts to hydrate",
            path.display()
        );
    }
    let bytes = tokio::fs::read(&path)
        .await
        .with_context(|| format!("reading memory snapshot {}", path.display()))?;
    let snapshot: MemorySnapshot = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing memory snapshot {}", path.display()))?;
    if snapshot.version != MEMORY_SNAPSHOT_VERSION {
        bail!(
            "memory snapshot {} has format version {} but this build expects {}; run a full \
             pass to refresh it",
            path.display(),
            snapshot.version,
            MEMORY_SNAPSHOT_VERSION
        );
    }
    let expected = memory_fingerprint(config);
    if snapshot.fingerprint != expected {
        bail!(
            "memory snapshot {} is stale — the project, language, models, or code state \
             changed since it was written.\n  snapshot: {}\n  current:  {}\nRun a full pass to \
             refresh it, then retry --only-agent-content.",
            path.display(),
            snapshot.fingerprint,
            expected
        );
    }
    let memory = Memory::from_snapshot(snapshot);
    if memory.is_empty() {
        bail!(
            "memory snapshot {} is empty; run a full pass to refresh it",
            path.display()
        );
    }
    Ok(memory)
}

/// Verify the hydrated snapshot carries the artifacts the agent outlet needs:
/// hard-fails when the research scope is entirely absent (nothing to render),
/// and warns per missing core report so a degraded page set is never emitted
/// silently.
async fn validate_agent_memory(context: &GeneratorContext) -> Result<()> {
    let research_keys = context
        .list_memory_keys(ResearchMemoryScope::STUDIES_RESEARCH)
        .await;
    if research_keys.is_empty() {
        bail!(
            "memory snapshot has no research artifacts (scope '{}' is empty); run a full pass \
             to refresh it",
            ResearchMemoryScope::STUDIES_RESEARCH
        );
    }
    // Reports the pipeline always produces; anything conditional (database,
    // per-area modules under macro-scan) is not warned about. `collect_bundle`
    // silently skips whatever is missing, so surface the core gaps here.
    for (label, agent_type) in [
        (
            "system context report",
            ResearchAgentType::SystemContextResearcher,
        ),
        (
            "boundary interface report",
            ResearchAgentType::BoundaryAnalyzer,
        ),
    ] {
        if !context
            .has_memory_data(
                ResearchMemoryScope::STUDIES_RESEARCH,
                &agent_type.to_string(),
            )
            .await
        {
            eprintln!(
                "\u{26a0}\u{fe0f}  Warning: snapshot has no {}; matching agent pages will be \
                 missing",
                label
            );
        }
    }
    if !context
        .has_memory_data(ResearchMemoryScope::STUDIES_RESEARCH, "AreaTree")
        .await
    {
        eprintln!(
            "\u{26a0}\u{fe0f}  Warning: snapshot has no refined area tree; the agent tree \
             falls back to a flat root layout"
        );
    }
    if !context
        .has_memory_data(PreprocessScope::PREPROCESS, ScopedKeys::CODE_INSIGHTS)
        .await
    {
        eprintln!(
            "\u{26a0}\u{fe0f}  Warning: snapshot has no preprocessing file insights; per-file \
             agent pages will be skipped"
        );
    }
    Ok(())
}

/// Hash of every input the persisted research artifacts were derived from.
/// Stored inside the snapshot and re-checked on hydration so a stale
/// snapshot fails loudly instead of rendering outdated pages.
fn memory_fingerprint(config: &Config) -> String {
    use md5::{Digest, Md5};

    let project_dir =
        std::fs::canonicalize(&config.project_path).unwrap_or_else(|_| config.project_path.clone());

    let fields = [
        project_dir.to_string_lossy().into_owned(),
        config.target_language.display_name().to_string(),
        config.llm.provider.to_string(),
        config.llm.model_efficient.clone(),
        config.llm.model_powerful.clone(),
        config.get_project_name(),
        config.boundary_analysis.code_insights_limit.to_string(),
        config.boundary_analysis.include_source_code.to_string(),
        config
            .boundary_analysis
            .only_directories_when_files_more_than
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string()),
        format!("{:?}", config.macro_scan.enabled),
        git_state_marker(config, &project_dir),
    ];

    let mut hasher = Md5::new();
    for field in &fields {
        hasher.update(field.as_bytes());
        hasher.update(b"\x1f");
    }
    format!("{:x}", hasher.finalize())
}

/// Best-effort code-state marker: git HEAD plus a hash of the working tree
/// (`git status --porcelain` + `git diff HEAD`), so both edits to tracked
/// files and added/removed files invalidate the snapshot. Directories the
/// run itself writes (`.litho/`, the output dir) are excluded from the status
/// scan so the snapshot's own artifacts cannot invalidate it on arrival.
/// Non-git projects fall back to the project root's mtime.
///
/// Known limits: edits to the *content* of an already-untracked file after
/// persist are invisible (its path presence is captured, its bytes are not),
/// and in a tree that stays dirty the marker hashes the diff — so staleness
/// detection is strong but not absolute. Any commit, even a message-only
/// amend, changes HEAD and forces a full refresh by design.
fn git_state_marker(config: &Config, project_dir: &Path) -> String {
    use md5::{Digest, Md5};

    let status_args = status_pathspecs(config, project_dir);
    let status_args: Vec<&str> = status_args.iter().map(String::as_str).collect();

    let git = |args: &[&str]| -> Option<Vec<u8>> {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(project_dir)
            .args(args)
            .output()
            .ok()?;
        if out.status.success() {
            Some(out.stdout)
        } else {
            None
        }
    };

    let Some(head_bytes) = git(&["rev-parse", "HEAD"]) else {
        let mtime = std::fs::metadata(project_dir)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        return format!("nogit:{}", mtime);
    };
    let head = String::from_utf8_lossy(&head_bytes).trim().to_string();
    let Some(status) = git(&status_args) else {
        return format!("{}:unknown", head);
    };
    let diff = git(&["diff", "HEAD"]).unwrap_or_default();

    let mut hasher = Md5::new();
    hasher.update(&status);
    hasher.update(b"\x00");
    hasher.update(&diff);
    format!("{}:{:x}", head, hasher.finalize())
}

/// `git status` arguments for the code-state marker: a porcelain scan of the
/// project with the run's own output dirs (`.litho/`, the docs dir) excluded,
/// so writing the snapshot cannot invalidate its own fingerprint — even when
/// those dirs are not gitignored.
fn status_pathspecs(config: &Config, project_dir: &Path) -> Vec<String> {
    let mut args = vec![
        "status".to_string(),
        "--porcelain".to_string(),
        "--".to_string(),
        ".".to_string(),
    ];
    for written in [&config.internal_path, &config.output_path] {
        if let Some(rel) = relative_to_project(project_dir, written) {
            args.push(format!(":(exclude){}", rel));
        }
    }
    args
}

/// Resolve `p` relative to the project root for use as a git pathspec, with a
/// lexical fallback for paths that do not exist yet (e.g. a fresh output dir).
fn relative_to_project(project_dir: &Path, p: &Path) -> Option<String> {
    let base = std::fs::canonicalize(project_dir).ok()?;
    let candidate = if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    };
    let rel = candidate.strip_prefix(&base).ok()?;
    if rel.as_os_str().is_empty() {
        None
    } else {
        Some(rel.to_string_lossy().replace('\\', "/"))
    }
}

/// Write the in-memory research/preprocess artifacts to `.litho/memory.json`
/// so a later `--only-agent-content` run can hydrate them. Best-effort
/// (warnings only), only called after the research pipeline reported success —
/// a run that died mid-research never replaces a previously good snapshot —
/// and never writes when the research scope came back empty.
async fn persist_memory_snapshot(context: &GeneratorContext, research_completed: bool) {
    if !research_completed {
        return;
    }
    let research_keys = context
        .list_memory_keys(ResearchMemoryScope::STUDIES_RESEARCH)
        .await;
    if research_keys.is_empty() {
        return;
    }

    let path = snapshot_path(&context.config);
    if let Some(parent) = path.parent()
        && let Err(e) = tokio::fs::create_dir_all(parent).await
    {
        eprintln!(
            "\u{26a0}\u{fe0f}  Warning: cannot create {}: {}",
            parent.display(),
            e
        );
        return;
    }

    let fingerprint = memory_fingerprint(&context.config);
    let snapshot = context.memory.read().await.to_snapshot(&fingerprint);
    let bytes = match serde_json::to_vec(&snapshot) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "\u{26a0}\u{fe0f}  Warning: failed to serialize memory snapshot: {}",
                e
            );
            return;
        }
    };
    match tokio::fs::write(&path, &bytes).await {
        Ok(()) => println!(
            "\u{1f4be} Memory snapshot \u{2192} {} (hydrate with --only-agent-content)",
            path.display()
        ),
        Err(e) => eprintln!(
            "\u{26a0}\u{fe0f}  Warning: failed to write memory snapshot {}: {}",
            path.display(),
            e
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_stable_for_identical_config() {
        let a = memory_fingerprint(&Config::default());
        let b = memory_fingerprint(&Config::default());
        assert_eq!(a, b);
        assert_eq!(a.len(), 32); // md5 hex
    }

    #[test]
    fn fingerprint_changes_with_analysis_inputs() {
        let base = memory_fingerprint(&Config::default());

        let cfg = Config {
            target_language: crate::i18n::TargetLanguage::Chinese,
            ..Default::default()
        };
        assert_ne!(memory_fingerprint(&cfg), base);

        let cfg = Config {
            llm: crate::config::LLMConfig {
                model_efficient: "some-other-model".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_ne!(memory_fingerprint(&cfg), base);

        let cfg = Config {
            boundary_analysis: crate::config::BoundaryAnalysisConfig {
                code_insights_limit: 999,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_ne!(memory_fingerprint(&cfg), base);
    }

    #[test]
    fn snapshot_path_lives_under_internal_path() {
        let cfg = Config {
            internal_path: PathBuf::from("/proj/.litho"),
            ..Default::default()
        };
        assert_eq!(
            snapshot_path(&cfg),
            PathBuf::from("/proj/.litho/memory.json")
        );
    }

    #[tokio::test]
    async fn snapshot_roundtrip_write_then_load() {
        let internal =
            std::env::temp_dir().join(format!("litho-wf-roundtrip-{}", std::process::id()));
        let cfg = Config {
            internal_path: internal.clone(),
            ..Default::default()
        };

        let mut memory = Memory::new();
        memory
            .store(
                ResearchMemoryScope::STUDIES_RESEARCH,
                "AreaTree",
                serde_json::json!({"root": {"id": "r"}}),
            )
            .unwrap();
        let snapshot = memory.to_snapshot(&memory_fingerprint(&cfg));

        let path = snapshot_path(&cfg);
        std::fs::create_dir_all(&internal).unwrap();
        std::fs::write(&path, serde_json::to_vec(&snapshot).unwrap()).unwrap();

        let loaded = load_memory_snapshot(&cfg)
            .await
            .expect("fresh snapshot loads");
        assert!(loaded.has_data(ResearchMemoryScope::STUDIES_RESEARCH, "AreaTree"));
        assert!(!loaded.is_empty());

        // A snapshot recorded under different models must be refused as stale.
        let stale_fp = memory_fingerprint(&Config {
            llm: crate::config::LLMConfig {
                model_efficient: "other-model".to_string(),
                ..Default::default()
            },
            ..Default::default()
        });
        let stale = Memory::new().to_snapshot(&stale_fp);
        std::fs::write(&path, serde_json::to_vec(&stale).unwrap()).unwrap();
        let err = load_memory_snapshot(&cfg).await.unwrap_err().to_string();
        assert!(err.contains("stale"), "unexpected error: {}", err);

        let _ = std::fs::remove_dir_all(&internal);
    }

    #[tokio::test]
    async fn load_fails_cleanly_when_snapshot_absent() {
        let cfg = Config {
            internal_path: std::env::temp_dir()
                .join(format!("litho-wf-absent-{}", std::process::id())),
            ..Default::default()
        };
        let err = load_memory_snapshot(&cfg).await.unwrap_err().to_string();
        assert!(
            err.contains("no memory snapshot"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn status_pathspecs_exclude_run_output_dirs() {
        let specs = status_pathspecs(&Config::default(), Path::new("."));
        assert!(
            specs.iter().any(|s| s == ":(exclude).litho"),
            "missing .litho exclusion: {:?}",
            specs
        );
        assert!(
            specs.iter().any(|s| s == ":(exclude)litho.docs"),
            "missing output-dir exclusion: {:?}",
            specs
        );
    }

    #[test]
    fn relative_to_project_resolves_inside_and_rejects_outside() {
        assert_eq!(
            relative_to_project(Path::new("."), Path::new("./.litho")).as_deref(),
            Some(".litho")
        );
        // Project root itself has no meaningful relative form.
        assert_eq!(relative_to_project(Path::new("."), Path::new(".")), None);
        // Paths outside the project are never excluded (they cannot appear in
        // its status anyway).
        assert_eq!(
            relative_to_project(Path::new("."), Path::new("/tmp/litho-elsewhere")),
            None
        );
    }
}
