use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use crate::generator::preprocess::extractors::original_document_extractor;
use crate::generator::preprocess::memory::{MemoryScope, ScopedKeys};
use crate::types::original_document::OriginalDocument;
use crate::{
    generator::{
        context::GeneratorContext,
        preprocess::extractors::structure_extractor::StructureExtractor,
        types::Generator,
    },
    types::{
        project_structure::ProjectStructure, CodeAndDirectoryInsights, DirectoryDossier,
        DirectoryPurpose,
    },
};

pub mod agents;
pub mod extractors;
pub mod memory;

use crate::generator::preprocess::agents::directory_summary::FileContent;
use crate::generator::preprocess::agents::relationships_analyze::RelationshipsAnalyze;

/// Preprocessing result — simplified to directory-only insights
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PreprocessingResult {
    pub original_document: OriginalDocument,
    pub project_structure: ProjectStructure,
    pub directory_dossiers: Vec<DirectoryDossier>,
    pub processing_time: f64,
}

pub struct PreProcessAgent {}

impl PreProcessAgent {
    pub fn new() -> Self {
        Self {}
    }
}

impl Generator<PreprocessingResult> for PreProcessAgent {
    async fn execute(&self, context: GeneratorContext) -> Result<PreprocessingResult> {
        let start_time = Instant::now();

        let structure_extractor = StructureExtractor::new(context.clone());
        let config = &context.config;

        println!("🔍 Starting project preprocessing phase...");

        // 1. Extract project original document materials
        println!("📁 Extracting project original document materials...");
        let original_document = original_document_extractor::extract(&context).await?;

        // 2. Extract project structure (includes all files and directories)
        println!("📁 Extracting project structure...");
        let project_structure = structure_extractor
            .extract_structure(&config.project_path)
            .await?;

        println!(
            "   🔭 Discovered {} files, {} directories",
            project_structure.total_files, project_structure.total_directories
        );

        // 3. Generate directory dossiers with LLM (reads files directly, no top-N filtering)
        println!("📂 Generating directory dossiers with LLM...");
        let directory_dossiers =
            generate_directory_dossiers(&context, &project_structure).await?;

        // 4. Generate relationship analysis based on directory dossiers
        println!("🔗 Generating relationship analysis...");
        let relationships_analyzer = RelationshipsAnalyze::new();
        let relationships = relationships_analyzer
            .execute(&context, &directory_dossiers)
            .await?;

        let processing_time = start_time.elapsed().as_secs_f64();

        println!(
            "✅ Project preprocessing completed, {} directories analyzed, took {:.2}s",
            directory_dossiers.len(),
            processing_time
        );

        // 4. Store results to Memory
        context
            .store_to_memory(
                MemoryScope::PREPROCESS,
                ScopedKeys::PROJECT_STRUCTURE,
                &project_structure,
            )
            .await?;
        context
            .store_to_memory(
                MemoryScope::PREPROCESS,
                ScopedKeys::CODE_INSIGHTS,
                &CodeAndDirectoryInsights {
                    file_insights: Vec::new(),
                    directory_insights: directory_dossiers.clone(),
                },
            )
            .await?;
        context
            .store_to_memory(
                MemoryScope::PREPROCESS,
                ScopedKeys::ORIGINAL_DOCUMENT,
                &original_document,
            )
            .await?;
        context
            .store_to_memory(
                MemoryScope::PREPROCESS,
                ScopedKeys::RELATIONSHIPS,
                &relationships,
            )
            .await?;

        Ok(PreprocessingResult {
            original_document,
            project_structure,
            directory_dossiers,
            processing_time,
        })
    }
}

/// Generate directory dossiers by reading files directly from disk.
/// Each directory's files are batched: if total content exceeds 256KB, split into
/// batches (sorted lexicographically for cache-friendly ordering) and merge results.
///
/// Disk reads and batch preparation run sequentially (cheap I/O), then the LLM
/// summarization calls run concurrently, bounded by `llm.max_parallels` via
/// `do_parallel_with_limit` - the same mechanism used by the key-modules
/// insight analyses. Result order matches directory order (join_all preserves
/// submission order), so downstream consumers are unaffected.
const MAX_BATCH_SIZE: usize = 256 * 1024;

async fn generate_directory_dossiers(
    context: &GeneratorContext,
    project_structure: &ProjectStructure,
) -> Result<Vec<DirectoryDossier>> {
    use crate::generator::preprocess::agents::directory_summary::DirectorySummarizer;
    use crate::utils::threads::do_parallel_with_limit;

    let config = &context.config;
    let total_dirs = project_structure.directories.len();
    let max_parallels = config.llm.max_parallels;

    // Work item: a directory plus its pre-split file content batches.
    struct WorkItem {
        idx: usize,
        dir: crate::types::DirectoryInfo,
        batches: Vec<Vec<FileContent>>,
    }

    // Phase 1 (sequential, disk I/O only): read each directory's files and
    // pre-split into batches. Directories with no readable files are skipped.
    let mut work_items = Vec::new();
    for (idx, dir) in project_structure.directories.iter().enumerate() {
        // Read all files in this directory from disk
        let mut files = read_directory_files(&dir.path, config)?;

        if files.is_empty() {
            continue;
        }

        // Sort lexicographically for cache-friendly batching
        files.sort_by(|a, b| a.name.cmp(&b.name));

        // Calculate total size and split into batches (single batch when small
        // enough, including the edge case of one file exceeding MAX_BATCH_SIZE)
        let total_size: usize = files.iter().map(|f| f.content.len()).sum();
        let batches = if total_size <= MAX_BATCH_SIZE {
            vec![files]
        } else {
            split_into_batches(&files, MAX_BATCH_SIZE)
        };

        work_items.push(WorkItem {
            idx,
            dir: dir.clone(),
            batches,
        });
    }

    // Phase 2 (parallel, bounded by llm.max_parallels): summarize directories.
    println!(
        "🚀 Generating directory dossiers concurrently, max parallelism: {}",
        max_parallels
    );
    let summarizer_futures: Vec<_> = work_items
        .into_iter()
        .map(|item| {
            let context_clone = context.clone();
            Box::pin(async move {
                let summarizer = DirectorySummarizer::new();
                let progress = Some((item.idx + 1, total_dirs));
                let result = if item.batches.len() == 1 {
                    let files = item.batches.into_iter().next().unwrap_or_default();
                    summarizer
                        .summarize_directory(&context_clone, &item.dir, &files, progress)
                        .await
                } else {
                    summarizer
                        .summarize_batch(&context_clone, &item.dir, &item.batches, progress)
                        .await
                };
                (item.dir, result)
            })
        })
        .collect();

    let results = do_parallel_with_limit(summarizer_futures, max_parallels).await;

    // Phase 3 (sequential): collect in directory order, falling back
    // per-directory on error.
    let mut dossiers = Vec::new();
    for (dir, result) in results {
        match result {
            Ok(dossier) => dossiers.push(dossier),
            Err(e) => {
                eprintln!(
                    "⚠️  Failed to summarize directory {}: {}, using fallback",
                    dir.name, e
                );
                dossiers.push(fallback_dossier(&dir));
            }
        }
    }

    Ok(dossiers)
}

/// Read all files in a directory, respecting config exclusions and max_file_size.
fn read_directory_files(
    dir_path: &std::path::PathBuf,
    config: &crate::config::Config,
) -> Result<Vec<FileContent>> {
    use crate::utils::file_utils::{is_binary_file_path, is_test_file};

    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir_path) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            // Skip binary files
            if is_binary_file_path(&path) {
                continue;
            }

            // Get file name for exclusion checks
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_lowercase();

            // Skip excluded files (wildcard and exact match)
            let should_skip_file = config.excluded_files.iter().any(|excluded| {
                if excluded.contains('*') {
                    let pattern = excluded.replace('*', "").to_lowercase();
                    file_name.contains(&pattern)
                } else {
                    file_name == excluded.to_lowercase()
                }
            });
            if should_skip_file {
                continue;
            }

            // Skip excluded extensions
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if config.excluded_extensions.contains(&ext.to_lowercase()) {
                    continue;
                }
            }

            // Skip hidden files (unless include_hidden is set)
            if !config.include_hidden && file_name.starts_with('.') {
                continue;
            }

            // Skip test files (unless include_tests is set)
            if !config.include_tests && is_test_file(&path) {
                continue;
            }

            if let Ok(metadata) = std::fs::metadata(&path) {
                let file_size = metadata.len() as usize;
                let name = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                // Read up to max_file_size bytes (not the full file for oversized files)
                let max_read_size = config.max_file_size as usize;
                let read_size = file_size.min(max_read_size);

                if let Ok(mut file) = std::fs::File::open(&path) {
                    use std::io::Read;
                    let mut buffer = vec![0u8; read_size];
                    if let Ok(bytes_read) = file.read(&mut buffer) {
                        buffer.truncate(bytes_read);
                        // Decode to string, handling potential UTF-8 issues
                        let content = String::from_utf8_lossy(&buffer).into_owned();
                        // Truncate per-file at 256KB for prompt (but only if under max_file_size, otherwise we already truncated at max_file_size)
                        let truncated = if content.chars().count() > 256 * 1024 {
                            content.chars().take(256 * 1024).collect()
                        } else {
                            content
                        };
                        files.push(FileContent {
                            name,
                            path,
                            content: truncated,
                        });
                    }
                }
            }
        }
    }

    Ok(files)
}

/// Split files into batches, each batch's total content <= max_size.
/// Files are kept in lexicographic order within each batch.
fn split_into_batches(files: &[FileContent], max_size: usize) -> Vec<Vec<FileContent>> {
    let mut batches = Vec::new();
    let mut current_batch = Vec::new();
    let mut current_size = 0usize;

    for file in files {
        if current_size + file.content.len() > max_size && !current_batch.is_empty() {
            batches.push(std::mem::take(&mut current_batch));
            current_size = 0;
        }
        current_size += file.content.len();
        current_batch.push(file.clone());
    }

    if !current_batch.is_empty() {
        batches.push(current_batch);
    }

    batches
}

fn fallback_dossier(dir: &crate::types::DirectoryInfo) -> DirectoryDossier {
    DirectoryDossier {
        path: dir.path.clone(),
        name: dir.name.clone(),
        purpose: DirectoryPurpose::Other,
        file_count: dir.file_count,
        subdirectory_count: dir.subdirectory_count,
        importance_score: 0.0,
        summary: String::new(),
        key_files: Vec::new(),
        file_insights: Vec::new(),
    }
}
