use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;
use glob::glob;

use crate::config::ChunkingConfig;

/// Metadata about processed local documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalDocMetadata {
    pub file_path: String,
    pub file_type: DocFileType,
    pub last_modified: String,
    pub processed_content: String,
    /// Category this document belongs to (e.g., "architecture", "database", "api")
    #[serde(default)]
    pub category: String,
    /// Agents that should receive this document
    #[serde(default)]
    pub target_agents: Vec<String>,
    /// Chunk information if this is part of a chunked document
    #[serde(default)]
    pub chunk_info: Option<ChunkInfo>,
}

/// Information about a document chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    /// Chunk index (0-based)
    pub chunk_index: usize,
    /// Total number of chunks
    pub total_chunks: usize,
    /// Section title or context for this chunk
    pub section_context: String,
}

/// Supported documentation file types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DocFileType {
    Pdf,
    Markdown,
    Text,
    Sql,
    Yaml,
    Json,
}

/// Document chunker for splitting large documents
pub struct DocumentChunker {
    config: ChunkingConfig,
}

impl DocumentChunker {
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }
    
    /// Check if content needs chunking based on size
    pub fn needs_chunking(&self, content: &str) -> bool {
        self.config.enabled && content.len() >= self.config.min_size_for_chunking
    }
    
    /// Chunk content based on configured strategy
    pub fn chunk_content(&self, content: &str, file_type: &DocFileType) -> Vec<DocumentChunk> {
        if !self.needs_chunking(content) {
            return vec![DocumentChunk {
                content: content.to_string(),
                chunk_index: 0,
                total_chunks: 1,
                section_context: String::new(),
            }];
        }
        
        match self.config.strategy.as_str() {
            "semantic" => self.chunk_semantic(content, file_type),
            "paragraph" => self.chunk_by_paragraph(content),
            "fixed" | _ => self.chunk_fixed_size(content),
        }
    }
    
    /// Semantic chunking - split by sections/headers (best for Markdown)
    fn chunk_semantic(&self, content: &str, file_type: &DocFileType) -> Vec<DocumentChunk> {
        match file_type {
            DocFileType::Markdown => self.chunk_markdown_by_sections(content),
            DocFileType::Sql => self.chunk_sql_by_statements(content),
            DocFileType::Yaml | DocFileType::Json => self.chunk_by_paragraph(content),
            _ => self.chunk_fixed_size(content),
        }
    }
    
    /// Chunk Markdown by headers (## or ###)
    fn chunk_markdown_by_sections(&self, content: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_section = String::new();
        let mut section_stack: Vec<String> = Vec::new();
        
        for line in content.lines() {
            // Detect headers
            if line.starts_with("# ") {
                // H1 - major section boundary
                if !current_chunk.is_empty() {
                    chunks.push(DocumentChunk {
                        content: current_chunk.clone(),
                        chunk_index: chunks.len(),
                        total_chunks: 0, // Will be updated later
                        section_context: current_section.clone(),
                    });
                    current_chunk.clear();
                }
                section_stack.clear();
                section_stack.push(line[2..].trim().to_string());
                current_section = line[2..].trim().to_string();
            } else if line.starts_with("## ") {
                // H2 - check if we should split
                if current_chunk.len() >= self.config.max_chunk_size {
                    chunks.push(DocumentChunk {
                        content: current_chunk.clone(),
                        chunk_index: chunks.len(),
                        total_chunks: 0,
                        section_context: current_section.clone(),
                    });
                    current_chunk.clear();
                }
                if section_stack.len() > 1 {
                    section_stack.truncate(1);
                }
                section_stack.push(line[3..].trim().to_string());
                current_section = section_stack.join(" > ");
            } else if line.starts_with("### ") {
                // H3 - subsection
                if current_chunk.len() >= self.config.max_chunk_size {
                    chunks.push(DocumentChunk {
                        content: current_chunk.clone(),
                        chunk_index: chunks.len(),
                        total_chunks: 0,
                        section_context: current_section.clone(),
                    });
                    current_chunk.clear();
                }
                if section_stack.len() > 2 {
                    section_stack.truncate(2);
                }
                section_stack.push(line[4..].trim().to_string());
                current_section = section_stack.join(" > ");
            }
            
            current_chunk.push_str(line);
            current_chunk.push('\n');
            
            // Force split if too large
            if current_chunk.len() >= self.config.max_chunk_size + self.config.chunk_overlap {
                chunks.push(DocumentChunk {
                    content: current_chunk.clone(),
                    chunk_index: chunks.len(),
                    total_chunks: 0,
                    section_context: current_section.clone(),
                });
                // Keep overlap (UTF-8 safe: move the cut point to a char boundary)
                let overlap_start = current_chunk.len().saturating_sub(self.config.chunk_overlap);
                current_chunk =
                    crate::utils::slice_from_char_boundary(&current_chunk, overlap_start).to_string();
            }
        }
        
        // Add remaining content
        if !current_chunk.trim().is_empty() {
            chunks.push(DocumentChunk {
                content: current_chunk,
                chunk_index: chunks.len(),
                total_chunks: 0,
                section_context: current_section,
            });
        }
        
        // Update total_chunks
        let total = chunks.len();
        for chunk in &mut chunks {
            chunk.total_chunks = total;
        }
        
        chunks
    }
    
    /// Chunk SQL by statement boundaries (CREATE, ALTER, etc.)
    fn chunk_sql_by_statements(&self, content: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_context = String::new();
        
        // SQL statement keywords that typically start new logical blocks
        let statement_keywords = ["CREATE", "ALTER", "DROP", "INSERT", "UPDATE", "DELETE", 
                                   "GRANT", "REVOKE", "-- ==", "-- --"];
        
        for line in content.lines() {
            let upper_line = line.to_uppercase();
            
            // Check if this line starts a new statement
            let is_new_statement = statement_keywords.iter()
                .any(|kw| upper_line.trim_start().starts_with(kw));
            
            if is_new_statement && current_chunk.len() >= self.config.max_chunk_size {
                chunks.push(DocumentChunk {
                    content: current_chunk.clone(),
                    chunk_index: chunks.len(),
                    total_chunks: 0,
                    section_context: current_context.clone(),
                });
                current_chunk.clear();
            }
            
            // Extract context from CREATE statements
            if upper_line.contains("CREATE TABLE") || upper_line.contains("CREATE VIEW") {
                if let Some(name) = Self::extract_sql_object_name(line) {
                    current_context = name;
                }
            }
            
            current_chunk.push_str(line);
            current_chunk.push('\n');
        }
        
        if !current_chunk.trim().is_empty() {
            chunks.push(DocumentChunk {
                content: current_chunk,
                chunk_index: chunks.len(),
                total_chunks: 0,
                section_context: current_context,
            });
        }
        
        let total = chunks.len();
        for chunk in &mut chunks {
            chunk.total_chunks = total;
        }
        
        chunks
    }
    
    /// Extract object name from SQL CREATE statement
    fn extract_sql_object_name(line: &str) -> Option<String> {
        // Case-insensitive search on the ORIGINAL line: `to_uppercase()` can
        // change byte lengths (e.g. 'ß' -> "SS"), which would misalign any
        // offset computed on the uppercased copy.
        if let Some(pos) = Self::find_case_insensitive(line, "CREATE TABLE") {
            let rest = &line[pos + "CREATE TABLE".len()..];
            return Self::extract_first_word(rest);
        }
        if let Some(pos) = Self::find_case_insensitive(line, "CREATE VIEW") {
            let rest = &line[pos + "CREATE VIEW".len()..];
            return Self::extract_first_word(rest);
        }
        None
    }

    /// Byte offset of a case-insensitive ASCII needle in `haystack`, computed
    /// without allocating an uppercased copy (so offsets stay valid).
    fn find_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
        let hay = haystack.as_bytes();
        let nee = needle.as_bytes();
        if nee.is_empty() || hay.len() < nee.len() {
            return None;
        }
        'outer: for start in 0..=(hay.len() - nee.len()) {
            for (i, b) in nee.iter().enumerate() {
                if !hay[start + i].eq_ignore_ascii_case(b) {
                    continue 'outer;
                }
            }
            // Only return offsets that are valid char boundaries (needle is
            // ASCII, but the haystack may contain multi-byte characters).
            if haystack.is_char_boundary(start) {
                return Some(start);
            }
        }
        None
    }
    
    fn extract_first_word(s: &str) -> Option<String> {
        s.trim()
            .split(|c: char| c.is_whitespace() || c == '(' || c == '[')
            .next()
            .map(|w| w.trim_matches(|c| c == '"' || c == '\'' || c == '`' || c == '[' || c == ']').to_string())
            .filter(|w| !w.is_empty())
    }
    
    /// Chunk by paragraphs (double newlines)
    fn chunk_by_paragraph(&self, content: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        
        // Split by double newlines (paragraphs)
        let paragraphs: Vec<&str> = content.split("\n\n").collect();
        
        for para in paragraphs {
            if current_chunk.len() + para.len() > self.config.max_chunk_size && !current_chunk.is_empty() {
                chunks.push(DocumentChunk {
                    content: current_chunk.clone(),
                    chunk_index: chunks.len(),
                    total_chunks: 0,
                    section_context: String::new(),
                });
                // Keep overlap from end of previous chunk
                // (UTF-8 safe: move the cut point to a char boundary)
                let overlap_start = current_chunk.len().saturating_sub(self.config.chunk_overlap);
                current_chunk =
                    crate::utils::slice_from_char_boundary(&current_chunk, overlap_start).to_string();
            }
            
            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(para);
        }
        
        if !current_chunk.trim().is_empty() {
            chunks.push(DocumentChunk {
                content: current_chunk,
                chunk_index: chunks.len(),
                total_chunks: 0,
                section_context: String::new(),
            });
        }
        
        let total = chunks.len();
        for chunk in &mut chunks {
            chunk.total_chunks = total;
        }
        
        chunks
    }
    
    /// Fixed-size chunking with overlap.
    ///
    /// Termination is guaranteed *structurally*: `start` advances by a
    /// constant `step >= 1` and the loop stops as soon as the tail is
    /// consumed. The previous formulation
    /// (`start = end - overlap; if start >= end { break }`) did **not**
    /// guarantee progress: as soon as `end` was clamped to `chars.len()`
    /// (i.e. on the final, partial block) `start` became `len - overlap`,
    /// which is strictly less than `len` and equal to the previous `start`,
    /// so the guard `start >= end` never fired and the loop pushed chunks
    /// forever until the process ran out of memory. It terminated only when
    /// `chunk_overlap == 0`.
    fn chunk_fixed_size(&self, content: &str) -> Vec<DocumentChunk> {
        let chars: Vec<char> = content.chars().collect();
        // Guard against `max_chunk_size == 0` (a zero-width step would also
        // fail to advance).
        let max_chunk_size = self.config.max_chunk_size.max(1);
        // An `overlap >= max_chunk_size` config cannot produce a meaningful
        // step. `ChunkingConfig::validate` rejects it for the normal
        // pipeline, but `DocumentChunker::new` is also constructed directly,
        // so fall back to no overlap instead of degenerating into
        // one-character steps.
        let overlap = if self.config.chunk_overlap >= max_chunk_size {
            0
        } else {
            self.config.chunk_overlap
        };
        let step = max_chunk_size - overlap; // >= 1

        let mut chunks = Vec::new();
        let mut start = 0usize;

        while start < chars.len() {
            let end = (start + max_chunk_size).min(chars.len());
            let chunk_content: String = chars[start..end].iter().collect();

            chunks.push(DocumentChunk {
                content: chunk_content,
                chunk_index: chunks.len(),
                total_chunks: 0,
                section_context: format!("Part {}", chunks.len() + 1),
            });

            // Tail consumed: nothing left to emit.
            if end == chars.len() {
                break;
            }
            // Advance by a constant, strictly positive amount.
            start += step;
        }

        let total = chunks.len();
        for chunk in &mut chunks {
            chunk.total_chunks = total;
        }

        chunks
    }
}

/// A chunk of document content
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub content: String,
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub section_context: String,
}

/// Local documentation processor
pub struct LocalDocsProcessor;

impl LocalDocsProcessor {
    /// Extract text content from a PDF file
    pub fn extract_pdf_text(pdf_path: &Path) -> Result<String> {
        let bytes = fs::read(pdf_path)
            .with_context(|| format!("Failed to read PDF file: {:?}", pdf_path))?;

        let text = pdf_extract::extract_text_from_mem(&bytes)
            .with_context(|| format!("Failed to extract text from PDF: {:?}", pdf_path))?;

        Ok(text)
    }

    /// Read markdown file content
    pub fn read_markdown(md_path: &Path) -> Result<String> {
        fs::read_to_string(md_path)
            .with_context(|| format!("Failed to read Markdown file: {:?}", md_path))
    }

    /// Read text file content
    pub fn read_text(txt_path: &Path) -> Result<String> {
        fs::read_to_string(txt_path)
            .with_context(|| format!("Failed to read text file: {:?}", txt_path))
    }
    
    /// Read SQL file content with schema header
    pub fn read_sql(sql_path: &Path) -> Result<String> {
        let content = fs::read_to_string(sql_path)
            .with_context(|| format!("Failed to read SQL file: {:?}", sql_path))?;
        
        // Add a header to help LLM understand this is database schema
        Ok(format!("-- Database Schema Definition\n-- File: {}\n\n{}", 
            sql_path.file_name().unwrap_or_default().to_string_lossy(),
            content
        ))
    }
    
    /// Read YAML file content (for OpenAPI specs, K8s configs, etc.)
    pub fn read_yaml(yaml_path: &Path) -> Result<String> {
        fs::read_to_string(yaml_path)
            .with_context(|| format!("Failed to read YAML file: {:?}", yaml_path))
    }
    
    /// Read JSON file content (for OpenAPI specs, configs, etc.)
    pub fn read_json(json_path: &Path) -> Result<String> {
        fs::read_to_string(json_path)
            .with_context(|| format!("Failed to read JSON file: {:?}", json_path))
    }
    
    /// Process a documentation file with chunking support
    /// Returns multiple LocalDocMetadata entries if the document is chunked
    pub fn process_file_with_chunking(
        file_path: &Path,
        category: &str,
        target_agents: &[String],
        chunking_config: Option<&ChunkingConfig>,
    ) -> Result<Vec<LocalDocMetadata>> {
        let file_type = Self::detect_file_type(file_path)?;
        
        let raw_content = match file_type {
            DocFileType::Pdf => Self::extract_pdf_text(file_path)?,
            DocFileType::Markdown => Self::read_markdown(file_path)?,
            DocFileType::Text => Self::read_text(file_path)?,
            DocFileType::Sql => Self::read_sql(file_path)?,
            DocFileType::Yaml => Self::read_yaml(file_path)?,
            DocFileType::Json => Self::read_json(file_path)?,
        };

        let metadata = fs::metadata(file_path)?;
        let last_modified = format!("{:?}", metadata.modified()?);
        let file_path_str = file_path.to_string_lossy().to_string();
        
        // Determine if we should chunk
        let config = chunking_config.cloned().unwrap_or_default();
        // Fail fast on invalid configs (e.g. overlap >= max_chunk_size would
        // otherwise make the fixed-size loop spin forever and OOM).
        config.validate()?;
        let chunker = DocumentChunker::new(config);
        
        if !chunker.needs_chunking(&raw_content) {
            // No chunking needed - return single document
            return Ok(vec![LocalDocMetadata {
                file_path: file_path_str,
                file_type,
                last_modified,
                processed_content: raw_content,
                category: category.to_string(),
                target_agents: target_agents.to_vec(),
                chunk_info: None,
            }]);
        }
        
        // Chunk the content
        let chunks = chunker.chunk_content(&raw_content, &file_type);
        
        // Create metadata for each chunk
        let docs: Vec<LocalDocMetadata> = chunks
            .into_iter()
            .map(|chunk| LocalDocMetadata {
                file_path: file_path_str.clone(),
                file_type: file_type.clone(),
                last_modified: last_modified.clone(),
                processed_content: chunk.content,
                category: category.to_string(),
                target_agents: target_agents.to_vec(),
                chunk_info: Some(ChunkInfo {
                    chunk_index: chunk.chunk_index,
                    total_chunks: chunk.total_chunks,
                    section_context: chunk.section_context,
                }),
            })
            .collect();
        
        Ok(docs)
    }
    
    /// Expand glob patterns to actual file paths
    pub fn expand_glob_patterns(patterns: &[String], base_path: Option<&Path>) -> Vec<std::path::PathBuf> {
        let mut files = Vec::new();
        
        for pattern in patterns {
            let pattern_path = Path::new(pattern);
            let full_pattern = if pattern_path.is_absolute() {
                pattern.clone()
            } else if let Some(base) = base_path {
                base.join(pattern_path).to_string_lossy().to_string()
            } else {
                pattern.clone()
            };
            
            match glob(&full_pattern) {
                Ok(paths) => {
                    for entry in paths.flatten() {
                        if entry.is_file() {
                            // Only include supported file types
                            if let Some(ext) = entry.extension().and_then(|e| e.to_str()) {
                                match ext.to_lowercase().as_str() {
                                    // Documentation files
                                    "pdf" | "md" | "markdown" | "txt" | "text" |
                                    // Database schema files
                                    "sql" |
                                    // API specs and config files
                                    "yaml" | "yml" | "json" => {
                                        files.push(entry);
                                    }
                                    _ => {} // Skip unsupported file types
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  ⚠️  Invalid glob pattern '{}': {}", pattern, e);
                }
            }
        }
        
        files
    }

    /// Detect file type from extension
    fn detect_file_type(file_path: &Path) -> Result<DocFileType> {
        let extension = file_path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow::anyhow!("No file extension found"))?;

        match extension.to_lowercase().as_str() {
            "pdf" => Ok(DocFileType::Pdf),
            "md" | "markdown" => Ok(DocFileType::Markdown),
            "txt" | "text" => Ok(DocFileType::Text),
            "sql" => Ok(DocFileType::Sql),
            "yaml" | "yml" => Ok(DocFileType::Yaml),
            "json" => Ok(DocFileType::Json),
            _ => Err(anyhow::anyhow!("Unsupported file type: {}", extension)),
        }
    }

    /// Format documentation content for LLM with custom header and options
    pub fn format_for_llm_with_options(
        docs: &[LocalDocMetadata],
        custom_header: Option<&str>,
        include_category: bool,
    ) -> String {
        let mut formatted = String::new();
        
        // Add header
        if let Some(header) = custom_header {
            formatted.push_str(header);
        } else {
            formatted.push_str("# Local Technical Documentation\n\n");
        }

        for doc in docs.iter() {
            let filename = Path::new(&doc.file_path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| doc.file_path.clone());
            
            // Handle chunked documents
            let title = if let Some(ref chunk_info) = doc.chunk_info {
                if chunk_info.total_chunks > 1 {
                    if chunk_info.section_context.is_empty() {
                        format!("{} (Part {}/{})", filename, chunk_info.chunk_index + 1, chunk_info.total_chunks)
                    } else {
                        format!("{} - {} (Part {}/{})", filename, chunk_info.section_context, chunk_info.chunk_index + 1, chunk_info.total_chunks)
                    }
                } else {
                    filename
                }
            } else {
                filename
            };
            
            formatted.push_str(&format!("\n---\n\n## {}\n\n", title));
            formatted.push_str(&format!("**Source:** {}\n", doc.file_path));
            
            if include_category && !doc.category.is_empty() {
                formatted.push_str(&format!("**Category:** {}\n", doc.category));
            }
            
            // Add chunk context if present
            if let Some(ref chunk_info) = doc.chunk_info {
                if chunk_info.total_chunks > 1 {
                    formatted.push_str(&format!("**Chunk:** {}/{}\n", chunk_info.chunk_index + 1, chunk_info.total_chunks));
                    if !chunk_info.section_context.is_empty() {
                        formatted.push_str(&format!("**Section:** {}\n", chunk_info.section_context));
                    }
                }
            }
            
            formatted.push_str(&format!("**Type:** {:?}\n\n", doc.file_type));
            formatted.push_str(&doc.processed_content);
            formatted.push_str("\n\n");
        }

        formatted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_file_type() {
        assert_eq!(
            LocalDocsProcessor::detect_file_type(Path::new("doc.pdf")).unwrap(),
            DocFileType::Pdf
        );
        assert_eq!(
            LocalDocsProcessor::detect_file_type(Path::new("readme.md")).unwrap(),
            DocFileType::Markdown
        );
        assert_eq!(
            LocalDocsProcessor::detect_file_type(Path::new("notes.txt")).unwrap(),
            DocFileType::Text
        );
    }

    /// Regression (P0-OOM): the fixed-size loop must always advance.
    ///
    /// With a *valid* config the old guard still froze `start` on the final
    /// partial block (`start = len - overlap`, unchanged), so the loop pushed
    /// chunks until the process was killed by the OOM killer — for every
    /// non-empty input, since the last block is always clamped.
    #[test]
    fn test_fixed_size_chunking_terminates_with_valid_config() {
        let config = ChunkingConfig {
            enabled: true,
            max_chunk_size: 8,
            chunk_overlap: 2, // valid, but the tail still gets clamped
            min_size_for_chunking: 1,
            strategy: "fixed".to_string(),
        };
        let chunker = DocumentChunker::new(config);
        let chunks = chunker.chunk_content("aaaaaaaabbbbbbbbcccccccc", &DocFileType::Text);
        // 24 chars, step = 8 - 2 = 6 -> starts 0, 6, 12, 18 -> 4 chunks
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].content, "aaaaaaaa");
        assert_eq!(chunks[1].content, "aabbbbbb");
        assert!(chunks.iter().all(|c| c.total_chunks == 4));

        // Content shorter than max_chunk_size must also terminate
        let short = chunker.chunk_content("hello", &DocFileType::Text);
        assert_eq!(short.len(), 1);
        assert_eq!(short[0].content, "hello");
    }

    /// Regression (P0-OOM): multi-byte content (char-indexed slicing) must
    /// terminate too — the old code hung here as well.
    #[test]
    fn test_fixed_size_chunking_multibyte_terminates() {
        let config = ChunkingConfig {
            enabled: true,
            max_chunk_size: 4,
            chunk_overlap: 1,
            min_size_for_chunking: 1,
            strategy: "fixed".to_string(),
        };
        let chunker = DocumentChunker::new(config);
        let content = "中文内容测试用例集合";
        let chunks = chunker.chunk_content(content, &DocFileType::Text);
        // step = 3 -> starts 0,3,6 -> last chunk consumes the tail
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[2].content, "用例集合");
        assert!(chunks.iter().all(|c| c.total_chunks == 3));
    }

    /// Regression (P0-OOM): `chunk_overlap == 0` and `max_chunk_size == 1`
    /// are the degenerate edges of the same loop.
    #[test]
    fn test_fixed_size_chunking_edge_configs_terminate() {
        let no_overlap = DocumentChunker::new(ChunkingConfig {
            enabled: true,
            max_chunk_size: 4,
            chunk_overlap: 0,
            min_size_for_chunking: 1,
            strategy: "fixed".to_string(),
        });
        let chunks = no_overlap.chunk_content("abcdefgh", &DocFileType::Text);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].content, "abcd");

        let width_one = DocumentChunker::new(ChunkingConfig {
            enabled: true,
            max_chunk_size: 1,
            chunk_overlap: 5, // invalid, falls back to zero overlap
            min_size_for_chunking: 1,
            strategy: "fixed".to_string(),
        });
        let chunks = width_one.chunk_content("abc", &DocFileType::Text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[2].content, "c");
    }

    /// Regression (P0-OOM): the real production path — default config
    /// (`semantic` strategy falls back to fixed-size for plain-text docs) with
    /// a document long enough to be chunked. This used to hang forever and
    /// grow the heap until the process was killed.
    #[test]
    fn test_default_config_plain_text_document_chunks_and_terminates() {
        let config = ChunkingConfig::default(); // max=8000, overlap=200, semantic
        let chunker = DocumentChunker::new(config);
        let content = "x".repeat(16_000);

        let chunks = chunker.chunk_content(&content, &DocFileType::Text);

        // step = 8000 - 200 = 7800 -> starts 0, 7800, 15600
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].content.len(), 8000);
        assert_eq!(chunks[1].content.len(), 8000);
        assert_eq!(chunks[2].content.len(), 400); // tail
        assert!(chunks.iter().all(|c| c.total_chunks == 3));
        // No chunk may be empty and the loop must not emit duplicates forever
        assert!(chunks.iter().all(|c| !c.content.is_empty()));
    }

    /// Regression (P0-5): overlap >= max_chunk_size must be rejected by
    /// validation instead of looping forever.
    #[test]
    fn test_chunking_config_rejects_bad_overlap() {
        let bad = ChunkingConfig {
            enabled: true,
            max_chunk_size: 100,
            chunk_overlap: 100,
            ..ChunkingConfig::default()
        };
        assert!(bad.validate().is_err());

        let worse = ChunkingConfig {
            enabled: true,
            max_chunk_size: 100,
            chunk_overlap: 500,
            ..ChunkingConfig::default()
        };
        assert!(worse.validate().is_err());

        let good = ChunkingConfig {
            enabled: true,
            max_chunk_size: 100,
            chunk_overlap: 20,
            ..ChunkingConfig::default()
        };
        assert!(good.validate().is_ok());
    }

    /// Regression (P0-5): even if a bad config bypasses validation, the
    /// fixed-size loop must terminate (hard progress guarantee).
    #[test]
    fn test_fixed_size_chunking_terminates_on_bad_config() {
        let bad = ChunkingConfig {
            enabled: true,
            max_chunk_size: 8,
            chunk_overlap: 16, // >= max_chunk_size
            min_size_for_chunking: 1,
            strategy: "fixed".to_string(),
        };
        let chunker = DocumentChunker::new(bad);
        let chunks = chunker.chunk_content("aaaaaaaabbbbbbbbcccccccc", &DocFileType::Text);
        assert!(!chunks.is_empty());
        // Terminates and produces bounded output
        assert!(chunks.len() < 100);
    }

    /// Regression (P0-4): multi-byte (Chinese) content must not panic when
    /// the semantic chunker keeps an overlap from the previous chunk.
    #[test]
    fn test_markdown_chunking_multibyte_no_panic() {
        let config = ChunkingConfig {
            enabled: true,
            max_chunk_size: 120,
            chunk_overlap: 50, // intentionally not char-aligned
            min_size_for_chunking: 1,
            strategy: "semantic".to_string(),
        };
        let chunker = DocumentChunker::new(config);
        // 6-byte-per-2-chars content: any byte offset lands mid-character
        let content = format!("# 标题\n\n{}\n\n## 二级\n\n{}", "中".repeat(200), "文".repeat(200));
        let chunks = chunker.chunk_content(&content, &DocFileType::Markdown);
        assert!(!chunks.is_empty());
    }

    /// Regression (P0-4): paragraph chunking with multibyte overlap.
    #[test]
    fn test_paragraph_chunking_multibyte_no_panic() {
        let config = ChunkingConfig {
            enabled: true,
            max_chunk_size: 90,
            chunk_overlap: 37, // intentionally not char-aligned
            min_size_for_chunking: 1,
            strategy: "paragraph".to_string(),
        };
        let chunker = DocumentChunker::new(config);
        let content = format!("{}\n\n{}\n\n{}", "甲".repeat(80), "乙".repeat(80), "丙".repeat(80));
        let chunks = chunker.chunk_content(&content, &DocFileType::Text);
        assert!(!chunks.is_empty());
    }

    /// Regression (P0-4): SQL object name extraction must handle content
    /// whose `to_uppercase` representation changes byte length (e.g. 'ß').
    #[test]
    fn test_extract_sql_object_name_with_special_chars() {
        assert_eq!(
            DocumentChunker::extract_sql_object_name("CREATE TABLE users (id INT)"),
            Some("users".to_string())
        );
        assert_eq!(
            DocumentChunker::extract_sql_object_name("create table `orders` (id int)"),
            Some("orders".to_string())
        );
        // 'ß' uppercases to "SS" (changing the byte length of the uppercased
        // copy) — must not panic or misread the object name.
        assert_eq!(
            DocumentChunker::extract_sql_object_name("-- straße note\ncreate view v_ß as select 1"),
            Some("v_ß".to_string())
        );
        assert_eq!(DocumentChunker::extract_sql_object_name("SELECT 1"), None);
    }
}
