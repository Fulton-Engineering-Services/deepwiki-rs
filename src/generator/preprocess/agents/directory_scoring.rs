use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::generator::agent_executor::{AgentExecuteParams, extract};
use crate::generator::context::GeneratorContext;
use crate::types::DirectoryInfo;

/// LLM directory scoring result — path-keyed to avoid index mismatch
#[derive(Debug, Serialize, Deserialize, Clone, Default, JsonSchema)]
#[serde(default)]
pub struct DirectoryScoreResult {
    /// Absolute or relative path matching DirectoryInfo.path
    #[serde(default)]
    pub path: String,
    #[serde(default, deserialize_with = "deserialize_f64_lenient")]
    pub score: f64,
    #[serde(default)]
    pub reasoning: String,
}

fn deserialize_f64_lenient<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Ok(s.parse::<f64>().unwrap_or(0.0)),
        serde_json::Value::Bool(v) => Ok(if v { 1.0 } else { 0.0 }),
        _ => Ok(0.0),
    }
}

/// Directory scoring response containing scores for all scored directories
#[derive(Debug, Serialize, Deserialize, Clone, Default, JsonSchema)]
#[serde(default)]
pub struct DirectoryScoringResponse {
    #[serde(default)]
    pub scores: Vec<DirectoryScoreResult>,
}

/// Directory scorer — uses LLM to score directories by business value
pub struct DirectoryScorer;

impl DirectoryScorer {
    pub fn new() -> Self {
        Self
    }

    /// Score multiple directories with LLM
    pub async fn score_directories(
        &self,
        context: &GeneratorContext,
        directories: &[DirectoryInfo],
    ) -> Result<HashMap<PathBuf, f64>> {
        if directories.is_empty() {
            return Ok(HashMap::new());
        }

        let prompt_sys = "You are a professional code architecture analyst specializing in evaluating the business importance of code directories.".to_string();
        let project_path = &context.config.project_path;
        let prompt_user = self.build_scoring_prompt(directories, project_path);

        let cache_scope = format!(
            "directory_scoring_{}",
            context.config.project_path.to_string_lossy().replace(['/', '\\', ':', '.'], "_")
        );
        let project_name = context
            .config
            .project_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let dir_list: String = directories
            .iter()
            .take(5)
            .map(|d| d.path.strip_prefix(project_path).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| d.name.clone()))
            .collect::<Vec<_>>()
            .join(", ");
        let more = if directories.len() > 5 {
            format!(", +{} more", directories.len() - 5)
        } else {
            String::new()
        };
        let log_tag = format!("dir_score({}): {} dirs ({}{})", project_name, directories.len(), dir_list, more);

        let response: DirectoryScoringResponse = extract(
            context,
            AgentExecuteParams {
                prompt_sys,
                prompt_user,
                cache_scope: cache_scope.to_string(),
                log_tag,
                progress: None,
            },
        )
        .await?;

        // Build path → score map from keyed response.
        // Keys are normalized (trailing '/', leading './', backslashes) so
        // small LLM formatting differences don't silently drop scores.
        let mut score_map: HashMap<String, f64> = HashMap::new();
        for result in &response.scores {
            let normalized_path = Self::normalize_dir_key(&result.path);
            if !normalized_path.is_empty() {
                score_map.insert(normalized_path, result.score.clamp(0.0, 1.0));
            }
        }

        // Match directories by path, warn on missing entries.
        // The prompt asks the LLM to key its response by `rel_path`, so the
        // primary lookup key here must also be the relative path (previously
        // an absolute-path lookup that always missed and fell back to the
        // bare directory name — wrong scores for nested same-named dirs).
        let mut scores = HashMap::new();
        let mut missing = 0usize;
        for dir in directories {
            let abs_path_str = Self::normalize_dir_key(&dir.path.to_string_lossy());
            let rel_path_str = Self::normalize_dir_key(
                &dir.path
                    .strip_prefix(project_path)
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| dir.name.clone()),
            );
            let name_key = Self::normalize_dir_key(&dir.name);
            if let Some(&score) = score_map.get(&rel_path_str) {
                scores.insert(dir.path.clone(), score);
            } else if let Some(&score) = score_map.get(&abs_path_str) {
                // Tolerate an LLM that answered with the absolute path
                scores.insert(dir.path.clone(), score);
            } else if let Some(&score) = score_map.get(&name_key) {
                // Last resort: bare directory name (ambiguous when nested
                // dirs share a name, but better than dropping the score)
                scores.insert(dir.path.clone(), score);
            } else {
                missing += 1;
                scores.insert(dir.path.clone(), 0.0);
            }
        }
        if missing > 0 {
            let example = directories
                .first()
                .map(|d| {
                    Self::normalize_dir_key(
                        &d.path
                            .strip_prefix(project_path)
                            .map(|p| p.to_string_lossy().into_owned())
                            .unwrap_or_else(|_| d.name.clone()),
                    )
                })
                .unwrap_or_else(|| "src".to_string());
            eprintln!(
                "⚠️  Warning: {} of {} directories had no matching LLM score (using 0.0). \
                 Expected the LLM to key scores by relative paths like '{}' — \
                 check the scoring response for path mismatches.",
                missing,
                directories.len(),
                example
            );
        }

        Ok(scores)
    }

    /// Normalize a directory key for matching: trim whitespace, drop a
    /// leading "./" and any trailing "/", and unify separators to '/'.
    fn normalize_dir_key(raw: &str) -> String {
        let mut key = raw.trim().replace('\\', "/");
        while key.starts_with("./") {
            key = key[2..].to_string();
        }
        while key.ends_with('/') && key.len() > 1 {
            key.pop();
        }
        if key == "." {
            key.clear();
        }
        key
    }

    fn build_scoring_prompt(&self, directories: &[DirectoryInfo], project_path: &PathBuf) -> String {
        let mut dir_list = String::new();
        for dir in directories {
            let relative_path = dir.path.strip_prefix(project_path)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| dir.name.clone());
            let file_names: Vec<String> = std::fs::read_dir(&dir.path)
                .ok()
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .filter_map(|e| e.file_name().to_str().map(String::from))
                        .filter(|n| !n.starts_with('.'))
                        .take(20)
                        .collect()
                })
                .unwrap_or_default();

            dir_list.push_str(&format!(
                "- {} (rel_path: {}, {} files, {} subdirs): {:?}\n",
                dir.name,
                relative_path,
                dir.file_count,
                dir.subdirectory_count,
                file_names
            ));
        }

        format!(
            r#"Rate the business importance of each directory for a software project.

Rate based on:
1. Business value - does it contain core business logic, APIs, or data layer?
2. Code concentration - is it a hub with many imports/exports?
3. Infrastructure role - is it a core package, main entry, or config layer?
IMPORTANT: Backend directories (*.py, *.go, *.rs, *.java, *.kt, etc.) should be rated higher than frontend directories (*.ts, *.js, *.tsx, *.vue, *.jsx, etc.) when business value is comparable.

Directories to rate:
{}

Output JSON with a "scores" array, each entry with "path" (use the exact rel_path shown), "score" (0.0-1.0) and "reasoning":
{{"scores": [{{"path": "src", "score": 0.8, "reasoning": "..."}}, {{"path": "cmd", "score": 0.7, "reasoning": "..."}}, ...]}}

IMPORTANT: Output valid JSON only, no markdown fences."#,
            dir_list
        )
    }
}

#[cfg(test)]
mod tests {
    use super::DirectoryScorer;

    #[test]
    fn test_normalize_dir_key() {
        assert_eq!(DirectoryScorer::normalize_dir_key("src"), "src");
        assert_eq!(DirectoryScorer::normalize_dir_key("  src/  "), "src");
        assert_eq!(DirectoryScorer::normalize_dir_key("./src"), "src");
        assert_eq!(DirectoryScorer::normalize_dir_key("./src/utils/"), "src/utils");
        assert_eq!(DirectoryScorer::normalize_dir_key("src\\utils"), "src/utils");
        assert_eq!(DirectoryScorer::normalize_dir_key("."), "");
        assert_eq!(DirectoryScorer::normalize_dir_key(""), "");
        // Must not mangle parent-dir prefixes (only matching keys are normalized)
        assert_eq!(DirectoryScorer::normalize_dir_key("../foo"), "../foo");
    }
}
