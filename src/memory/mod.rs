use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Memory metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub access_counts: HashMap<String, u64>,
    pub data_sizes: HashMap<String, usize>,
    pub total_size: usize,
}

impl MemoryMetadata {
    pub fn new() -> Self {
        Self {
            created_at: Utc::now(),
            last_updated: Utc::now(),
            access_counts: HashMap::new(),
            data_sizes: HashMap::new(),
            total_size: 0,
        }
    }
}

/// On-disk snapshot format version. Bump when `MemorySnapshot` changes shape;
/// loaders refuse versions they don't understand instead of guessing.
pub const MEMORY_SNAPSHOT_VERSION: u32 = 1;

/// Snapshot file name, written under `Config::internal_path` (`.litho/`).
pub const MEMORY_SNAPSHOT_FILE: &str = "memory.json";

/// Persisted copy of [`Memory`], written after a full run so a later
/// `--only-agent-content` run can rebuild the agent content set without
/// re-running the pipeline. `fingerprint` records the inputs the artifacts
/// were derived from (project, language, models, git state); hydration
/// compares it against the current config so stale snapshots fail loudly.
#[derive(Debug, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub version: u32,
    pub fingerprint: String,
    pub data: HashMap<String, Value>,
    pub metadata: MemoryMetadata,
}

/// Unified memory manager
#[derive(Debug)]
pub struct Memory {
    data: HashMap<String, Value>,
    metadata: MemoryMetadata,
}

impl Memory {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            metadata: MemoryMetadata::new(),
        }
    }

    /// Store data to specified scope and key
    pub fn store<T>(&mut self, scope: &str, key: &str, data: T) -> Result<()>
    where
        T: Serialize,
    {
        let full_key = format!("{}:{}", scope, key);
        let serialized = serde_json::to_value(data)?;

        // Calculate data size
        let data_size = serialized.to_string().len();

        // Update metadata
        if let Some(old_size) = self.metadata.data_sizes.get(&full_key) {
            self.metadata.total_size -= old_size;
        }
        self.metadata.data_sizes.insert(full_key.clone(), data_size);
        self.metadata.total_size += data_size;
        self.metadata.last_updated = Utc::now();

        self.data.insert(full_key, serialized);
        Ok(())
    }

    /// Get data from specified scope and key
    pub fn get<T>(&mut self, scope: &str, key: &str) -> Option<T>
    where
        T: for<'a> Deserialize<'a>,
    {
        let full_key = format!("{}:{}", scope, key);

        // Update access count
        *self
            .metadata
            .access_counts
            .entry(full_key.clone())
            .or_insert(0) += 1;

        self.data
            .get(&full_key)
            .and_then(|value| serde_json::from_value(value.clone()).ok())
    }

    /// List all keys in the specified scope
    pub fn list_keys(&self, scope: &str) -> Vec<String> {
        let prefix = format!("{}:", scope);
        self.data
            .keys()
            .filter(|key| key.starts_with(&prefix))
            .map(|key| key[prefix.len()..].to_string())
            .collect()
    }

    /// Check if specified data exists
    pub fn has_data(&self, scope: &str, key: &str) -> bool {
        let full_key = format!("{}:{}", scope, key);
        self.data.contains_key(&full_key)
    }

    /// Get memory usage statistics
    pub fn get_usage_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        for (key, size) in &self.metadata.data_sizes {
            let scope = key.split(':').next().unwrap_or("unknown").to_string();
            *stats.entry(scope).or_insert(0) += size;
        }

        stats
    }

    /// Export the contents as a versioned snapshot for persistence.
    pub fn to_snapshot(&self, fingerprint: &str) -> MemorySnapshot {
        MemorySnapshot {
            version: MEMORY_SNAPSHOT_VERSION,
            fingerprint: fingerprint.to_string(),
            data: self.data.clone(),
            metadata: self.metadata.clone(),
        }
    }

    /// Rebuild memory from a persisted snapshot.
    pub fn from_snapshot(snapshot: MemorySnapshot) -> Self {
        Memory {
            data: snapshot.data,
            metadata: snapshot.metadata,
        }
    }

    /// Whether any artifacts are stored.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip_preserves_artifacts() {
        let mut m = Memory::new();
        m.store("studies_research", "AreaTree", serde_json::json!({"a": 1}))
            .unwrap();
        assert!(!m.is_empty());

        let snapshot = m.to_snapshot("fp1");
        assert_eq!(snapshot.version, MEMORY_SNAPSHOT_VERSION);
        assert_eq!(snapshot.fingerprint, "fp1");

        let mut restored = Memory::from_snapshot(snapshot);
        assert!(!restored.is_empty());
        assert_eq!(
            restored
                .get::<Value>("studies_research", "AreaTree")
                .unwrap(),
            serde_json::json!({"a": 1})
        );
        assert!(restored.has_data("studies_research", "AreaTree"));
        assert_eq!(
            restored.list_keys("studies_research"),
            vec!["AreaTree".to_string()]
        );
    }

    #[test]
    fn snapshot_serializes_to_json_and_back() {
        let mut m = Memory::new();
        m.store(
            "preprocess",
            "code_insights",
            serde_json::json!({"x": [1, 2]}),
        )
        .unwrap();

        let json = serde_json::to_string(&m.to_snapshot("fp")).unwrap();
        let parsed: MemorySnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, MEMORY_SNAPSHOT_VERSION);
        assert_eq!(parsed.fingerprint, "fp");

        let mut restored = Memory::from_snapshot(parsed);
        assert_eq!(
            restored
                .get::<Value>("preprocess", "code_insights")
                .unwrap(),
            serde_json::json!({"x": [1, 2]})
        );
    }

    #[test]
    fn empty_memory_roundtrips_as_empty() {
        let snapshot = Memory::new().to_snapshot("fp");
        let restored = Memory::from_snapshot(snapshot);
        assert!(restored.is_empty());
    }
}
