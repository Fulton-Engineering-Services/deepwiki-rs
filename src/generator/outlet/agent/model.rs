//! Assembly of the agent-facing page tree from structured research data.
//!
//! This module is pure and synchronous: the outlet loads research from memory into
//! a [`ResearchBundle`], [`build_site`] walks the recursive [`AreaTree`] and emits a
//! flat [`AgentSite`] (pages + diagram files) with relative paths resolved. Hrefs
//! are computed later by `render.rs` once the agent root is known.
//!
//! The tree mirrors the area tree at arbitrary depth; modules hang off the area
//! they were analyzed under and tightly-scoped per-file pages sit at the leaves.

use crate::generator::outlet::agent::diagrams;
use crate::generator::outlet::agent::links::{NameRegistry, clamp_component, slugify, verbatim_leaf};
use crate::generator::research::area_tree::{AreaNode, AreaTree};
use crate::generator::research::types::{
    BoundaryAnalysisReport, BusinessFlow, DatabaseOverviewReport, DomainModulesReport,
    KeyModuleReport, SystemContextReport,
};
use crate::types::FileInsight;
use std::collections::HashMap;
use std::path::PathBuf;

/// Directory names reserved for non-area content beneath an area folder. A
/// sub-area whose slug collides with one of these gets an `-area` suffix so it
/// never shadows the `modules/`, `files/`, or `diagrams/` directories.
const RESERVED_AREA_SLUGS: &[&str] = &["modules", "files", "diagrams", "topics", "index"];

/// Output-relative path of the agent root's index page.
pub const ROOT_INDEX: &str = "index.md";

/// Per-page content limits that keep every agent page short.
#[derive(Debug, Clone)]
pub struct AgentOptions {
    /// Emit dedicated diagram files (and their page links).
    pub diagrams: bool,
    /// Cap on bullets rendered in any detail block.
    pub max_list_items: usize,
    /// Cap on characters for any single prose field (summary, descriptions).
    pub max_desc_chars: usize,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            diagrams: true,
            max_list_items: 8,
            max_desc_chars: 280,
        }
    }
}

/// A key-module report tagged with the area it was analyzed under, when known.
#[derive(Debug, Clone)]
pub struct OwnedKeyModule {
    /// Area id from the `KeyModulesInsight_<area.id>_*` research key, if present.
    pub area_id: Option<String>,
    pub report: KeyModuleReport,
}

/// All research needed to synthesize the agent set, loaded once from memory.
#[derive(Debug, Clone, Default)]
pub struct ResearchBundle {
    pub project_name: String,
    pub system_context: Option<SystemContextReport>,
    pub domain_modules: Option<DomainModulesReport>,
    pub boundary: Option<BoundaryAnalysisReport>,
    pub database: Option<DatabaseOverviewReport>,
    pub area_tree: Option<AreaTree>,
    pub modules: Vec<OwnedKeyModule>,
    pub file_insights: Vec<FileInsight>,
}

/// A single navigable link to another page or diagram within the agent set.
#[derive(Debug, Clone)]
pub struct NavLink {
    pub label: String,
    /// Target path relative to the agent root (e.g. `tree/core/cache/index.md`).
    pub target: String,
}

/// A titled group of navigation links rendered as a short bullet list.
#[derive(Debug, Clone)]
pub struct NavGroup {
    pub heading: String,
    pub links: Vec<NavLink>,
}

/// A short bullet block used for tight facts (responsibilities, interfaces, ...).
#[derive(Debug, Clone)]
pub struct DetailBlock {
    pub heading: String,
    pub items: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageKind {
    Index,
    Topic,
    Area,
    Dir,
    Module,
    File,
}

/// One short, interlinked agent page.
#[derive(Debug, Clone)]
pub struct AgentPage {
    /// Path relative to the agent root.
    pub rel_path: String,
    pub kind: PageKind,
    pub title: String,
    /// One-to-three-line role/description (already capped).
    pub summary: String,
    /// Trail from the root down to this page.
    pub breadcrumbs: Vec<NavLink>,
    /// Navigation groups (sub-areas, modules, files, topics, ...).
    pub nav_groups: Vec<NavGroup>,
    /// Project-relative source paths/folders this page is grounded in.
    pub source_paths: Vec<String>,
    /// Diagram pages this page links to (targets relative to agent root).
    pub diagram_links: Vec<NavLink>,
    /// Tight fact blocks (per-file responsibilities, module interaction, ...).
    pub detail_blocks: Vec<DetailBlock>,
    /// Short prose paragraphs (e.g. a module's implementation note).
    pub notes: Vec<String>,
}

/// A dedicated Mermaid diagram file.
#[derive(Debug, Clone)]
pub struct DiagramFile {
    /// Path relative to the agent root.
    pub rel_path: String,
    pub title: String,
    pub caption: String,
    /// Raw Mermaid source (without fences).
    pub body: String,
    /// Back-link to the page that owns this diagram.
    pub owner: NavLink,
}

/// The complete agent output set.
#[derive(Debug, Clone, Default)]
pub struct AgentSite {
    pub project_name: String,
    pub pages: Vec<AgentPage>,
    pub diagrams: Vec<DiagramFile>,
}

/// Intermediate frame for one area node while the tree is assembled.
#[derive(Debug)]
struct Frame {
    id: String,
    name: String,
    description: String,
    root_paths: Vec<String>,
    depth: usize,
    rel_dir: String,
    children: Vec<usize>,
    /// Indices into the assembled module page list.
    module_slots: Vec<OwnedKeyModule>,
    /// File entries assigned to this area.
    file_slots: Vec<FileEntry>,
}

/// A tightly-scoped per-file page candidate, carrying its mirrored location.
#[derive(Debug, Clone)]
struct FileEntry {
    /// Project-relative source path (used for the Source link).
    path: String,
    /// Verbatim source basename (used as the leaf page name stem).
    name: String,
    /// Source directory segments relative to the matched area root, i.e. the
    /// folder chain the page mirrors under `files/` (excludes the basename).
    mirror_dirs: Vec<String>,
    /// The matched area root this file's mirror is relative to.
    source_root: String,
    summary: String,
    responsibilities: Vec<String>,
    interfaces: Vec<String>,
    dependencies: Vec<String>,
}

/// Build the whole agent site from a research bundle.
pub fn build_site(bundle: &ResearchBundle, opts: &AgentOptions) -> AgentSite {
    let mut site = AgentSite {
        project_name: bundle.project_name.clone(),
        ..Default::default()
    };

    // --- Flatten the area tree into frames ---------------------------------
    let mut frames: Vec<Frame> = Vec::new();
    match &bundle.area_tree {
        Some(tree) => {
            let root_rel = String::new();
            let root_idx = frames.len();
            frames.push(Frame {
                id: tree.root.id.clone(),
                name: tree.root.name.clone(),
                description: tree.root.description.clone(),
                root_paths: normalize_roots(&tree.root.root_paths),
                depth: 0,
                rel_dir: root_rel,
                children: Vec::new(),
                module_slots: Vec::new(),
                file_slots: Vec::new(),
            });
            add_children(&tree.root, root_idx, &mut frames);
        }
        None => {
            frames.push(Frame {
                id: "root".to_string(),
                name: bundle.project_name.clone(),
                description: "Project root".to_string(),
                root_paths: vec![".".to_string()],
                depth: 0,
                rel_dir: String::new(),
                children: Vec::new(),
                module_slots: Vec::new(),
                file_slots: Vec::new(),
            });
        }
    }

    // --- Collect file entries ---------------------------------------------
    let file_entries = collect_file_entries(bundle);

    // --- Assign modules to frames -----------------------------------------
    for owned in &bundle.modules {
        let frame = resolve_module_frame(&frames, owned);
        frames[frame].module_slots.push(owned.clone());
    }

    // --- Assign files to frames -------------------------------------------
    for mut entry in file_entries {
        let (frame, root) = match deepest_frame_for(&frames, &entry.path) {
            Some((idx, root)) => (idx, root.to_string()),
            None => (0, ".".to_string()),
        };
        let segments = mirror_segments(&entry.path, &root);
        if let Some(last) = segments.last() {
            entry.name = last.clone();
        }
        entry.mirror_dirs = segments[..segments.len().saturating_sub(1)].to_vec();
        entry.source_root = root;
        frames[frame].file_slots.push(entry);
    }

    // --- Emit root index, topics, and root diagrams ------------------------
    emit_root_and_topics(&mut site, bundle, &frames, opts);

    // --- Emit area tree pages recursively ---------------------------------
    let home = NavLink { label: "Home".to_string(), target: ROOT_INDEX.to_string() };
    emit_area(&mut site, &frames, 0, opts, std::slice::from_ref(&home));

    site
}

/// Recursively add child frames mirroring the area tree.
fn add_children(node: &AreaNode, parent_idx: usize, frames: &mut Vec<Frame>) {
    for child in &node.children {
        let parent_rel = frames[parent_idx].rel_dir.clone();
        let child_slug = unique_area_slug(&child.name, &parent_rel, frames);
        let rel_dir = if parent_rel.is_empty() {
            format!("tree/{}", child_slug)
        } else {
            format!("{}/{}", parent_rel, child_slug)
        };
        let idx = frames.len();
        frames.push(Frame {
            id: child.id.clone(),
            name: child.name.clone(),
            description: child.description.clone(),
            root_paths: normalize_roots(&child.root_paths),
            depth: frames[parent_idx].depth + 1,
            rel_dir,
            children: Vec::new(),
            module_slots: Vec::new(),
            file_slots: Vec::new(),
        });
        frames[parent_idx].children.push(idx);
        add_children(child, idx, frames);
    }
}

/// Allocate a collision-free, non-reserved slug for an area under its parent.
fn unique_area_slug(name: &str, parent_rel: &str, frames: &[Frame]) -> String {
    let base = slugify(name);
    let mut candidate = if RESERVED_AREA_SLUGS.contains(&base.as_str()) {
        format!("{}-area", base)
    } else {
        base
    };
    // De-dupe against siblings already created under this parent.
    let mut n = 1;
    while frames
        .iter()
        .any(|f| f.rel_dir == format!("{}/{}", parent_rel, candidate).trim_start_matches('/'))
    {
        n += 1;
        candidate = format!("{}-{}", slugify(name), n);
    }
    candidate
}

/// Resolve the frame index a module belongs to: prefer the explicit area id,
/// otherwise match its associated files against area roots, else the root frame.
fn resolve_module_frame(frames: &[Frame], owned: &OwnedKeyModule) -> usize {
    if let Some(area_id) = &owned.area_id
        && let Some(idx) = frames.iter().position(|f| &f.id == area_id)
    {
        return idx;
    }
    // Fall back to the deepest area owning any of the module's files.
    let mut best: Option<usize> = None;
    for path in &owned.report.associated_files {
        if let Some((idx, _)) = deepest_frame_for(frames, path) {
            best = Some(match best {
                Some(b) if frames[b].depth >= frames[idx].depth => b,
                _ => idx,
            });
        }
    }
    best.unwrap_or(0)
}

/// Find the deepest frame whose root paths contain `path`, returning both the
/// frame index and the matched root string (the basis for mirroring).
fn deepest_frame_for<'a>(frames: &'a [Frame], path: &str) -> Option<(usize, &'a str)> {
    let norm = path.replace('\\', "/");
    let mut best: Option<(usize, &str)> = None;
    for (idx, frame) in frames.iter().enumerate() {
        for root in &frame.root_paths {
            if path_under(&norm, root) {
                let score = root.len();
                if best.map(|(_, r)| score > r.len()).unwrap_or(true) {
                    best = Some((idx, root.as_str()));
                }
            }
        }
    }
    best
}

/// Split a project-relative source path into its segments relative to the
/// matched area root. The final segment is the basename.
fn mirror_segments(path: &str, matched_root: &str) -> Vec<String> {
    let norm = path.replace('\\', "/");
    let rel = if matched_root == "." {
        norm.trim_start_matches("./").to_string()
    } else {
        norm.strip_prefix(matched_root)
            .unwrap_or(&norm)
            .trim_start_matches('/')
            .to_string()
    };
    rel.split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .map(|s| s.to_string())
        .collect()
}

fn path_under(path: &str, root: &str) -> bool {
    if root == "." {
        return !path.is_empty();
    }
    path == root || path.starts_with(&format!("{}/", root))
}

/// Normalize area root paths to project-relative strings.
fn normalize_roots(raw: &[PathBuf]) -> Vec<String> {
    let roots: Vec<String> = raw
        .iter()
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if roots.is_empty() {
        vec![".".to_string()]
    } else {
        roots
    }
}

/// Build tightly-scoped file entries from file insights and module references.
fn collect_file_entries(bundle: &ResearchBundle) -> Vec<FileEntry> {
    let mut by_path: HashMap<String, FileEntry> = HashMap::new();

    for fi in &bundle.file_insights {
        let path = fi.file_path.to_string_lossy().replace('\\', "/");
        if path.is_empty() {
            continue;
        }
        let name = fi
            .name
            .clone()
            .split(['/', '\\'])
            .next_back()
            .unwrap_or(&fi.name)
            .to_string();
        by_path.insert(
            path.clone(),
            FileEntry {
                path,
                name,
                mirror_dirs: Vec::new(),
                source_root: String::new(),
                summary: fi.summary.clone(),
                responsibilities: fi.responsibilities.clone(),
                interfaces: fi.interfaces.iter().map(|i| i.name.clone()).collect(),
                dependencies: fi.dependencies.iter().map(|d| d.name.clone()).collect(),
            },
        );
    }

    // Ensure files referenced by modules get a page even without a file insight.
    for owned in &bundle.modules {
        for raw in &owned.report.associated_files {
            let path = raw.replace('\\', "/");
            if path.is_empty() || by_path.contains_key(&path) {
                continue;
            }
            // Skip directory-ish references (no file extension).
            let is_file = path.contains('.') && !path.ends_with('/');
            if !is_file {
                continue;
            }
            let name = path.split('/').next_back().unwrap_or(&path).to_string();
            by_path.insert(
                path.clone(),
                FileEntry {
                    path,
                    name,
                    mirror_dirs: Vec::new(),
                    source_root: String::new(),
                    summary: String::new(),
                    responsibilities: Vec::new(),
                    interfaces: Vec::new(),
                    dependencies: Vec::new(),
                },
            );
        }
    }

    by_path.into_values().collect()
}

/// Emit the root index page, the five topic pages, and root-level diagrams.
fn emit_root_and_topics(
    site: &mut AgentSite,
    bundle: &ResearchBundle,
    frames: &[Frame],
    opts: &AgentOptions,
) {
    // Root-level diagrams (context, area hierarchy, domain dependencies).
    let mut root_diagrams: Vec<NavLink> = Vec::new();
    if opts.diagrams {
        if let Some(report) = &bundle.system_context
            && let Some(body) = diagrams::context_diagram(report)
        {
            let rel = "diagrams/context.md".to_string();
            site.diagrams.push(DiagramFile {
                rel_path: rel.clone(),
                title: "System Context".to_string(),
                caption: "External systems and users around the project.".to_string(),
                body,
                owner: NavLink {
                    label: "index".to_string(),
                    target: ROOT_INDEX.to_string(),
                },
            });
            root_diagrams.push(NavLink {
                label: "System context".to_string(),
                target: rel,
            });
        }
        if let Some(tree) = &bundle.area_tree
            && let Some(body) = diagrams::area_hierarchy_diagram(tree, 2)
        {
            let rel = "diagrams/area-hierarchy.md".to_string();
            site.diagrams.push(DiagramFile {
                rel_path: rel.clone(),
                title: "Area Hierarchy".to_string(),
                caption: "Top-level decomposition of the project into areas.".to_string(),
                body,
                owner: NavLink {
                    label: "index".to_string(),
                    target: ROOT_INDEX.to_string(),
                },
            });
            root_diagrams.push(NavLink {
                label: "Area hierarchy".to_string(),
                target: rel,
            });
        }
        if let Some(dm) = &bundle.domain_modules
            && let Some(body) = diagrams::domain_dependencies_diagram(&dm.domain_relations)
        {
            let rel = "diagrams/domain-dependencies.md".to_string();
            site.diagrams.push(DiagramFile {
                rel_path: rel.clone(),
                title: "Domain Dependencies".to_string(),
                caption: "How functional domains depend on one another.".to_string(),
                body,
                owner: NavLink {
                    label: "index".to_string(),
                    target: ROOT_INDEX.to_string(),
                },
            });
            root_diagrams.push(NavLink {
                label: "Domain dependencies".to_string(),
                target: rel,
            });
        }
    }

    // Topic links for the index navigate block.
    let topic_links = vec![
        NavLink { label: "Overview".into(), target: "topics/overview.md".into() },
        NavLink { label: "Architecture".into(), target: "topics/architecture.md".into() },
        NavLink { label: "Workflow".into(), target: "topics/workflow.md".into() },
        NavLink { label: "Boundary & Interfaces".into(), target: "topics/boundary.md".into() },
        NavLink { label: "Database".into(), target: "topics/database.md".into() },
    ];

    // Top-level area links.
    let area_links: Vec<NavLink> = frames[0]
        .children
        .iter()
        .map(|&c| NavLink {
            label: frames[c].name.clone(),
            target: format!("{}/index.md", frames[c].rel_dir),
        })
        .collect();

    let mut nav_groups = vec![
        NavGroup { heading: "Topics".to_string(), links: topic_links },
    ];
    if !area_links.is_empty() {
        nav_groups.push(NavGroup { heading: "Areas".to_string(), links: area_links });
    }
    // Root-frame modules and files (those not claimed by any area) hang off the
    // root map directly so they are not orphaned.
    let home_crumb = NavLink { label: "Home".to_string(), target: ROOT_INDEX.to_string() };
    let root_module_links = emit_module_links(site, &frames[0], opts);
    if !root_module_links.is_empty() {
        nav_groups.push(NavGroup { heading: "Modules".to_string(), links: root_module_links });
    }
    let root_file_links = emit_file_tree(site, &frames[0], std::slice::from_ref(&home_crumb), opts);
    if !root_file_links.is_empty() {
        nav_groups.push(NavGroup { heading: "Files".to_string(), links: root_file_links });
    }
    if !root_diagrams.is_empty() {
        nav_groups.push(NavGroup { heading: "Diagrams".to_string(), links: root_diagrams });
    }

    site.pages.push(AgentPage {
        rel_path: ROOT_INDEX.to_string(),
        kind: PageKind::Index,
        title: if bundle.project_name.is_empty() {
            "Agent Content Map".to_string()
        } else {
            format!("{} — Agent Map", bundle.project_name)
        },
        summary: "Entry point for the agent-focused analysis. Follow a topic or area, drill "
            .to_string()
            + "down to modules and files, then jump straight to the source.",
        breadcrumbs: Vec::new(),
        nav_groups,
        source_paths: Vec::new(),
        diagram_links: Vec::new(),
        detail_blocks: Vec::new(),
        notes: vec![
            "Start at a Topic for a functional view or an Area for a structural view. Each page "
                .to_string()
                + "is short and links deeper until you reach the exact source file.",
        ],
    });

    emit_topics(site, bundle, opts);
}

/// Emit the five short topic pages (overview/architecture/workflow/boundary/database).
fn emit_topics(site: &mut AgentSite, bundle: &ResearchBundle, opts: &AgentOptions) {
    let crumbs = |title: &str| -> Vec<NavLink> {
        vec![
            NavLink { label: "Home".into(), target: ROOT_INDEX.into() },
            NavLink { label: title.into(), target: format!("topics/{}.md", title.to_lowercase()) },
        ]
    };

    // --- Overview ---
    let mut diagram_links = Vec::new();
    if opts.diagrams
        && site.diagrams.iter().any(|d| d.rel_path == "diagrams/context.md")
    {
        diagram_links.push(NavLink {
            label: "System context".into(),
            target: "diagrams/context.md".into(),
        });
    }
    let (summary, facts) = match &bundle.system_context {
        Some(sc) => {
            let mut facts = Vec::new();
            if !sc.business_value.is_empty() {
                facts.push(DetailBlock {
                    heading: "Business value".into(),
                    items: vec![cap(&sc.business_value, opts.max_desc_chars)],
                });
            }
            if !sc.external_systems.is_empty() {
                facts.push(DetailBlock {
                    heading: "External systems".into(),
                    items: sc
                        .external_systems
                        .iter()
                        .take(opts.max_list_items)
                        .map(|e| e.name.clone())
                        .collect(),
                });
            }
            (cap(&sc.project_description, opts.max_desc_chars), facts)
        }
        None => (String::new(), Vec::new()),
    };
    site.pages.push(AgentPage {
        rel_path: "topics/overview.md".into(),
        kind: PageKind::Topic,
        title: "Overview".into(),
        summary,
        breadcrumbs: crumbs("Overview"),
        nav_groups: Vec::new(),
        source_paths: Vec::new(),
        diagram_links,
        detail_blocks: facts,
        notes: Vec::new(),
    });

    // --- Architecture ---
    let mut diagram_links = Vec::new();
    let mut detail_blocks = Vec::new();
    if let Some(dm) = &bundle.domain_modules {
        if opts.diagrams {
            if let Some(body) = diagrams::domains_diagram(dm) {
                let rel = "topics/diagrams/architecture-domains.md";
                site.diagrams.push(DiagramFile {
                    rel_path: rel.into(),
                    title: "Domain Modules".into(),
                    caption: "Functional domains and their sub-modules.".into(),
                    body,
                    owner: NavLink { label: "Architecture".into(), target: "topics/architecture.md".into() },
                });
                diagram_links.push(NavLink { label: "Domain modules".into(), target: rel.into() });
            }
            if site.diagrams.iter().any(|d| d.rel_path == "diagrams/domain-dependencies.md") {
                diagram_links.push(NavLink {
                    label: "Domain dependencies".into(),
                    target: "diagrams/domain-dependencies.md".into(),
                });
            }
        }
        if !dm.architecture_summary.is_empty() {
            detail_blocks.push(DetailBlock {
                heading: "Summary".into(),
                items: vec![cap(&dm.architecture_summary, opts.max_desc_chars)],
            });
        }
    }
    site.pages.push(AgentPage {
        rel_path: "topics/architecture.md".into(),
        kind: PageKind::Topic,
        title: "Architecture".into(),
        summary: "Structural view: how the system is decomposed into domains and layers.".into(),
        breadcrumbs: crumbs("Architecture"),
        nav_groups: Vec::new(),
        source_paths: Vec::new(),
        diagram_links,
        detail_blocks,
        notes: Vec::new(),
    });

    // --- Workflow ---
    let mut diagram_links = Vec::new();
    let mut detail_blocks = Vec::new();
    if let Some(dm) = &bundle.domain_modules {
        // Workflow diagrams are nested by domain; the topic page links the
        // domain hubs rather than all 80+ flows.
        diagram_links.extend(emit_workflows(site, bundle, opts));
        if !dm.business_flows.is_empty() {
            detail_blocks.push(DetailBlock {
                heading: "Key flows".into(),
                items: dm
                    .business_flows
                    .iter()
                    .take(opts.max_list_items)
                    .map(|f| f.name.clone())
                    .collect(),
            });
        }
    }
    site.pages.push(AgentPage {
        rel_path: "topics/workflow.md".into(),
        kind: PageKind::Topic,
        title: "Workflow".into(),
        summary: "Behavioral view: the key end-to-end flows through the system.".into(),
        breadcrumbs: crumbs("Workflow"),
        nav_groups: Vec::new(),
        source_paths: Vec::new(),
        diagram_links,
        detail_blocks,
        notes: Vec::new(),
    });

    // --- Boundary ---
    let mut diagram_links = Vec::new();
    let mut detail_blocks = Vec::new();
    if let Some(b) = &bundle.boundary {
        if opts.diagrams
            && let Some(body) = diagrams::boundary_surfaces_diagram(b)
        {
            let rel = "topics/diagrams/boundary-surfaces.md";
            site.diagrams.push(DiagramFile {
                rel_path: rel.into(),
                title: "Boundary Surfaces".into(),
                caption: "CLI commands, API endpoints, and routes exposed by the system.".into(),
                body,
                owner: NavLink { label: "Boundary".into(), target: "topics/boundary.md".into() },
            });
            diagram_links.push(NavLink { label: "Boundary surfaces".into(), target: rel.into() });
        }
        if !b.api_boundaries.is_empty() {
            detail_blocks.push(DetailBlock {
                heading: "API surfaces".into(),
                items: b.api_boundaries.iter().take(opts.max_list_items)
                    .map(|a| format!("{} {}", a.method, a.endpoint)).collect(),
            });
        }
        if !b.cli_boundaries.is_empty() {
            detail_blocks.push(DetailBlock {
                heading: "CLI surfaces".into(),
                items: b.cli_boundaries.iter().take(opts.max_list_items)
                    .map(|c| c.command.clone()).collect(),
            });
        }
    }
    site.pages.push(AgentPage {
        rel_path: "topics/boundary.md".into(),
        kind: PageKind::Topic,
        title: "Boundary & Interfaces".into(),
        summary: "Everything the system exposes to the outside world.".into(),
        breadcrumbs: crumbs("Boundary"),
        nav_groups: Vec::new(),
        source_paths: Vec::new(),
        diagram_links,
        detail_blocks,
        notes: Vec::new(),
    });

    // --- Database ---
    let mut diagram_links = Vec::new();
    let mut detail_blocks = Vec::new();
    if let Some(db) = &bundle.database {
        if opts.diagrams {
            if let Some(body) = diagrams::database_er_diagram(db) {
                let rel = "topics/diagrams/database-er.md";
                site.diagrams.push(DiagramFile {
                    rel_path: rel.into(),
                    title: "Database ER".into(),
                    caption: "Table relationships (foreign keys and references).".into(),
                    body,
                    owner: NavLink { label: "Database".into(), target: "topics/database.md".into() },
                });
                diagram_links.push(NavLink { label: "Entity relationships".into(), target: rel.into() });
            }
            if let Some(body) = diagrams::database_flows_diagram(db) {
                let rel = "topics/diagrams/database-flows.md";
                site.diagrams.push(DiagramFile {
                    rel_path: rel.into(),
                    title: "Database Flows".into(),
                    caption: "How data moves between tables and systems.".into(),
                    body,
                    owner: NavLink { label: "Database".into(), target: "topics/database.md".into() },
                });
                diagram_links.push(NavLink { label: "Data flows".into(), target: rel.into() });
            }
        }
        if !db.tables.is_empty() {
            detail_blocks.push(DetailBlock {
                heading: "Tables".into(),
                items: db.tables.iter().take(opts.max_list_items)
                    .map(|t| format!("{}.{}", t.schema, t.name)).collect(),
            });
        }
    }
    site.pages.push(AgentPage {
        rel_path: "topics/database.md".into(),
        kind: PageKind::Topic,
        title: "Database".into(),
        summary: "Persistent data model, tables, and how data moves.".into(),
        breadcrumbs: crumbs("Database"),
        nav_groups: Vec::new(),
        source_paths: Vec::new(),
        diagram_links,
        detail_blocks,
        notes: Vec::new(),
    });
}

/// Recursively emit an area page plus its modules, files, and diagram.
///
/// The root frame (idx 0) is emitted by `emit_root_and_topics`; here it only
/// contributes its children's breadcrumbs.
fn emit_area(
    site: &mut AgentSite,
    frames: &[Frame],
    idx: usize,
    opts: &AgentOptions,
    ancestor_crumbs: &[NavLink],
) {
    let frame = &frames[idx];
    let index_rel = if frame.rel_dir.is_empty() {
        ROOT_INDEX.to_string()
    } else {
        format!("{}/index.md", frame.rel_dir)
    };

    // Breadcrumbs: ancestors + this area.
    let mut breadcrumbs: Vec<NavLink> = ancestor_crumbs.to_vec();
    let mut child_crumbs = ancestor_crumbs.to_vec();
    if idx != 0 {
        let own = NavLink { label: frame.name.clone(), target: index_rel.clone() };
        breadcrumbs.push(own.clone());
        child_crumbs.push(own);
    }

    if idx != 0 {
        // Area structure diagram (children + modules).
        let mut diagram_links = Vec::new();
        if opts.diagrams {
            let module_names: Vec<String> = frame
                .module_slots
                .iter()
                .map(|m| m.report.domain_name.clone())
                .collect();
            if let Some(body) = diagrams::area_structure_diagram(&AreaNode {
                id: frame.id.clone(),
                name: frame.name.clone(),
                description: frame.description.clone(),
                root_paths: frame.root_paths.iter().map(PathBuf::from).collect(),
                children: Vec::new(),
                dossier_paths: Vec::new(),
            }, &module_names)
            {
                let rel = format!("{}/diagrams/{}.md", frame.rel_dir, slugify(&frame.name));
                site.diagrams.push(DiagramFile {
                    rel_path: rel.clone(),
                    title: format!("{} — Structure", frame.name),
                    caption: "Sub-areas and modules within this area.".into(),
                    body,
                    owner: NavLink { label: frame.name.clone(), target: index_rel.clone() },
                });
                diagram_links.push(NavLink { label: "Structure".into(), target: rel });
            }
        }

        // Sub-area nav group.
        let mut nav_groups: Vec<NavGroup> = Vec::new();
        let child_links: Vec<NavLink> = frame
            .children
            .iter()
            .map(|&c| NavLink {
                label: frames[c].name.clone(),
                target: format!("{}/index.md", frames[c].rel_dir),
            })
            .collect();
        if !child_links.is_empty() {
            nav_groups.push(NavGroup { heading: "Sub-areas".into(), links: child_links });
        }

        // Module folder pages + nav.
        let module_links = emit_module_links(site, frame, opts);
        if !module_links.is_empty() {
            nav_groups.push(NavGroup { heading: "Modules".into(), links: module_links });
        }

        // Nested file tree + top-level nav.
        let file_links = emit_file_tree(site, frame, &breadcrumbs, opts);
        if !file_links.is_empty() {
            nav_groups.push(NavGroup { heading: "Files".into(), links: file_links });
        }

        site.pages.push(AgentPage {
            rel_path: index_rel.clone(),
            kind: PageKind::Area,
            title: frame.name.clone(),
            summary: cap(&frame.description, opts.max_desc_chars),
            breadcrumbs,
            nav_groups,
            source_paths: frame.root_paths.clone(),
            diagram_links,
            detail_blocks: Vec::new(),
            notes: Vec::new(),
        });
    }

    for &c in &frame.children {
        emit_area(site, frames, c, opts, &child_crumbs);
    }
}

/// Emit every module of a frame into its own folder and return nav links,
/// sorted by display name.
fn emit_module_links(site: &mut AgentSite, frame: &Frame, opts: &AgentOptions) -> Vec<NavLink> {
    let mut reg = NameRegistry::new();
    let mut links: Vec<NavLink> = frame
        .module_slots
        .iter()
        .map(|owned| {
            let rel = emit_module(site, frame, owned, opts, &mut reg);
            NavLink {
                label: if owned.report.module_name.is_empty() {
                    owned.report.domain_name.clone()
                } else {
                    owned.report.module_name.clone()
                },
                target: rel,
            }
        })
        .collect();
    links.sort_by_key(|l| l.label.to_lowercase());
    links
}

/// Emit one module folder (`modules/<slug>/index.md` plus its flowchart and
/// sequence diagrams as siblings). Returns the module page path.
fn emit_module(
    site: &mut AgentSite,
    frame: &Frame,
    owned: &OwnedKeyModule,
    opts: &AgentOptions,
    reg: &mut NameRegistry,
) -> String {
    let report = &owned.report;
    let slug = reg.allocate(&slugify(if report.module_name.is_empty() {
        &report.domain_name
    } else {
        &report.module_name
    }));
    let rel_dir = if frame.rel_dir.is_empty() {
        format!("modules/{}", slug)
    } else {
        format!("{}/modules/{}", frame.rel_dir, slug)
    };
    let rel = format!("{}/index.md", rel_dir);

    let mut diagram_links = Vec::new();
    if opts.diagrams {
        if !report.flowchart_mermaid.trim().is_empty() {
            let drel = format!("{}/flowchart.md", rel_dir);
            site.diagrams.push(DiagramFile {
                rel_path: drel.clone(),
                title: format!("{} — Flow", report.domain_name),
                caption: "Module flowchart.".into(),
                body: report.flowchart_mermaid.clone(),
                owner: NavLink { label: report.module_name.clone(), target: rel.clone() },
            });
            diagram_links.push(NavLink { label: "Flowchart".into(), target: drel });
        }
        if !report.sequence_diagram_mermaid.trim().is_empty() {
            let drel = format!("{}/sequence.md", rel_dir);
            site.diagrams.push(DiagramFile {
                rel_path: drel.clone(),
                title: format!("{} — Sequence", report.domain_name),
                caption: "Module interaction sequence.".into(),
                body: report.sequence_diagram_mermaid.clone(),
                owner: NavLink { label: report.module_name.clone(), target: rel.clone() },
            });
            diagram_links.push(NavLink { label: "Sequence".into(), target: drel });
        }
    }

    let mut detail_blocks = Vec::new();
    if !report.interaction.is_empty() {
        detail_blocks.push(DetailBlock {
            heading: "Interaction".into(),
            items: vec![cap(&report.interaction, opts.max_desc_chars)],
        });
    }

    // Source files for this module.
    let source_paths: Vec<String> = report
        .associated_files
        .iter()
        .map(|p| p.replace('\\', "/"))
        .collect();

    site.pages.push(AgentPage {
        rel_path: rel.clone(),
        kind: PageKind::Module,
        title: if report.module_name.is_empty() {
            report.domain_name.clone()
        } else {
            report.module_name.clone()
        },
        summary: cap(&report.module_description, opts.max_desc_chars),
        breadcrumbs: module_crumbs(frame, &report.module_name),
        nav_groups: Vec::new(),
        source_paths,
        diagram_links,
        detail_blocks,
        notes: vec![cap(&report.implementation, opts.max_desc_chars)]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect(),
    });

    rel
}

fn module_crumbs(frame: &Frame, module_name: &str) -> Vec<NavLink> {
    let mut crumbs = vec![NavLink { label: "Home".into(), target: ROOT_INDEX.into() }];
    if !frame.rel_dir.is_empty() {
        crumbs.push(NavLink {
            label: frame.name.clone(),
            target: format!("{}/index.md", frame.rel_dir),
        });
    }
    crumbs.push(NavLink {
        label: if module_name.is_empty() { "Module".into() } else { module_name.into() },
        target: String::new(),
    });
    crumbs
}

/// A directory node in the mirrored source tree under an area's `files/`.
#[derive(Default)]
struct DirTrie {
    dirs: std::collections::BTreeMap<String, DirTrie>,
    files: Vec<FileEntry>,
}

/// Build the mirrored file tree for a frame and emit its hub + leaf pages,
/// returning the top-level nav links (folders first, then files).
fn emit_file_tree(
    site: &mut AgentSite,
    frame: &Frame,
    base_crumbs: &[NavLink],
    opts: &AgentOptions,
) -> Vec<NavLink> {
    if frame.file_slots.is_empty() {
        return Vec::new();
    }
    let files_rel = if frame.rel_dir.is_empty() {
        "files".to_string()
    } else {
        format!("{}/files", frame.rel_dir)
    };
    let mut root = DirTrie::default();
    for entry in &frame.file_slots {
        let mut node = &mut root;
        for dir in &entry.mirror_dirs {
            node = node.dirs.entry(dir.clone()).or_default();
        }
        node.files.push(entry.clone());
    }
    let mut top = Vec::new();
    emit_dir(site, &root, &files_rel, "files", &[], base_crumbs, true, opts, &mut top);
    top
}

/// Recursively emit one mirrored directory: its hub page, child directories,
/// and file pages. `prefix` holds the raw source-dir segments from the tree
/// root down to this node (used to reconstruct the hub's Source link).
#[allow(clippy::too_many_arguments)]
fn emit_dir(
    site: &mut AgentSite,
    node: &DirTrie,
    cur_rel: &str,
    display: &str,
    prefix: &[String],
    ancestor_crumbs: &[NavLink],
    is_root: bool,
    opts: &AgentOptions,
    top_out: &mut Vec<NavLink>,
) {
    let mut reg = NameRegistry::new();
    reg.reserve("index.md");

    // Reserve directory names first so a file leaf never shadows a folder.
    let mut child_dirs: Vec<(String, String, &DirTrie)> = Vec::new();
    for (raw, child) in &node.dirs {
        let alloc = reg.allocate(&clamp_component(raw));
        child_dirs.push((raw.clone(), alloc, child));
    }

    // Allocate verbatim leaf names.
    let mut sorted = node.files.clone();
    sorted.sort_by_key(|e| e.name.to_lowercase());
    let mut leaves: Vec<(String, FileEntry)> = sorted
        .into_iter()
        .map(|entry| {
            let leaf = reg.allocate(&verbatim_leaf(&entry.name));
            (leaf, entry)
        })
        .collect();
    leaves.sort_by_key(|(name, _)| name.to_lowercase());

    let hub_rel = format!("{}/index.md", cur_rel);
    let mut crumbs = ancestor_crumbs.to_vec();
    crumbs.push(NavLink { label: display.to_string(), target: hub_rel.clone() });

    let folder_links: Vec<NavLink> = child_dirs
        .iter()
        .map(|(_, alloc, _)| NavLink {
            label: alloc.clone(),
            target: format!("{}/{}/index.md", cur_rel, alloc),
        })
        .collect();
    let leaf_links: Vec<NavLink> = leaves
        .iter()
        .map(|(leaf, entry)| NavLink {
            label: entry.name.clone(),
            target: format!("{}/{}", cur_rel, leaf),
        })
        .collect();

    // Hub Source link: the real folder = first descendant file's matched root
    // plus this node's raw source-dir segments. The project root (".") has no
    // meaningful link, so it is omitted.
    let source_paths = match first_file(node) {
        Some(f) => {
            let mut segs = vec![f.source_root.clone()];
            segs.extend(prefix.iter().cloned());
            let joined = segs.join("/");
            if joined.trim_matches('/') == "." {
                Vec::new()
            } else {
                vec![joined]
            }
        }
        None => Vec::new(),
    };

    let mut nav_groups = Vec::new();
    if !folder_links.is_empty() {
        nav_groups.push(NavGroup { heading: "Folders".into(), links: folder_links.clone() });
    }
    if !leaf_links.is_empty() {
        nav_groups.push(NavGroup { heading: "Files".into(), links: leaf_links.clone() });
    }

    site.pages.push(AgentPage {
        rel_path: hub_rel,
        kind: PageKind::Dir,
        title: display.to_string(),
        summary: format!(
            "{} file(s) across {} folder(s).",
            count_files(node),
            count_dirs(node)
        ),
        breadcrumbs: crumbs.clone(),
        nav_groups,
        source_paths,
        diagram_links: Vec::new(),
        detail_blocks: Vec::new(),
        notes: Vec::new(),
    });

    if is_root {
        top_out.extend(folder_links);
        top_out.extend(leaf_links);
    }

    for (raw, alloc, child) in &child_dirs {
        let mut child_prefix = prefix.to_vec();
        child_prefix.push(raw.clone());
        emit_dir(
            site,
            child,
            &format!("{}/{}", cur_rel, alloc),
            alloc,
            &child_prefix,
            &crumbs,
            false,
            opts,
            top_out,
        );
    }

    for (leaf, entry) in &leaves {
        let rel = format!("{}/{}", cur_rel, leaf);
        let mut leaf_crumbs = crumbs.clone();
        leaf_crumbs.push(NavLink { label: entry.name.clone(), target: rel.clone() });
        emit_leaf(site, entry, rel, leaf_crumbs, opts);
    }
}

/// First file in the subtree (DFS) — used to derive a directory's Source link.
/// Picks the lexicographically smallest path so the choice is deterministic
/// regardless of the (HashMap-derived) insertion order of `file_slots`.
fn first_file(node: &DirTrie) -> Option<&FileEntry> {
    if let Some(f) = node.files.iter().min_by(|a, b| a.path.cmp(&b.path)) {
        return Some(f);
    }
    node.dirs.values().find_map(first_file)
}

/// Total files in a subtree.
fn count_files(node: &DirTrie) -> usize {
    node.files.len() + node.dirs.values().map(count_files).sum::<usize>()
}

/// Total (recursive) directories in a subtree.
fn count_dirs(node: &DirTrie) -> usize {
    node.dirs.len() + node.dirs.values().map(count_dirs).sum::<usize>()
}

/// Emit one tightly-scoped per-file page at an already-computed path.
fn emit_leaf(
    site: &mut AgentSite,
    entry: &FileEntry,
    rel: String,
    breadcrumbs: Vec<NavLink>,
    opts: &AgentOptions,
) {
    let mut detail_blocks = Vec::new();
    if !entry.responsibilities.is_empty() {
        detail_blocks.push(DetailBlock {
            heading: "Responsibilities".into(),
            items: entry.responsibilities.iter().take(opts.max_list_items).cloned().collect(),
        });
    }
    if !entry.interfaces.is_empty() {
        detail_blocks.push(DetailBlock {
            heading: "Key interfaces".into(),
            items: entry.interfaces.iter().take(opts.max_list_items).cloned().collect(),
        });
    }
    if !entry.dependencies.is_empty() {
        detail_blocks.push(DetailBlock {
            heading: "Depends on".into(),
            items: entry.dependencies.iter().take(opts.max_list_items).cloned().collect(),
        });
    }

    site.pages.push(AgentPage {
        rel_path: rel,
        kind: PageKind::File,
        title: entry.name.clone(),
        summary: cap(&entry.summary, opts.max_desc_chars),
        breadcrumbs,
        nav_groups: Vec::new(),
        source_paths: vec![entry.path.clone()],
        diagram_links: Vec::new(),
        detail_blocks,
        notes: Vec::new(),
    });
}

/// Emit workflow diagrams nested by domain under `topics/diagrams/workflows/`
/// and return the domain hub links for the Workflow topic page.
fn emit_workflows(
    site: &mut AgentSite,
    bundle: &ResearchBundle,
    opts: &AgentOptions,
) -> Vec<NavLink> {
    if !opts.diagrams {
        return Vec::new();
    }
    let Some(dm) = &bundle.domain_modules else {
        return Vec::new();
    };
    if dm.business_flows.is_empty() {
        return Vec::new();
    }

    // Known domain names, keyed by slug, for canonical bucketing.
    let known: Vec<(String, String)> = dm
        .domain_modules
        .iter()
        .map(|d| (slugify(&d.name), d.name.clone()))
        .collect();

    // Bucket flows by the domain of their first non-empty step.
    let mut buckets: std::collections::BTreeMap<String, (String, Vec<&BusinessFlow>)> =
        std::collections::BTreeMap::new();
    for flow in &dm.business_flows {
        let domain = flow
            .steps
            .iter()
            .map(|s| s.domain_module.trim())
            .find(|s| !s.is_empty());
        let (bucket, display) = match domain {
            Some(d) => {
                let ds = slugify(d);
                match known.iter().find(|(slug, _)| *slug == ds) {
                    Some((slug, name)) => (slug.clone(), name.clone()),
                    None => ("general".to_string(), "General".to_string()),
                }
            }
            None => ("general".to_string(), "General".to_string()),
        };
        buckets
            .entry(bucket)
            .or_insert_with(|| (display, Vec::new()))
            .1
            .push(flow);
    }

    let workflows_root = "topics/diagrams/workflows";
    let root_hub_rel = format!("{}/index.md", workflows_root);
    let mut bucket_links = Vec::new();

    for (bucket, (display, flows)) in &buckets {
        let bucket_rel = format!("{}/{}", workflows_root, bucket);
        let bucket_hub = format!("{}/index.md", bucket_rel);

        let mut reg = NameRegistry::new();
        reg.reserve("index.md");
        let mut sorted: Vec<&BusinessFlow> = flows.clone();
        sorted.sort_by_key(|f| f.name.to_lowercase());

        let mut flow_links = Vec::new();
        for flow in sorted {
            let Some((_, body)) =
                diagrams::workflow_diagrams(std::slice::from_ref(flow)).into_iter().next()
            else {
                continue;
            };
            let fslug = reg.allocate(&slugify(&flow.name));
            let rel = format!("{}/{}.md", bucket_rel, fslug);
            site.diagrams.push(DiagramFile {
                rel_path: rel.clone(),
                title: format!("Workflow — {}", flow.name),
                caption: cap(&flow.description, opts.max_desc_chars),
                body,
                owner: NavLink { label: "Workflow".into(), target: "topics/workflow.md".into() },
            });
            flow_links.push(NavLink { label: flow.name.clone(), target: rel });
        }

        site.pages.push(AgentPage {
            rel_path: bucket_hub.clone(),
            kind: PageKind::Dir,
            title: display.clone(),
            summary: format!("{} workflow(s).", flow_links.len()),
            breadcrumbs: vec![
                NavLink { label: "Home".into(), target: ROOT_INDEX.into() },
                NavLink { label: "Workflow".into(), target: "topics/workflow.md".into() },
                NavLink { label: "Workflows".into(), target: root_hub_rel.clone() },
                NavLink { label: display.clone(), target: bucket_hub.clone() },
            ],
            nav_groups: vec![NavGroup { heading: "Workflows".into(), links: flow_links }],
            source_paths: Vec::new(),
            diagram_links: Vec::new(),
            detail_blocks: Vec::new(),
            notes: Vec::new(),
        });

        bucket_links.push(NavLink { label: display.clone(), target: bucket_hub });
    }

    site.pages.push(AgentPage {
        rel_path: root_hub_rel.clone(),
        kind: PageKind::Dir,
        title: "Workflows".into(),
        summary: format!("{} domain(s).", bucket_links.len()),
        breadcrumbs: vec![
            NavLink { label: "Home".into(), target: ROOT_INDEX.into() },
            NavLink { label: "Workflow".into(), target: "topics/workflow.md".into() },
            NavLink { label: "Workflows".into(), target: root_hub_rel },
        ],
        nav_groups: vec![NavGroup { heading: "Domains".into(), links: bucket_links.clone() }],
        source_paths: Vec::new(),
        diagram_links: Vec::new(),
        detail_blocks: Vec::new(),
        notes: Vec::new(),
    });

    bucket_links
}

/// Cap prose length at a word boundary, appending an ellipsis when truncated.
pub fn cap(text: &str, max: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max {
        return trimmed.to_string();
    }
    let mut out = String::new();
    for (i, ch) in trimmed.chars().enumerate() {
        if i >= max {
            break;
        }
        out.push(ch);
    }
    // Trim back to the last space to avoid mid-word cuts.
    if let Some(pos) = out.rfind(' ') {
        out.truncate(pos);
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::research::area_tree::AreaNode as AN;

    fn tree_two_levels() -> AreaTree {
        AreaTree {
            root: AN {
                id: "root".into(),
                name: "proj".into(),
                description: "root".into(),
                root_paths: vec![PathBuf::from(".")],
                children: vec![AN {
                    id: "core".into(),
                    name: "Core".into(),
                    description: "core area".into(),
                    root_paths: vec![PathBuf::from("core")],
                    children: vec![AN {
                        id: "cache".into(),
                        name: "Cache".into(),
                        description: "cache area".into(),
                        root_paths: vec![PathBuf::from("core/cache")],
                        children: Vec::new(),
                        dossier_paths: Vec::new(),
                    }],
                    dossier_paths: Vec::new(),
                }],
                dossier_paths: Vec::new(),
            },
        }
    }

    fn module(name: &str, files: &[&str]) -> OwnedKeyModule {
        OwnedKeyModule {
            area_id: None,
            report: KeyModuleReport {
                domain_name: name.into(),
                module_name: name.into(),
                module_description: "desc".into(),
                associated_files: files.iter().map(|s| s.to_string()).collect(),
                ..Default::default()
            },
        }
    }

    #[test]
    fn build_site_emits_index_topics_and_recursive_areas() {
        let bundle = ResearchBundle {
            project_name: "proj".into(),
            area_tree: Some(tree_two_levels()),
            modules: vec![module("Cache", &["core/cache/Cache.rs"])],
            ..Default::default()
        };
        let site = build_site(&bundle, &AgentOptions::default());
        let paths: Vec<&str> = site.pages.iter().map(|p| p.rel_path.as_str()).collect();
        assert!(paths.contains(&ROOT_INDEX));
        assert!(paths.contains(&"topics/overview.md"));
        assert!(paths.contains(&"tree/core/index.md"));
        assert!(paths.contains(&"tree/core/cache/index.md"));
    }

    #[test]
    fn module_file_pages_are_tightly_scoped() {
        let bundle = ResearchBundle {
            project_name: "proj".into(),
            area_tree: Some(tree_two_levels()),
            modules: vec![module("Cache", &["core/cache/Cache.rs"])],
            file_insights: vec![FileInsight {
                name: "Cache.rs".into(),
                file_path: PathBuf::from("core/cache/Cache.rs"),
                summary: "cache manager".into(),
                responsibilities: vec!["store".into(), "evict".into()],
                ..Default::default()
            }],
            ..Default::default()
        };
        let site = build_site(&bundle, &AgentOptions::default());
        let paths: Vec<&str> = site.pages.iter().map(|p| p.rel_path.as_str()).collect();
        // Module is now a self-contained folder.
        assert!(paths.contains(&"tree/core/cache/modules/cache/index.md"), "paths: {:?}", paths);
        // File page mirrors the source tree (root stripped: core/cache/Cache.rs).
        assert!(paths.contains(&"tree/core/cache/files/Cache.rs.md"), "paths: {:?}", paths);
        assert!(paths.contains(&"tree/core/cache/files/index.md"), "paths: {:?}", paths);
        // File page is emitted and grounded in its source path.
        let file_page = site.pages.iter().find(|p| p.kind == PageKind::File).unwrap();
        assert_eq!(file_page.source_paths, vec!["core/cache/Cache.rs".to_string()]);
    }

    #[test]
    fn file_pages_mirror_source_subdirectories() {
        let bundle = ResearchBundle {
            project_name: "proj".into(),
            area_tree: Some(tree_two_levels()),
            file_insights: vec![FileInsight {
                name: "Store.rs".into(),
                file_path: PathBuf::from("core/cache/registry/Store.rs"),
                summary: "registry store".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let site = build_site(&bundle, &AgentOptions::default());
        let paths: Vec<&str> = site.pages.iter().map(|p| p.rel_path.as_str()).collect();
        // Mirror relative to the deepest matched root (core/cache).
        assert!(paths.contains(&"tree/core/cache/files/registry/Store.rs.md"), "paths: {:?}", paths);
        assert!(paths.contains(&"tree/core/cache/files/registry/index.md"), "paths: {:?}", paths);
    }

    #[test]
    fn root_files_get_hub_and_index_link() {
        let bundle = ResearchBundle {
            project_name: "proj".into(),
            area_tree: Some(tree_two_levels()),
            file_insights: vec![FileInsight {
                name: "main.rs".into(),
                file_path: PathBuf::from("main.rs"),
                summary: "entry".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let site = build_site(&bundle, &AgentOptions::default());
        let paths: Vec<&str> = site.pages.iter().map(|p| p.rel_path.as_str()).collect();
        assert!(paths.contains(&"files/main.rs.md"), "paths: {:?}", paths);
        assert!(paths.contains(&"files/index.md"), "paths: {:?}", paths);
        // Root index links the root file so it is not orphaned.
        let index = site.pages.iter().find(|p| p.rel_path == ROOT_INDEX).unwrap();
        let files_group = index.nav_groups.iter().find(|g| g.heading == "Files").unwrap();
        assert!(files_group.links.iter().any(|l| l.target == "files/main.rs.md"));
    }

    #[test]
    fn workflow_diagrams_group_by_domain() {
        use crate::generator::research::types::{
            BusinessFlow, BusinessFlowStep, DomainModule,
        };
        let bundle = ResearchBundle {
            project_name: "proj".into(),
            domain_modules: Some(DomainModulesReport {
                domain_modules: vec![DomainModule {
                    name: "Identity, Auth & Access".into(),
                    ..Default::default()
                }],
                business_flows: vec![
                    BusinessFlow {
                        name: "Login Flow".into(),
                        description: "sign in".into(),
                        steps: vec![BusinessFlowStep {
                            step: 1,
                            domain_module: "Identity, Auth & Access".into(),
                            operation: "auth".into(),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    BusinessFlow {
                        name: "Orphan Flow".into(),
                        description: "unmatched".into(),
                        steps: vec![BusinessFlowStep {
                            step: 1,
                            domain_module: "Nonexistent Domain".into(),
                            operation: "do".into(),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        let site = build_site(&bundle, &AgentOptions::default());
        let paths: Vec<&str> = site.pages.iter().map(|p| p.rel_path.as_str()).collect();
        let dpaths: Vec<&str> = site.diagrams.iter().map(|d| d.rel_path.as_str()).collect();
        assert!(dpaths.contains(&"topics/diagrams/workflows/identity-auth-access/login-flow.md"), "diagrams: {:?}", dpaths);
        // Unmatched domain falls back to general/.
        assert!(dpaths.contains(&"topics/diagrams/workflows/general/orphan-flow.md"), "diagrams: {:?}", dpaths);
        assert!(paths.contains(&"topics/diagrams/workflows/index.md"), "paths: {:?}", paths);
        assert!(paths.contains(&"topics/diagrams/workflows/identity-auth-access/index.md"), "paths: {:?}", paths);
    }

    #[test]
    fn reserved_area_slugs_are_avoided() {
        let mut tree = tree_two_levels();
        tree.root.children.push(AN {
            id: "mods".into(),
            name: "Modules".into(),
            description: "a".into(),
            root_paths: vec![PathBuf::from("mods")],
            children: Vec::new(),
            dossier_paths: Vec::new(),
        });
        let bundle = ResearchBundle { project_name: "proj".into(), area_tree: Some(tree), ..Default::default() };
        let site = build_site(&bundle, &AgentOptions::default());
        let paths: Vec<&str> = site.pages.iter().map(|p| p.rel_path.as_str()).collect();
        assert!(paths.contains(&"tree/modules-area/index.md"), "paths: {:?}", paths);
    }

    #[test]
    fn no_diagrams_when_disabled() {
        let bundle = ResearchBundle { project_name: "proj".into(), area_tree: Some(tree_two_levels()), ..Default::default() };
        let opts = AgentOptions { diagrams: false, ..Default::default() };
        let site = build_site(&bundle, &opts);
        assert!(site.diagrams.is_empty());
    }

    #[test]
    fn cap_truncates_at_word_boundary() {
        let s = cap("the quick brown fox jumps", 12);
        assert!(s.ends_with('…'));
        assert!(s.chars().count() <= 13);
    }
}
