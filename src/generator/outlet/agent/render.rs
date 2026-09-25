//! Rendering of agent pages and diagram files into short Markdown.
//!
//! Hrefs are resolved here: page-to-page and page-to-diagram links are computed
//! between agent-root-relative paths (both live under `.agent-content/`), while
//! source links are computed from a page's absolute directory back into the
//! project tree. When a relative path cannot be formed (cross-root), source links
//! degrade to inline code spans so the text is never silently dropped.

use std::path::Path;

use crate::generator::outlet::agent::diagrams::render_diagram_markdown;
use crate::generator::outlet::agent::links::{relative_href, source_href};
use crate::generator::outlet::agent::model::{AgentPage, DiagramFile, NavLink, PageKind};

/// Absolute roots needed to resolve links while rendering.
pub struct RenderContext<'a> {
    /// Absolute project root — source paths are relative to this.
    pub project_root: &'a Path,
    /// Absolute `.agent-content` directory — page/diagram paths are relative to this.
    pub agent_root: &'a Path,
}

/// Render a page into Markdown. Kept intentionally short: a title, a capped
/// summary, optional notes, a location breadcrumb, and compact Diagrams /
/// Navigate / Source / detail blocks.
pub fn render_page(page: &AgentPage, ctx: &RenderContext) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {}\n\n", page.title));
    out.push_str(&format!("*{} · agent map*\n\n", kind_label(page.kind)));

    if !page.summary.trim().is_empty() {
        out.push_str(page.summary.trim());
        out.push_str("\n\n");
    }

    for note in &page.notes {
        if !note.trim().is_empty() {
            out.push_str(note.trim());
            out.push_str("\n\n");
        }
    }

    // Location breadcrumb — every ancestor is a link; the current page is plain.
    if !page.breadcrumbs.is_empty() {
        out.push_str(&format!("**Location:** {}\n\n", render_breadcrumb(page, ctx)));
    }

    // Diagrams.
    if !page.diagram_links.is_empty() {
        out.push_str("## Diagrams\n\n");
        for link in &page.diagram_links {
            out.push_str(&format!(
                "- [{}]({})\n",
                link.label,
                inner_href(&page.rel_path, &link.target)
            ));
        }
        out.push('\n');
    }

    // Navigate.
    if !page.nav_groups.is_empty() {
        out.push_str("## Navigate\n\n");
        for group in &page.nav_groups {
            if group.links.is_empty() {
                continue;
            }
            let links: Vec<String> = group
                .links
                .iter()
                .map(|l| format!("[{}]({})", l.label, inner_href(&page.rel_path, &l.target)))
                .collect();
            out.push_str(&format!("- **{}:** {}\n", group.heading, links.join(", ")));
        }
        out.push('\n');
    }

    // Source links (folders and files into the codebase).
    if !page.source_paths.is_empty() {
        out.push_str("## Source\n\n");
        for src in &page.source_paths {
            out.push_str(&format!("- {}\n", source_item(page, src, ctx)));
        }
        out.push('\n');
    }

    // Detail blocks (responsibilities, interfaces, ...).
    for block in &page.detail_blocks {
        if block.items.is_empty() {
            continue;
        }
        out.push_str(&format!("## {}\n\n", block.heading));
        for item in &block.items {
            out.push_str(&format!("- {}\n", item.trim()));
        }
        out.push('\n');
    }

    out.trim_end().to_string();
    out.push('\n');
    out
}

/// Render a diagram file (title + caption + mermaid fence + back-link).
pub fn render_diagram(diagram: &DiagramFile, _ctx: &RenderContext) -> String {
    let back_href = inner_href(&diagram.rel_path, &diagram.owner.target);
    render_diagram_markdown(
        &diagram.title,
        Some(&diagram.caption),
        &diagram.body,
        Some((&back_href, &diagram.owner.label)),
    )
}

/// Render the `Home › Area › Module` trail, linking every non-current crumb.
fn render_breadcrumb(page: &AgentPage, ctx: &RenderContext) -> String {
    let _ = ctx; // breadcrumbs use inner (agent-root-relative) hrefs
    let parts: Vec<String> = page
        .breadcrumbs
        .iter()
        .map(|crumb| {
            if is_current(crumb, page) {
                format!("**{}**", crumb.label)
            } else {
                format!("[{}]({})", crumb.label, inner_href(&page.rel_path, &crumb.target))
            }
        })
        .collect();
    parts.join(" › ")
}

fn is_current(crumb: &NavLink, page: &AgentPage) -> bool {
    crumb.target.is_empty() || crumb.target == page.rel_path
}

/// A single Source bullet: a link when a relative path exists, else a code span.
fn source_item(page: &AgentPage, src: &str, ctx: &RenderContext) -> String {
    let page_dir = agent_page_dir(&page.rel_path, ctx.agent_root);
    match source_href(ctx.project_root, &page_dir, src) {
        Some(href) => format!("[`{}`]({})", src, href),
        None => format!("`{}`", src),
    }
}

/// Absolute directory containing a page (agent_root + page's parent dir).
fn agent_page_dir(rel_path: &str, agent_root: &Path) -> std::path::PathBuf {
    agent_root.join(parent_dir(rel_path))
}

/// Relative href between two agent-root-relative paths (`from` is a document).
fn inner_href(from_rel_path: &str, to_rel_path: &str) -> String {
    if to_rel_path.is_empty() {
        return "#".to_string();
    }
    let from_dir = parent_dir(from_rel_path);
    match relative_href(Path::new(from_dir), Path::new(to_rel_path)) {
        Some(href) => href,
        None => to_rel_path.to_string(),
    }
}

/// Human-readable page-type label used in the page subtitle.
fn kind_label(kind: PageKind) -> &'static str {
    match kind {
        PageKind::Index => "index",
        PageKind::Topic => "topic",
        PageKind::Area => "area",
        PageKind::Module => "module",
        PageKind::File => "file",
    }
}

/// Parent directory of an agent-root-relative file path ("." when top-level).
fn parent_dir(rel_path: &str) -> &str {
    match rel_path.rfind('/') {
        Some(i) if i > 0 => &rel_path[..i],
        Some(_) => ".",
        None => ".",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::outlet::agent::model::{DetailBlock, NavGroup, PageKind};
    use std::path::PathBuf;

    fn ctx() -> (PathBuf, PathBuf) {
        (
            PathBuf::from("/proj"),
            PathBuf::from("/proj/litho.docs/.agent-content"),
        )
    }

    fn sample_page() -> AgentPage {
        AgentPage {
            rel_path: "tree/core/cache/index.md".into(),
            kind: PageKind::Area,
            title: "Cache".into(),
            summary: "Caching layer.".into(),
            breadcrumbs: vec![
                NavLink { label: "Home".into(), target: "index.md".into() },
                NavLink { label: "Cache".into(), target: "tree/core/cache/index.md".into() },
            ],
            nav_groups: vec![NavGroup {
                heading: "Modules".into(),
                links: vec![NavLink {
                    label: "CacheMgr".into(),
                    target: "tree/core/cache/modules/cachemgr.md".into(),
                }],
            }],
            source_paths: vec!["core/cache".into()],
            diagram_links: vec![NavLink {
                label: "Structure".into(),
                target: "tree/core/cache/diagrams/cache.md".into(),
            }],
            detail_blocks: vec![DetailBlock {
                heading: "Responsibilities".into(),
                items: vec!["store".into()],
            }],
            notes: Vec::new(),
        }
    }

    #[test]
    fn inner_href_resolves_relative_siblings() {
        assert_eq!(
            inner_href("tree/core/cache/index.md", "tree/core/cache/modules/cachemgr.md"),
            "modules/cachemgr.md"
        );
    }

    #[test]
    fn inner_href_resolves_upward() {
        assert_eq!(inner_href("tree/core/cache/index.md", "index.md"), "../../../index.md");
    }

    #[test]
    fn render_page_includes_all_blocks() {
        let (proj, root) = ctx();
        let rctx = RenderContext { project_root: &proj, agent_root: &root };
        let md = render_page(&sample_page(), &rctx);
        assert!(md.contains("# Cache"));
        assert!(md.contains("**Location:**"));
        assert!(md.contains("## Diagrams"));
        assert!(md.contains("## Navigate"));
        assert!(md.contains("## Source"));
        assert!(md.contains("## Responsibilities"));
        // Current breadcrumb is bold, not a link.
        assert!(md.contains("**Cache**"));
    }

    #[test]
    fn source_link_points_into_codebase() {
        let (proj, root) = ctx();
        let rctx = RenderContext { project_root: &proj, agent_root: &root };
        let md = render_page(&sample_page(), &rctx);
        // Page dir is /proj/litho.docs/.agent-content/tree/core/cache; reaching
        // /proj/core/cache climbs five levels then descends into core/cache.
        assert!(md.contains("(../../../../../core/cache)"), "md: {}", md);
    }

    #[test]
    fn render_diagram_has_fence_and_backlink() {
        let (proj, root) = ctx();
        let rctx = RenderContext { project_root: &proj, agent_root: &root };
        let d = DiagramFile {
            rel_path: "tree/core/cache/diagrams/cache.md".into(),
            title: "Structure".into(),
            caption: "cap".into(),
            body: "flowchart TD\n    A-->B".into(),
            owner: NavLink { label: "Cache".into(), target: "tree/core/cache/index.md".into() },
        };
        let md = render_diagram(&d, &rctx);
        assert!(md.contains("```mermaid"));
        assert!(md.contains("A-->B"));
        assert!(md.contains("../index.md"));
    }

    #[test]
    fn empty_target_is_safe() {
        assert_eq!(inner_href("index.md", ""), "#");
    }
}
