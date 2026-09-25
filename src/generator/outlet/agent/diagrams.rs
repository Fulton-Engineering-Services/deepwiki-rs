//! Deterministic Mermaid diagram synthesis from structured research data.
//!
//! Only `KeyModuleReport` carries ready-made Mermaid (`flowchart_mermaid`,
//! `sequence_diagram_mermaid`); those are emitted verbatim by the model layer.
//! Every other diagram (system context, area hierarchy, domain dependencies,
//! workflows, database ER/flows, boundary surfaces) is *synthesized* here from the
//! structured research reports so the agent set never depends on LLM prose.
//!
//! Each synthesizer returns `None` when there is nothing meaningful to draw, so
//! the caller can skip creating an empty diagram file. Node ids are made valid via
//! [`crate::generator::outlet::agent::links::mermaid_node_id`]; human text is kept
//! in quoted labels via [`crate::generator::outlet::agent::links::escape_mermaid_label`].

use crate::generator::outlet::agent::links::{
    SlugRegistry, escape_mermaid_label, mermaid_node_id,
};
use crate::generator::research::area_tree::{AreaNode, AreaTree};
use crate::generator::research::types::{
    BoundaryAnalysisReport, BusinessFlow, DatabaseOverviewReport, DomainModulesReport,
    DomainRelation, SystemContextReport,
};

/// Build a quoted Mermaid label for a display name.
fn label(name: &str) -> String {
    format!("\"{}\"", escape_mermaid_label(name))
}

/// Wrap a raw Mermaid body into a standalone Markdown diagram file with a title,
/// an optional caption, and an optional back-link to the owning page.
pub fn render_diagram_markdown(
    title: &str,
    caption: Option<&str>,
    body: &str,
    back_link: Option<(&str, &str)>,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {}\n\n", title));
    if let Some(cap) = caption
        && !cap.trim().is_empty()
    {
        out.push_str(cap.trim());
        out.push_str("\n\n");
    }
    out.push_str("```mermaid\n");
    out.push_str(body.trim());
    out.push_str("\n```\n");
    if let Some((href, text)) = back_link {
        out.push_str(&format!("\n[← Back to {}]({})\n", text, href));
    }
    out
}

/// C4 Level-1 system context: the system, its users, and external systems.
pub fn context_diagram(report: &SystemContextReport) -> Option<String> {
    let sys_name = if report.project_name.trim().is_empty() {
        "System".to_string()
    } else {
        report.project_name.trim().to_string()
    };
    let mut reg = SlugRegistry::new();
    let sys_id = mermaid_node_id(&sys_name, &mut reg);

    let mut lines: Vec<String> = Vec::new();
    lines.push("flowchart TD".to_string());
    lines.push(format!("    {}[{}]", sys_id, label(&sys_name)));

    let mut edges: Vec<String> = Vec::new();
    for persona in &report.target_users {
        if persona.name.trim().is_empty() {
            continue;
        }
        let id = mermaid_node_id(&persona.name, &mut reg);
        lines.push(format!(
            "    {}[{}{}]",
            id,
            label(&persona.name),
            shape_suffix("User")
        ));
        edges.push(format!("    {} --> {}", id, sys_id));
    }
    for ext in &report.external_systems {
        if ext.name.trim().is_empty() {
            continue;
        }
        let id = mermaid_node_id(&ext.name, &mut reg);
        lines.push(format!("    {}[{}]", id, label(&ext.name)));
        let edge_label = if ext.interaction_type.trim().is_empty() {
            String::new()
        } else {
            format!("|{}|", escape_mermaid_label(&ext.interaction_type))
        };
        edges.push(format!("    {} -->{} {}", sys_id, edge_label, id));
    }

    if edges.is_empty() && lines.len() <= 2 {
        // Only the system node — not worth a diagram.
        return None;
    }
    lines.extend(edges);
    Some(lines.join("\n"))
}

/// Area hierarchy across the top `max_depth` levels of the tree.
pub fn area_hierarchy_diagram(tree: &AreaTree, max_depth: usize) -> Option<String> {
    let mut reg = SlugRegistry::new();
    let mut lines: Vec<String> = vec!["flowchart TD".to_string()];
    let mut edges: Vec<String> = Vec::new();
    let root_id = mermaid_node_id(&tree.root.name, &mut reg);
    lines.push(format!("    {}[{}]", root_id, label(&tree.root.name)));
    collect_hierarchy(&tree.root, &root_id, 1, max_depth, &mut reg, &mut lines, &mut edges);
    if edges.is_empty() {
        return None;
    }
    lines.extend(edges);
    Some(lines.join("\n"))
}

fn collect_hierarchy(
    node: &AreaNode,
    node_id: &str,
    depth: usize,
    max_depth: usize,
    reg: &mut SlugRegistry,
    lines: &mut Vec<String>,
    edges: &mut Vec<String>,
) {
    if depth > max_depth {
        return;
    }
    for child in &node.children {
        let cid = mermaid_node_id(&child.name, reg);
        lines.push(format!("    {}[{}]", cid, label(&child.name)));
        edges.push(format!("    {} --> {}", node_id, cid));
        collect_hierarchy(child, &cid, depth + 1, max_depth, reg, lines, edges);
    }
}

/// One area node with its immediate children and the modules that live under it.
pub fn area_structure_diagram(
    node: &AreaNode,
    module_names: &[String],
) -> Option<String> {
    let mut reg = SlugRegistry::new();
    let mut lines: Vec<String> = vec!["flowchart TD".to_string()];
    let root_id = mermaid_node_id(&node.name, &mut reg);
    lines.push(format!("    {}[{}]", root_id, label(&node.name)));

    let mut edges: Vec<String> = Vec::new();
    for child in &node.children {
        let cid = mermaid_node_id(&child.name, &mut reg);
        lines.push(format!("    {}[{}]", cid, label(&child.name)));
        edges.push(format!("    {} --> {}", root_id, cid));
    }
    for module in module_names {
        if module.trim().is_empty() {
            continue;
        }
        let mid = mermaid_node_id(module, &mut reg);
        lines.push(format!("    {}[{}]", mid, label(module)));
        edges.push(format!("    {} -.-> {}", root_id, mid));
    }

    if edges.is_empty() {
        return None;
    }
    lines.extend(edges);
    Some(lines.join("\n"))
}

/// Inter-domain dependency graph (left-to-right).
pub fn domain_dependencies_diagram(relations: &[DomainRelation]) -> Option<String> {
    let mut reg = SlugRegistry::new();
    let mut lines: Vec<String> = vec!["flowchart LR".to_string()];
    let mut edges: Vec<String> = Vec::new();
    for rel in relations {
        if rel.from_domain.trim().is_empty() || rel.to_domain.trim().is_empty() {
            continue;
        }
        let from = mermaid_node_id(&rel.from_domain, &mut reg);
        let to = mermaid_node_id(&rel.to_domain, &mut reg);
        lines.push(format!("    {}[{}]", from, label(&rel.from_domain)));
        lines.push(format!("    {}[{}]", to, label(&rel.to_domain)));
        let edge_label = if rel.relation_type.trim().is_empty() {
            String::new()
        } else {
            format!("|{}|", escape_mermaid_label(&rel.relation_type))
        };
        edges.push(format!("    {} -->{} {}", from, edge_label, to));
    }
    if edges.is_empty() {
        return None;
    }
    lines.extend(edges);
    // Deduplicate identical node declarations (a name can appear in many edges).
    Some(dedupe_declarations(&lines))
}

/// Domain modules and their sub-modules (top-down decomposition).
pub fn domains_diagram(report: &DomainModulesReport) -> Option<String> {
    let mut reg = SlugRegistry::new();
    let mut lines: Vec<String> = vec!["flowchart TD".to_string()];
    let mut edges: Vec<String> = Vec::new();
    for domain in &report.domain_modules {
        if domain.name.trim().is_empty() {
            continue;
        }
        let dom_id = mermaid_node_id(&domain.name, &mut reg);
        lines.push(format!("    {}[{}]", dom_id, label(&domain.name)));
        for sub in &domain.sub_modules {
            if sub.name.trim().is_empty() {
                continue;
            }
            let sub_id = mermaid_node_id(&sub.name, &mut reg);
            lines.push(format!("    {}[{}]", sub_id, label(&sub.name)));
            edges.push(format!("    {} --> {}", dom_id, sub_id));
        }
    }
    if lines.len() <= 1 {
        return None;
    }
    lines.extend(edges);
    Some(dedupe_declarations(&lines))
}

/// A single business flow as a left-to-right chain of its steps.
///
/// Returns `(flow_name, body)` pairs — one diagram per business flow.
pub fn workflow_diagrams(flows: &[BusinessFlow]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for flow in flows {
        if flow.name.trim().is_empty() || flow.steps.is_empty() {
            continue;
        }
        let mut reg = SlugRegistry::new();
        let mut lines: Vec<String> = vec!["flowchart LR".to_string()];
        let mut ordered = flow.steps.clone();
        ordered.sort_by_key(|s| s.step);

        let mut prev: Option<String> = None;
        for (idx, step) in ordered.iter().enumerate() {
            let caption = if step.operation.trim().is_empty() {
                format!("Step {}", idx + 1)
            } else {
                step.operation.trim().to_string()
            };
            let mut name = format!("{}. {}", idx + 1, caption);
            if !step.domain_module.trim().is_empty() {
                name = format!("{}\n[{}]", name, step.domain_module.trim());
            }
            let sid = mermaid_node_id(&format!("step-{}", idx), &mut reg);
            let label_text = caption.replace('\n', " ");
            lines.push(format!(
                "    {}[\"{}. {}\"]",
                sid,
                idx + 1,
                escape_mermaid_label(&label_text)
            ));
            if let Some(p) = &prev {
                lines.push(format!("    {} --> {}", p, sid));
            }
            prev = Some(sid);
            let _ = name;
        }
        if prev.is_some() {
            out.push((flow.name.trim().to_string(), lines.join("\n")));
        }
    }
    out
}

/// Database ER diagram from table foreign-key/reference relationships.
pub fn database_er_diagram(report: &DatabaseOverviewReport) -> Option<String> {
    if report.table_relationships.is_empty() {
        return None;
    }
    let mut reg = SlugRegistry::new();
    let mut lines: Vec<String> = vec!["erDiagram".to_string()];
    for rel in &report.table_relationships {
        let from = er_entity(&rel.from_table, &mut reg);
        let to = er_entity(&rel.to_table, &mut reg);
        let rel_label = if rel.relationship_type.trim().is_empty() {
            "references".to_string()
        } else {
            rel.relationship_type.trim().to_string()
        };
        lines.push(format!(
            "    {} ||--o{{ {} : \"{}\"",
            from,
            to,
            escape_mermaid_label(&rel_label)
        ));
    }
    if lines.len() <= 1 {
        return None;
    }
    Some(dedupe_declarations(&lines))
}

/// Data-flow graph from source to destination with the operations involved.
pub fn database_flows_diagram(report: &DatabaseOverviewReport) -> Option<String> {
    let mut reg = SlugRegistry::new();
    let mut lines: Vec<String> = vec!["flowchart LR".to_string()];
    let mut edges: Vec<String> = Vec::new();
    for flow in &report.data_flows {
        if flow.source.trim().is_empty() || flow.destination.trim().is_empty() {
            continue;
        }
        let s = mermaid_node_id(&flow.source, &mut reg);
        let d = mermaid_node_id(&flow.destination, &mut reg);
        lines.push(format!("    {}[{}]", s, label(&flow.source)));
        lines.push(format!("    {}[{}]", d, label(&flow.destination)));
        let ops = flow.operations.join(", ");
        let edge_label = if ops.trim().is_empty() {
            String::new()
        } else {
            format!("|{}|", escape_mermaid_label(&ops))
        };
        edges.push(format!("    {} -->{} {}", s, edge_label, d));
    }
    if edges.is_empty() {
        return None;
    }
    lines.extend(edges);
    Some(dedupe_declarations(&lines))
}

/// Boundary surfaces: CLI commands, API endpoints, and router paths clustered
/// under a single system node.
pub fn boundary_surfaces_diagram(report: &BoundaryAnalysisReport) -> Option<String> {
    let mut reg = SlugRegistry::new();
    let mut lines: Vec<String> = vec!["flowchart TD".to_string()];
    let sys_id = mermaid_node_id("system", &mut reg);
    lines.push(format!("    {}[{}]", sys_id, label("System")));
    let mut edges: Vec<String> = Vec::new();

    for cli in &report.cli_boundaries {
        if cli.command.trim().is_empty() {
            continue;
        }
        let id = mermaid_node_id(&format!("cli-{}", cli.command), &mut reg);
        lines.push(format!("    {}[{}]", id, label(&cli.command)));
        edges.push(format!("    {} --> {}", sys_id, id));
    }
    for api in &report.api_boundaries {
        let name = format!("{} {}", api.method, api.endpoint);
        if api.endpoint.trim().is_empty() {
            continue;
        }
        let id = mermaid_node_id(&format!("api-{}", name), &mut reg);
        lines.push(format!("    {}[{}]", id, label(&name)));
        edges.push(format!("    {} --> {}", sys_id, id));
    }
    for router in &report.router_boundaries {
        if router.path.trim().is_empty() {
            continue;
        }
        let id = mermaid_node_id(&format!("router-{}", router.path), &mut reg);
        lines.push(format!("    {}[{}]", id, label(&router.path)));
        edges.push(format!("    {} --> {}", sys_id, id));
    }

    if edges.is_empty() {
        return None;
    }
    lines.extend(edges);
    Some(dedupe_declarations(&lines))
}

/// erDiagram entity names must be bare identifiers; keep the table name readable
/// but strip characters that break the grammar (e.g. `schema.table`).
fn er_entity(raw: &str, reg: &mut SlugRegistry) -> String {
    let cleaned = raw.replace('.', "_").replace([' ', '-', '/'], "_");
    mermaid_node_id(&cleaned, reg)
}

/// Append a small decorative shape to distinguish actor-type nodes. Kept simple
/// so the emitted diagram stays broadly renderable.
fn shape_suffix(kind: &str) -> String {
    match kind {
        "User" => "(( ))".to_string(),
        _ => String::new(),
    }
}

/// Remove duplicate `id["label"]` declaration lines while preserving order.
/// A node referenced by several edges is declared once. Header lines (the diagram
/// type) and edge lines are kept as-is.
fn dedupe_declarations(lines: &[String]) -> String {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        let is_declaration = trimmed.contains("[\"") || trimmed.contains("(( ))");
        if is_declaration {
            if seen.insert(trimmed.to_string()) {
                out.push(line.clone());
            }
        } else {
            out.push(line.clone());
        }
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::research::types::{
        APIBoundary, CLIBoundary, DataFlow, ExternalSystem, SubModule, TableRelationship,
        UserPersona,
    };

    fn system_context() -> SystemContextReport {
        SystemContextReport {
            project_name: "LockBox".to_string(),
            target_users: vec![UserPersona {
                name: "Ops Team".into(),
                ..Default::default()
            }],
            external_systems: vec![ExternalSystem {
                name: "AWS Bedrock".into(),
                interaction_type: "REST".into(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn context_diagram_renders_system_and_edges() {
        let body = context_diagram(&system_context()).expect("should render");
        assert!(body.starts_with("flowchart TD"));
        assert!(body.contains("LockBox"));
        assert!(body.contains("AWS Bedrock"));
    }

    #[test]
    fn context_diagram_none_when_empty() {
        let report = SystemContextReport::default();
        assert!(context_diagram(&report).is_none());
    }

    #[test]
    fn domain_dependencies_diagram_draws_relations() {
        let relations = vec![DomainRelation {
            from_domain: "Auth & Access".into(),
            to_domain: "Data Pipeline".into(),
            relation_type: "Service Call".into(),
            ..Default::default()
        }];
        let body = domain_dependencies_diagram(&relations).expect("should render");
        assert!(body.starts_with("flowchart LR"));
        assert!(body.contains("Service Call"));
    }

    #[test]
    fn workflow_diagram_orders_steps() {
        let flow = BusinessFlow {
            name: "Ingest".into(),
            steps: vec![
                crate::generator::research::types::BusinessFlowStep {
                    step: 2,
                    operation: "Load".into(),
                    ..Default::default()
                },
                crate::generator::research::types::BusinessFlowStep {
                    step: 1,
                    operation: "Scan".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let diagrams = workflow_diagrams(&[flow]);
        assert_eq!(diagrams.len(), 1);
        let (_, body) = &diagrams[0];
        let scan = body.find("Scan").unwrap();
        let load = body.find("Load").unwrap();
        assert!(scan < load, "steps must be ordered by step number");
    }

    #[test]
    fn database_er_diagram_uses_relationships() {
        let report = DatabaseOverviewReport {
            table_relationships: vec![TableRelationship {
                from_table: "dbo.orders".into(),
                to_table: "dbo.customers".into(),
                relationship_type: "ForeignKey".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let body = database_er_diagram(&report).expect("should render");
        assert!(body.starts_with("erDiagram"));
        assert!(body.contains("ForeignKey"));
    }

    #[test]
    fn database_flows_diagram_draws_edges() {
        let report = DatabaseOverviewReport {
            data_flows: vec![DataFlow {
                source: "staging".into(),
                destination: "warehouse".into(),
                operations: vec!["INSERT".into()],
                ..Default::default()
            }],
            ..Default::default()
        };
        let body = database_flows_diagram(&report).expect("should render");
        assert!(body.contains("INSERT"));
    }

    #[test]
    fn boundary_diagram_clusters_surfaces() {
        let report = BoundaryAnalysisReport {
            cli_boundaries: vec![CLIBoundary {
                command: "etl run".into(),
                ..Default::default()
            }],
            api_boundaries: vec![APIBoundary {
                endpoint: "/api/v1".into(),
                method: "GET".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let body = boundary_surfaces_diagram(&report).expect("should render");
        assert!(body.contains("etl run"));
        assert!(body.contains("/api/v1"));
    }

    #[test]
    fn domains_diagram_includes_submodules() {
        let report = DomainModulesReport {
            domain_modules: vec![crate::generator::research::types::DomainModule {
                name: "Cache".into(),
                sub_modules: vec![SubModule {
                    name: "Registry".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let body = domains_diagram(&report).expect("should render");
        assert!(body.contains("Cache"));
        assert!(body.contains("Registry"));
    }

    #[test]
    fn render_diagram_markdown_wraps_fence_and_backlink() {
        let md = render_diagram_markdown(
            "Title",
            Some("A caption."),
            "flowchart TD\n    A-->B",
            Some(("../index.md", "area")),
        );
        assert!(md.contains("# Title"));
        assert!(md.contains("```mermaid"));
        assert!(md.contains("A-->B"));
        assert!(md.contains("[← Back to area](../index.md)"));
    }

    #[test]
    fn dedupe_removes_duplicate_node_declarations() {
        let lines = vec![
            "flowchart LR".to_string(),
            "    a[\"A\"]".to_string(),
            "    a[\"A\"]".to_string(),
            "    a --> b".to_string(),
        ];
        let body = dedupe_declarations(&lines);
        assert_eq!(body.matches("a[\"A\"]").count(), 1);
    }
}
