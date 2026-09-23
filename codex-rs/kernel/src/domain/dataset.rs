use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use rustpython_ast as ast;
use rustpython_ast::Ranged;
use rustpython_ast::Visitor;
use rustpython_parser::Parse;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;
use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::path::Path;

#[derive(Default)]
struct LoadedNames(BTreeSet<String>);

impl Visitor for LoadedNames {
    fn visit_expr_name(&mut self, node: ast::ExprName) {
        if node.ctx == ast::ExprContext::Load {
            self.0.insert(node.id.to_string());
        }
    }

    fn visit_arguments(&mut self, node: ast::Arguments) {
        for argument in node
            .posonlyargs
            .into_iter()
            .chain(node.args)
            .chain(node.kwonlyargs)
        {
            self.visit_arg(argument.def);
            if let Some(default) = argument.default {
                self.visit_expr(*default);
            }
        }
        for argument in node.vararg.into_iter().chain(node.kwarg) {
            self.visit_arg(*argument);
        }
    }

    fn visit_arg(&mut self, node: ast::Arg) {
        if let Some(annotation) = node.annotation {
            self.visit_expr(*annotation);
        }
    }

    fn visit_keyword(&mut self, node: ast::Keyword) {
        self.visit_expr(node.value);
    }

    fn visit_comprehension(&mut self, node: ast::Comprehension) {
        self.visit_expr(node.target);
        self.visit_expr(node.iter);
        for condition in node.ifs {
            self.visit_expr(condition);
        }
    }

    fn visit_withitem(&mut self, node: ast::WithItem) {
        self.visit_expr(node.context_expr);
        if let Some(target) = node.optional_vars {
            self.visit_expr(*target);
        }
    }

    fn visit_match_case(&mut self, node: ast::MatchCase) {
        self.visit_pattern(node.pattern);
        if let Some(guard) = node.guard {
            self.visit_expr(*guard);
        }
        for statement in node.body {
            self.visit_stmt(statement);
        }
    }
}

#[derive(Default)]
struct ChildStatements(Vec<ast::Stmt>);

impl Visitor for ChildStatements {
    fn visit_stmt(&mut self, node: ast::Stmt) {
        self.0.push(node);
    }

    fn visit_match_case(&mut self, node: ast::MatchCase) {
        self.0.extend(node.body);
    }
}

fn defines(target: &ast::Expr, name: &str) -> bool {
    match target {
        ast::Expr::Name(node) => node.id.as_str() == name,
        ast::Expr::Tuple(node) => node.elts.iter().any(|e| defines(e, name)),
        ast::Expr::List(node) => node.elts.iter().any(|e| defines(e, name)),
        _ => false,
    }
}

fn definition(node: &ast::Stmt, name: &str) -> bool {
    let aliases = match node {
        ast::Stmt::Import(node) => &node.names,
        ast::Stmt::ImportFrom(node) => &node.names,
        ast::Stmt::Assign(node) => return node.targets.iter().any(|t| defines(t, name)),
        _ => return false,
    };
    aliases
        .iter()
        .any(|alias| alias.asname.as_ref().unwrap_or(&alias.name).as_str() == name)
}

fn source_lines(code: &str, node: &impl Ranged) -> String {
    let start = usize::from(node.start());
    let end = usize::from(node.end());
    let start = code[..start].rfind('\n').map_or(0, |index| index + 1);
    let end = code[end..]
        .find('\n')
        .map_or(code.len(), |index| end + index);
    code[start..end].to_owned()
}

fn clean_docstring(value: &str) -> String {
    let mut expanded = String::new();
    let mut column = 0;
    for character in value.chars() {
        if character == '\t' {
            let count = 8 - column % 8;
            expanded.push_str(&" ".repeat(count));
            column += count;
        } else {
            expanded.push(character);
            column = if character == '\n' || character == '\r' {
                0
            } else {
                column + 1
            };
        }
    }
    let mut lines: Vec<_> = expanded.split('\n').map(str::to_owned).collect();
    let margin = lines
        .iter()
        .skip(1)
        .filter(|line| !line.trim_start().is_empty())
        .map(|line| line.chars().count() - line.trim_start().chars().count())
        .min();
    if let Some(first) = lines.first_mut() {
        *first = first.trim_start().to_owned();
    }
    if let Some(margin) = margin {
        for line in lines.iter_mut().skip(1) {
            *line = line.chars().skip(margin).collect();
        }
    }
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    let start = lines
        .iter()
        .position(|line| !line.is_empty())
        .unwrap_or(lines.len());
    lines[start..].join("\n")
}

fn parse(bytes: &[u8], path: &Path, level: u64) -> Result<Value> {
    let code = std::str::from_utf8(bytes)?
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let tree = ast::Suite::parse(&code, &path.to_string_lossy())?;
    let mut selected = BTreeSet::new();
    let mut names = LoadedNames::default();
    let mut functions = BTreeSet::new();
    for (index, node) in tree.iter().enumerate() {
        if let ast::Stmt::FunctionDef(function) = node
            && matches!(function.name.as_str(), "get_inputs" | "get_init_inputs")
        {
            selected.insert(index);
            functions.insert(function.name.as_str());
            names.visit_stmt(node.clone());
        }
    }
    ensure!(
        functions.contains("get_inputs") && functions.contains("get_init_inputs"),
        "benchmark must define get_inputs() and get_init_inputs()"
    );
    let mut processed = BTreeSet::new();
    while let Some(name) = names.0.difference(&processed).next().cloned() {
        processed.insert(name.clone());
        if let Some((index, node)) = tree
            .iter()
            .enumerate()
            .find(|(_, node)| definition(node, &name))
            && selected.insert(index)
        {
            names.visit_stmt(node.clone());
        }
    }
    let input_generator = selected
        .into_iter()
        .map(|index| source_lines(&code, &tree[index]))
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut queue: VecDeque<_> = tree.into();
    let mut prompt = code.clone();
    let mut description = String::new();
    while let Some(node) = queue.pop_front() {
        if let ast::Stmt::ClassDef(class) = &node
            && class.name.as_str() == "Model"
        {
            prompt = source_lines(&code, class);
            if let Some(ast::Stmt::Expr(expression)) = class.body.first()
                && let ast::Expr::Constant(constant) = expression.value.as_ref()
                && let ast::Constant::Str(value) = &constant.value
            {
                description = clean_docstring(value);
            }
            break;
        }
        let mut children = ChildStatements::default();
        children.generic_visit_stmt(node);
        queue.extend(children.0);
    }
    let filename = path.file_name().context("task filename")?.to_string_lossy();
    if description.is_empty() {
        description = format!(
            "Optimize PyTorch model: {}",
            path.file_stem()
                .context("task stem")?
                .to_string_lossy()
                .replace('_', " ")
        );
    }
    Ok(json!({
        "task_id": format!("KernelBench/level{level}/{filename}"), "prompt": prompt,
        "description": description, "entry_point": "Model", "canonical_solution": code,
        "input_generator": input_generator, "test": "", "completion_list": [],
        "test_case_list": [], "test_results": null, "need_reproduce": true,
        "dataset": "kernelbench", "benchmark_type": "performance", "requires_gpu": true,
        "baseline_metrics": null, "file_path": path,
        "sha256": format!("{:x}", Sha256::digest(code.as_bytes())),
        "file_sha256": format!("{:x}", Sha256::digest(bytes)),
    }))
}

pub(crate) async fn prepare(payload: &Value) -> Result<Value> {
    let root =
        Path::new(payload["dataset_root"].as_str().context("dataset root")?).canonicalize()?;
    let veomni = payload["dataset"] == "kernelbench_veomni";
    let level = if veomni {
        1
    } else {
        payload["level"].as_u64().unwrap_or(1)
    };
    let directory = if veomni {
        root
    } else {
        root.join(format!("level{level}"))
    };
    let mut files = tokio::fs::read_dir(directory).await?;
    let mut paths = Vec::new();
    while let Some(entry) = files.next_entry().await? {
        if entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "py")
        {
            paths.push(entry.path());
        }
    }
    paths.sort();
    let requested = payload["task"].as_str().context("task selector")?;
    let mut matches = Vec::new();
    for path in paths {
        let result = async { parse(&tokio::fs::read(&path).await?, &path, level) }.await;
        let problem = match result {
            Ok(problem) => problem,
            Err(error) if !veomni => {
                eprintln!("Warning: Failed to parse {}: {error}", path.display());
                continue;
            }
            Err(error) => return Err(error),
        };
        if problem["task_id"] == requested
            || path.file_name().is_some_and(|name| name == requested)
            || path.file_stem().is_some_and(|name| name == requested)
        {
            matches.push(problem);
        }
    }
    ensure!(
        matches.len() == 1,
        "task must match exactly one filename or task ID, matched {}",
        matches.len()
    );
    Ok(matches.remove(0))
}

#[cfg(test)]
#[path = "dataset_tests.rs"]
mod tests;
