use std::fs;
use std::path::{Path, PathBuf};

use chelis_prove::{
    ProofStatus, ProofTier, SourceKind, TierSelection, VerificationRequest, VerificationResult,
    verify_source,
};

/// Dialect for requirement source interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Auto,
    Kiro,
    Traditional,
}

pub fn cmd_verify_spec(
    path: PathBuf,
    impl_path: Option<PathBuf>,
    tier: &str,
    smt_timeout: u64,
    samples: u32,
    format: &str,
    dialect: Dialect,
) -> Result<i32, String> {
    let (source, spans_json) = if path.is_file() {
        (read_source_file(&path, impl_path.as_deref())?, None)
    } else if path.is_dir() {
        read_spec_directory(&path, impl_path.as_deref(), dialect)?
    } else {
        return Err(format!("path `{}` does not exist", path.display()));
    };

    let tier_sel = match tier {
        "auto" => TierSelection::Auto,
        "fuzz-only" => TierSelection::FuzzOnly,
        "smt-only" => TierSelection::SmtOnly,
        "type-only" => TierSelection::TypeOnly,
        other => return Err(format!("invalid tier: {other}")),
    };

    let req = VerificationRequest {
        source,
        source_kind: SourceKind::Surf,
        tier: tier_sel,
        smt_timeout_ms: smt_timeout,
        fuzz_samples: samples,
        fuzz_seed: 0,
        inlining_depth_limit: 3,
        spans_json,
    };

    let result = verify_source(req).map_err(|e| e.to_string())?;

    match format {
        "json" => print_json(&result),
        _ => print_human(&result, &path),
    }

    if result.summary.failed > 0 {
        Ok(1)
    } else {
        Ok(0)
    }
}

fn translate_ears_file(ears_path: &Path) -> Result<String, String> {
    let output = std::process::Command::new("c-earchin")
        .args([
            "translate",
            ears_path.to_str().unwrap_or(""),
            "--inline",
            "--no-spans",
        ])
        .output()
        .map_err(|e| format!("failed to run c-earchin: {e}. Is it on PATH?"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("c-earchin translate failed: {stderr}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn translate_md_kiro(md_path: &Path) -> Option<String> {
    let output = std::process::Command::new("c-earchin")
        .args([
            "translate",
            md_path.to_str().unwrap_or(""),
            "--dialect",
            "kiro",
            "--inline",
            "--no-spans",
        ])
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        None
    }
}

fn read_source_file(path: &Path, impl_path: Option<&Path>) -> Result<String, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let main_source = match ext {
        "ears" => translate_ears_file(path)?,
        "md" => {
            // Try c-earchin --dialect kiro; fall back to reading as-is
            translate_md_kiro(path).unwrap_or_else(|| fs::read_to_string(path).unwrap_or_default())
        }
        _ => fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?,
    };

    if let Some(impl_p) = impl_path {
        let impl_source =
            fs::read_to_string(impl_p).map_err(|e| format!("read {}: {e}", impl_p.display()))?;
        return Ok(format!("{impl_source}\n{main_source}"));
    }

    // Look for sibling .ch implementation file
    if ext == "ears" || ext == "md" {
        let ch_path = path.with_extension("ch");
        if ch_path.exists() {
            let impl_source = fs::read_to_string(&ch_path)
                .map_err(|e| format!("read {}: {e}", ch_path.display()))?;
            return Ok(format!("{impl_source}\n{main_source}"));
        }
    }

    Ok(main_source)
}

fn read_spec_directory(
    dir: &Path,
    impl_path: Option<&Path>,
    dialect: Dialect,
) -> Result<(String, Option<String>), String> {
    let mut sources = Vec::new();
    let mut spans_json = None;

    // Kiro dialect: look for requirements.md + implementation.ch
    if matches!(dialect, Dialect::Kiro | Dialect::Auto) {
        let req_md = dir.join("requirements.md");
        let impl_ch = dir.join("implementation.ch");
        if req_md.exists() && impl_ch.exists() {
            let impl_source = fs::read_to_string(&impl_ch)
                .map_err(|e| format!("read {}: {e}", impl_ch.display()))?;
            sources.push(impl_source);

            // Try translating requirements.md via c-earchin
            if let Some(translated) = translate_md_kiro(&req_md) {
                sources.push(translated);
            }
            // If c-earchin not available, just use the .ch files
        }
    }

    // Collect .ch files
    for entry in fs::read_dir(dir).map_err(|e| format!("read dir: {e}"))? {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let p = entry.path();
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "ch" => {
                // Skip implementation.ch if already loaded via kiro dialect
                if matches!(dialect, Dialect::Kiro | Dialect::Auto)
                    && p.file_name().and_then(|n| n.to_str()) == Some("implementation.ch")
                    && dir.join("requirements.md").exists()
                {
                    continue;
                }
                let content =
                    fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
                sources.push(content);
            }
            "ears" => {
                // Translate .ears files via c-earchin
                if let Ok(translated) = translate_ears_file(&p) {
                    sources.push(translated);
                }
                // Look for sibling .spans.json
                let spans_path = p.with_extension("spans.json");
                if spans_path.exists() {
                    spans_json = fs::read_to_string(&spans_path).ok();
                }
            }
            _ => {
                // Check for .spans.json files
                if p.to_str()
                    .map(|s| s.ends_with(".spans.json"))
                    .unwrap_or(false)
                {
                    spans_json = fs::read_to_string(&p).ok();
                }
            }
        }
    }

    if let Some(impl_p) = impl_path {
        let content =
            fs::read_to_string(impl_p).map_err(|e| format!("read {}: {e}", impl_p.display()))?;
        sources.push(content);
    }
    if sources.is_empty() {
        return Err(format!("no .ch or .ears files found in {}", dir.display()));
    }
    Ok((sources.join("\n"), spans_json))
}

fn print_json(result: &VerificationResult) {
    for prop in &result.properties {
        println!("{}", serde_json::to_string(prop).unwrap_or_default());
    }
    println!(
        "{}",
        serde_json::json!({
            "kind": "summary",
            "total": result.summary.total,
            "proved": result.summary.proved,
            "statistically_validated": result.summary.statistically_validated,
            "failed": result.summary.failed,
        })
    );
}

fn print_human(result: &VerificationResult, path: &Path) {
    println!("Verifying: {}\n", path.display());
    for prop in &result.properties {
        let icon = match &prop.status {
            ProofStatus::Proved => "\u{2713}",
            ProofStatus::StatisticallyValidated { .. } => "\u{2713}",
            ProofStatus::Disproved { .. } => "\u{2717}",
            _ => "-",
        };
        let tier = match prop.proof_tier {
            ProofTier::Smt => "SMT",
            ProofTier::Fuzz => "fuzz",
            ProofTier::TypeSystem => "type",
        };
        println!(
            "  {} {}  [{}  {}ms]",
            icon, prop.name, tier, prop.duration_ms
        );
    }
    println!(
        "\nSummary: {}/{} verified",
        result.summary.proved + result.summary.statistically_validated,
        result.summary.total
    );
    println!("  Proved by SMT: {}", result.summary.proved);
    println!(
        "  Validated by fuzz: {}",
        result.summary.statistically_validated
    );
    println!("  Failed: {}", result.summary.failed);
}
