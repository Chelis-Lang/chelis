use std::fs;
use std::path::{Path, PathBuf};

use chelis_prove::{
    ProofStatus, ProofTier, SourceKind, TierSelection, VerificationRequest, VerificationResult,
    verify_source,
};

pub fn cmd_verify_spec(
    path: PathBuf,
    impl_path: Option<PathBuf>,
    tier: &str,
    smt_timeout: u64,
    samples: u32,
    format: &str,
) -> Result<i32, String> {
    let source = if path.is_file() {
        read_source_file(&path, impl_path.as_deref())?
    } else if path.is_dir() {
        read_spec_directory(&path, impl_path.as_deref())?
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

fn read_source_file(path: &Path, impl_path: Option<&Path>) -> Result<String, String> {
    let main_source =
        fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;

    if let Some(impl_p) = impl_path {
        let impl_source =
            fs::read_to_string(impl_p).map_err(|e| format!("read {}: {e}", impl_p.display()))?;
        return Ok(format!("{impl_source}\n{main_source}"));
    }

    if path.extension().map(|e| e == "ears").unwrap_or(false) {
        let ch_path = path.with_extension("ch");
        if ch_path.exists() {
            let impl_source = fs::read_to_string(&ch_path)
                .map_err(|e| format!("read {}: {e}", ch_path.display()))?;
            return Ok(format!("{impl_source}\n{main_source}"));
        }
    }

    Ok(main_source)
}

fn read_spec_directory(dir: &Path, impl_path: Option<&Path>) -> Result<String, String> {
    let mut sources = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("read dir: {e}"))? {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let p = entry.path();
        if p.extension().map(|e| e == "ch").unwrap_or(false) {
            let content =
                fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
            sources.push(content);
        }
    }
    if let Some(impl_p) = impl_path {
        let content =
            fs::read_to_string(impl_p).map_err(|e| format!("read {}: {e}", impl_p.display()))?;
        sources.push(content);
    }
    if sources.is_empty() {
        return Err(format!("no .ch files found in {}", dir.display()));
    }
    Ok(sources.join("\n"))
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
