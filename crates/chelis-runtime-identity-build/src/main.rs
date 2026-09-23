use chelis_runtime_identity::*;
use serde::Deserialize;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::{
    env, fs,
    io::{self, Read},
    process::{Command, ExitCode},
};

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Derivation {
    recipe: RuntimeRecipe,
    captured: Vec<CapturedInput>,
    kind: RecordKind,
    provenance: BuildProvenance,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputFile {
    logical_path: String,
    physical: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Normalization {
    values: Vec<String>,
    roots: Vec<PathMapping>,
}

fn kind(s: &str) -> Result<RecordKind> {
    match s {
        "runtime" => Ok(RecordKind::Runtime),
        "cli" => Ok(RecordKind::Cli),
        "python" => Ok(RecordKind::Python),
        _ => Err(format!("unknown producer role {s}").into()),
    }
}
fn inspect(role: RecordKind, path: &str) -> Result<Descriptor> {
    let bytes = fs::read(path)?;
    Ok(match role {
        RecordKind::Runtime => decode_archive(&bytes)?,
        _ => decode_image(&bytes, role)?,
    })
}
fn hash_reader(reader: &mut impl Read) -> io::Result<ContentDigest> {
    let mut hasher = ContentHasher::new();
    let mut buffer = [0; 65536];
    loop {
        let length = reader.read(&mut buffer)?;
        if length == 0 {
            return Ok(hasher.finish());
        }
        hasher.update(&buffer[..length]);
    }
}
fn run(args: &[String]) -> Result<i32> {
    match args.first().map(String::as_str) {
        Some("core-plan") => {
            let roots: Vec<InventoryRoot> = serde_json::from_reader(io::stdin().lock())?;
            println!("{}", serde_json::to_string(&plan_inputs(&roots)?)?);
        }
        Some("core-normalize") => {
            let input: Normalization = serde_json::from_reader(io::stdin().lock())?;
            let values = input
                .values
                .iter()
                .map(|value| normalize_paths(value, &input.roots))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            println!("{}", serde_json::to_string(&values)?);
        }
        Some("core-capture") => {
            let files: Vec<InputFile> = serde_json::from_reader(io::stdin().lock())?;
            let inputs = files
                .into_iter()
                .map(|file| {
                    Ok(CapturedInput {
                        logical_path: file.logical_path,
                        digest: hash_reader(&mut fs::File::open(file.physical)?)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            println!("{}", serde_json::to_string(&inputs)?);
        }
        Some("core-derive") => {
            let input: Derivation = serde_json::from_reader(io::stdin().lock())?;
            let descriptor = derive_descriptor(&input.recipe, &input.captured)?;
            let record = encode_record(&descriptor, input.kind)?;
            let provenance = encode_provenance(&input.provenance)?;
            println!(
                "{}",
                serde_json::json!({"descriptor": descriptor, "record": record.as_slice(), "provenance": provenance})
            );
        }
        Some("core-hash") => {
            println!("{}", hash_reader(&mut io::stdin().lock())?);
        }
        Some("inspect") if args.len() == 3 => println!(
            "{}",
            serde_json::to_string(&inspect(kind(&args[1])?, &args[2])?)?
        ),
        Some("inspect-provenance") if args.len() == 3 => {
            let role = kind(&args[1])?;
            let bytes = fs::read(&args[2])?;
            let provenance = match role {
                RecordKind::Runtime => decode_archive_provenance(&bytes)?,
                _ => decode_image_provenance(&bytes, role)?,
            };
            println!("{}", serde_json::to_string(&provenance)?);
        }
        Some("verify-producers")
            if args.len() == 7
                && args[1] == "--runtime"
                && args[3] == "--cli"
                && args[5] == "--python" =>
        {
            let runtime = inspect(RecordKind::Runtime, &args[2])?;
            compare(&runtime, &inspect(RecordKind::Cli, &args[4])?)?;
            compare(&runtime, &inspect(RecordKind::Python, &args[6])?)?;
        }
        _ => {
            // Embed the same engine used by Cargo. Nix need not retain a source checkout.
            let executable = env::current_exe()?;
            let mut command = Command::new(
                env::var_os("CHELIS_IDENTITY_PYTHON").unwrap_or_else(|| "python3".into()),
            );
            command
                .arg("-c")
                .arg(include_str!(
                    "../../../scripts/runtime_identity_observer.py"
                ))
                .args(args)
                .env("CHELIS_IDENTITY_HELPER", executable);
            // Preserve Cargo's inherited jobserver descriptors through the
            // observer instead of spawning an intermediate process on macOS.
            #[cfg(unix)]
            return Err(command.exec().into());
            #[cfg(not(unix))]
            return Ok(command.status()?.code().unwrap_or(1));
        }
    }
    Ok(0)
}
fn main() -> ExitCode {
    match run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(1)),
        Err(error) => {
            eprintln!("runtime identity: {error}");
            ExitCode::FAILURE
        }
    }
}
