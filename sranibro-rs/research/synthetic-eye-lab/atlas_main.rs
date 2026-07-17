//! Standalone, renderer-only entry point for the preregistered moment atlas.
//!
//! Its module tree deliberately has no EyeNet, experiment, solver, or recording loader.

mod atlas;
mod atlas_output;
#[allow(dead_code)]
mod renderer;

use std::error::Error;
use std::path::{Path, PathBuf};

use atlas_output::{
    ensure_output_empty, repository_state, verify_compiled_identity, verify_preregistration,
    write_atlas, BUILD_PROFILE,
};

fn main() {
    match run(std::env::args().skip(1)) {
        Ok(RunOutcome::Completed(out)) => println!("Atlas results: {}", out.display()),
        Ok(RunOutcome::Help) => print_help(),
        Err(error) => {
            eprintln!("synthetic-eye-atlas: {error}");
            std::process::exit(1);
        }
    }
}

fn run(args: impl IntoIterator<Item = String>) -> Result<RunOutcome, Box<dyn Error>> {
    let Some(args) = Args::parse(args)? else {
        return Ok(RunOutcome::Help);
    };
    require_release_build()?;
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Both checks precede the expensive atlas preparation. They are repeated by the
    // writer so a destination or worktree change cannot silently mix the record.
    ensure_output_empty(&args.out)?;
    let repository = repository_state(manifest_dir)?;
    if repository.dirty {
        return Err("recorded atlas run requires a clean Git worktree".into());
    }
    verify_compiled_identity(&repository)?;
    verify_preregistration(manifest_dir)?;

    let prepared = atlas::prepare().map_err(|error| format!("atlas preparation NO-GO: {error}"))?;

    let final_repository = repository_state(manifest_dir)?;
    if final_repository != repository || final_repository.dirty {
        return Err(
            "Git worktree changed during atlas preparation; no artifacts were written".into(),
        );
    }
    verify_compiled_identity(&final_repository)?;
    write_atlas(&args.out, manifest_dir, &final_repository, &prepared)?;
    Ok(RunOutcome::Completed(args.out))
}

fn require_release_build() -> Result<(), Box<dyn Error>> {
    if BUILD_PROFILE != "release" {
        return Err(
            format!(
                "recorded atlas runs require Cargo profile release (--release); current profile is {BUILD_PROFILE}"
            )
            .into(),
        );
    }
    if cfg!(debug_assertions) {
        return Err(
            "the release profile must disable debug assertions for recorded atlas runs".into(),
        );
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum RunOutcome {
    Completed(PathBuf),
    Help,
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    out: PathBuf,
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>, Box<dyn Error>> {
        let mut out = None;
        let mut args = args.into_iter();
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--out" => {
                    if out.is_some() {
                        return Err("--out may be specified only once".into());
                    }
                    let value = args.next().ok_or("missing value after --out")?;
                    if value.starts_with('-') {
                        return Err(format!(
                            "missing value after --out; option {value} is not an output path"
                        )
                        .into());
                    }
                    out = Some(PathBuf::from(value));
                }
                "-h" | "--help" => {
                    if args.next().is_some() || out.is_some() {
                        return Err("--help cannot be combined with other arguments".into());
                    }
                    return Ok(None);
                }
                "--model" => {
                    return Err("--model is prohibited: this binary never loads EyeNet".into())
                }
                unknown => return Err(format!("unknown argument: {unknown}").into()),
            }
        }
        Ok(Some(Self {
            out: out.ok_or("missing required --out <new-or-empty-directory>")?,
        }))
    }
}

fn print_help() {
    println!(
        "{}",
        concat!(
            "SRanibro renderer moment-feasibility atlas (research only)\n\n",
            "cargo run --release --features research-synthetic-eye-lab ",
            "--bin synthetic-eye-atlas --\n",
            "  --out <new-or-empty-directory>\n\n",
            "This binary accepts no model or recording input."
        )
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_accepts_only_one_output_argument() {
        let args = Args::parse(["--out".into(), "atlas-output".into()])
            .unwrap()
            .unwrap();
        assert_eq!(args.out, PathBuf::from("atlas-output"));

        assert!(
            Args::parse(["--out".into(), "a".into(), "--out".into(), "b".into()])
                .unwrap_err()
                .to_string()
                .contains("only once")
        );
    }

    #[test]
    fn cli_explicitly_rejects_model_and_recording_inputs() {
        let model_error = Args::parse([
            "--out".into(),
            "atlas-output".into(),
            "--model".into(),
            "model.params".into(),
        ])
        .unwrap_err()
        .to_string();
        assert!(model_error.contains("prohibited"));
        assert!(Args::parse(["--out".into(), "--model".into()])
            .unwrap_err()
            .to_string()
            .contains("not an output path"));

        let recording_error = Args::parse([
            "--out".into(),
            "atlas-output".into(),
            "--recording".into(),
            "session".into(),
        ])
        .unwrap_err()
        .to_string();
        assert!(recording_error.contains("unknown argument"));
    }

    #[test]
    fn cli_requires_output_and_help_is_standalone() {
        assert!(Args::parse([]).unwrap_err().to_string().contains("--out"));
        assert_eq!(Args::parse(["--help".into()]).unwrap(), None);
        assert!(Args::parse(["--out".into(), "a".into(), "--help".into()]).is_err());
    }

    #[test]
    fn build_profile_gate_matches_debug_assertions() {
        if BUILD_PROFILE != "release" || cfg!(debug_assertions) {
            assert!(require_release_build()
                .unwrap_err()
                .to_string()
                .contains("release"));
        } else {
            require_release_build().unwrap();
        }
    }
}
