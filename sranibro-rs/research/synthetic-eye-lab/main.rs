//! Offline, research-only synthetic stimulus laboratory for the fixed SRanipal EyeNet.
//!
//! This binary is unavailable unless `research-synthetic-eye-lab` is explicitly enabled.
//! It never changes SRanibro configuration or runtime state and never writes model bytes.

mod experiment;
mod luminance;
mod model;
mod moments;
mod output;
mod renderer;

use std::error::Error;
use std::path::{Path, PathBuf};

use experiment::{luminance_match_cases, milestone1_cases, phase0_cases, two_moment_cases};
use model::{canonical_tensor, load_exact, tensor_sha256};
use output::{
    case_id, classify, generated_unix_s, phase0_decision, repository_state, write_run,
    write_two_moment_renderer_no_go, CaseResult, Manifest, Phase0Decision, ANCHOR_PRESENCE_MARGIN,
    MIN_INTERPRETATION_COVERAGE, PRODUCTION_PRESENCE_GATE,
};
use renderer::{apply_photometric, render_stereo};

fn main() {
    if let Err(error) = run() {
        eprintln!("synthetic-eye-lab: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse(std::env::args().skip(1))?;
    // Phase 1.2 is renderer-gated before model bytes are read. A NO-GO is a complete,
    // reportable instrument result and must not fall through to Phase 0 or EyeNet.
    let (two_moment_cases, two_moment_plan) = if args.experiment == Experiment::TwoMomentMatch {
        match two_moment_cases() {
            Ok((cases, plan)) => {
                println!(
                    "TWO-MOMENT RENDERER GO: {} common indices, refs {:?}",
                    plan.common_indices.len(),
                    plan.reference_indices
                );
                (cases, Some(plan))
            }
            Err(reason) => {
                let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
                let (repository_commit, repository_dirty) = repository_state(manifest_dir);
                write_two_moment_renderer_no_go(
                    &args.out,
                    &repository_commit,
                    repository_dirty,
                    &reason,
                )?;
                println!("TWO-MOMENT RENDERER NO-GO: {reason}");
                println!("EyeNet model was not loaded.");
                println!("Results: {}", args.out.display());
                return Ok(());
            }
        }
    } else {
        (Vec::new(), None)
    };
    // Complete Phase 1.1 renderer feasibility before model bytes are even loaded, so
    // model outcomes can never influence its selected aperture range.
    let (luminance_cases, luminance_plan, luminance_renderer_no_go_reason) = if args.experiment
        == Experiment::LuminanceMatch
    {
        match luminance_match_cases() {
            Ok((cases, plan)) => {
                println!(
                        "LUMINANCE RENDERER GO: aperture {:.6}..{:.6} ({} points), target mean {:.9}, D matched {}/{}",
                        plan.selected_aperture_range[0],
                        plan.selected_aperture_range[1],
                        plan.selected_count,
                        plan.constant_mean_target,
                        plan.d_matched_count,
                        plan.selected_count
                    );
                (cases, Some(plan), None)
            }
            Err(reason) => {
                eprintln!("Luminance-match renderer NO-GO: {reason}");
                (Vec::new(), None, Some(reason))
            }
        }
    } else {
        (Vec::new(), None, None)
    };
    let mut model = load_exact(&args.model)?;

    let mut results = evaluate_cases(&mut model.net, phase0_cases())?;
    let decision = phase0_decision(&results);
    let mut suites_evaluated = vec!["phase0"];
    if decision == Phase0Decision::Go {
        match args.experiment {
            Experiment::Milestone1 => {
                results.extend(evaluate_cases(&mut model.net, milestone1_cases())?);
                suites_evaluated.push("milestone1");
            }
            Experiment::LuminanceMatch if luminance_renderer_no_go_reason.is_none() => {
                results.extend(evaluate_cases(&mut model.net, luminance_cases)?);
                suites_evaluated.push("luminance_match");
            }
            Experiment::TwoMomentMatch => {
                results.extend(evaluate_cases_repeated(&mut model.net, two_moment_cases)?);
                suites_evaluated.push("two_moment_match");
            }
            _ => {}
        }
    } else if args.experiment != Experiment::Phase0 {
        eprintln!(
            "Phase 0 was NO-GO; {} suites were not evaluated.",
            args.experiment.as_str()
        );
    }
    if args.experiment == Experiment::LuminanceMatch
        && decision == Phase0Decision::Go
        && luminance_renderer_no_go_reason.is_some()
    {
        eprintln!("Phase 1.1 was not inferred because renderer calibration was NO-GO.");
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (repository_commit, repository_dirty) = repository_state(manifest_dir);
    let recognized_cases = results
        .iter()
        .filter(|result| result.recognition == output::Recognition::RecognizedSynthetic)
        .count();
    let manifest = Manifest {
        tool: "SRanibro synthetic eye stimulus lab",
        suite_version: match args.experiment {
            Experiment::Phase0 => experiment::SUITE_VERSION,
            Experiment::Milestone1 => experiment::MILESTONE1_VERSION,
            Experiment::LuminanceMatch => experiment::LUMINANCE_MATCH_VERSION,
            Experiment::TwoMomentMatch => experiment::TWO_MOMENT_VERSION,
        },
        generated_unix_s: generated_unix_s(),
        repository_commit,
        repository_dirty,
        model_identity_sha256: model.identity_sha256,
        model_bytes_written: false,
        canonical_input: "CHW [2,100,100], left=channel0, right=channel1",
        input_normalization: "u8 / 255 only; no mean/std normalization on EyeNet path",
        presence_gate: PRODUCTION_PRESENCE_GATE,
        anchor_presence_margin: ANCHOR_PRESENCE_MARGIN,
        minimum_interpretation_coverage: MIN_INTERPRETATION_COVERAGE,
        anchor_rule: "at least three preregistered neighboring anchors; every anchor finite, eye-like, and presence >= 0.10",
        requested_experiment: args.experiment.as_str(),
        suites_evaluated,
        luminance_plan,
        luminance_renderer_no_go_reason,
        two_moment_plan,
        two_moment_renderer_no_go_reason: None,
        phase0_decision: decision,
        total_cases: results.len(),
        recognized_cases,
        sanitized_cli: vec![
            "--experiment".into(),
            args.experiment.as_str().into(),
            "--model".into(),
            "<redacted-model-path>".into(),
            "--out".into(),
            "<redacted-output-path>".into(),
        ],
        deterministic_scope: "bit-exact on the tested target/toolchain; cross-platform bit identity is not claimed",
        interpretation_limit: "synthetic renderer-family response only; no real-camera or production-setting generalization",
    };
    write_run(&args.out, &manifest, &results)?;

    println!();
    match decision {
        Phase0Decision::Go => println!(
            "PHASE 0 GO: all preregistered anchors cleared presence >= {:.2}.",
            ANCHOR_PRESENCE_MARGIN
        ),
        Phase0Decision::NoGo => println!(
            "PHASE 0 NO-GO: the preregistered anchor family was not stably recognized; do not interpret openness/squeeze sweeps."
        ),
    }
    println!("Results: {}", args.out.display());
    Ok(())
}

fn evaluate_cases(
    net: &mut sranibro_rs::ml::eye_net::EyeNet,
    cases: Vec<experiment::CaseDefinition>,
) -> Result<Vec<CaseResult>, Box<dyn Error>> {
    evaluate_cases_impl(net, cases, false)
}

fn evaluate_cases_repeated(
    net: &mut sranibro_rs::ml::eye_net::EyeNet,
    cases: Vec<experiment::CaseDefinition>,
) -> Result<Vec<CaseResult>, Box<dyn Error>> {
    evaluate_cases_impl(net, cases, true)
}

fn evaluate_cases_impl(
    net: &mut sranibro_rs::ml::eye_net::EyeNet,
    cases: Vec<experiment::CaseDefinition>,
    repeat_inference: bool,
) -> Result<Vec<CaseResult>, Box<dyn Error>> {
    let mut results = Vec::with_capacity(cases.len());
    for definition in cases {
        let mut rendered = render_stereo(
            &definition.left,
            &definition.right,
            definition.stereo_policy,
        );
        apply_photometric(&mut rendered.left, definition.photometric);
        apply_photometric(&mut rendered.right, definition.photometric);
        let tensor = canonical_tensor(&rendered);
        let tensor_sha256 = tensor_sha256(&tensor);
        let raw = net.forward_one(&tensor);
        let inference_repeat_raw = repeat_inference.then(|| net.forward_one(&tensor));
        let inference_repeat_bits_match = inference_repeat_raw.map(|repeat| {
            raw.iter()
                .zip(repeat)
                .all(|(first, second)| first.to_bits() == second.to_bits())
        });
        let recognition = classify(raw, rendered.left.eye_like && rendered.right.eye_like);
        let case_id = case_id(&definition)?;
        println!(
            "{:<30} {:<28} p={:>8.4} open={:>8.4}/{:>8.4} squeeze={:>8.4}/{:>8.4} {:?}",
            definition.experiment,
            definition.case_name,
            raw[0],
            raw[1],
            raw[2],
            raw[3],
            raw[4],
            recognition
        );
        results.push(CaseResult {
            definition,
            case_id,
            rendered,
            tensor_sha256,
            raw,
            inference_repeat_raw,
            inference_repeat_bits_match,
            recognition,
        });
    }
    Ok(results)
}

#[derive(Debug, PartialEq)]
struct Args {
    model: PathBuf,
    out: PathBuf,
    experiment: Experiment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Experiment {
    Phase0,
    Milestone1,
    LuminanceMatch,
    TwoMomentMatch,
}

impl Experiment {
    fn as_str(self) -> &'static str {
        match self {
            Self::Phase0 => "phase0",
            Self::Milestone1 => "milestone1",
            Self::LuminanceMatch => "luminance-match",
            Self::TwoMomentMatch => "two-moment-match",
        }
    }
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, Box<dyn Error>> {
        let mut model = None;
        let mut out = None;
        let mut experiment = Experiment::Phase0;
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => model = Some(PathBuf::from(next_value(&mut args, "--model")?)),
                "--out" => out = Some(PathBuf::from(next_value(&mut args, "--out")?)),
                "--experiment" => {
                    experiment = match next_value(&mut args, "--experiment")?.as_str() {
                        "phase0" => Experiment::Phase0,
                        "milestone1" => Experiment::Milestone1,
                        "luminance-match" => Experiment::LuminanceMatch,
                        "two-moment-match" => Experiment::TwoMomentMatch,
                        value => return Err(format!("unknown experiment: {value}").into()),
                    }
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                unknown => return Err(format!("unknown argument: {unknown}").into()),
            }
        }
        Ok(Self {
            model: model.ok_or("missing required --model <params file>")?,
            out: out.unwrap_or_else(|| {
                PathBuf::from(match experiment {
                    Experiment::Phase0 => "research-output/synthetic-eye-phase0",
                    Experiment::Milestone1 => "research-output/synthetic-eye-milestone1",
                    Experiment::LuminanceMatch => "research-output/synthetic-eye-luminance-match",
                    Experiment::TwoMomentMatch => "research-output/synthetic-eye-two-moment-match",
                })
            }),
            experiment,
        })
    }
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value after {option}").into())
}

fn print_help() {
    println!(
        "SRanibro synthetic eye lab (research only)\n\n\
         cargo run --release --features research-synthetic-eye-lab --bin synthetic-eye-lab -- \\\n+           --experiment <phase0|milestone1|luminance-match|two-moment-match> --model <EyePrediction params> --out <directory>\n\n\
         The lab is observational and cannot modify production settings."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_requires_a_model_and_defaults_to_phase0() {
        let args = Args::parse(["--model".into(), "model.params".into()]).unwrap();
        assert_eq!(args.model, PathBuf::from("model.params"));
        assert_eq!(
            args.out,
            PathBuf::from("research-output/synthetic-eye-phase0")
        );
        assert_eq!(args.experiment, Experiment::Phase0);
    }

    #[test]
    fn unknown_suites_are_not_silently_enabled() {
        let error = Args::parse([
            "--model".into(),
            "model.params".into(),
            "--experiment".into(),
            "factorial".into(),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("unknown experiment"));
    }

    #[test]
    fn milestone_one_has_its_own_safe_default_output_directory() {
        let args = Args::parse([
            "--model".into(),
            "model.params".into(),
            "--experiment".into(),
            "milestone1".into(),
        ])
        .unwrap();
        assert_eq!(
            args.out,
            PathBuf::from("research-output/synthetic-eye-milestone1")
        );
    }

    #[test]
    fn luminance_match_has_an_independent_safe_default_output_directory() {
        let args = Args::parse([
            "--model".into(),
            "model.params".into(),
            "--experiment".into(),
            "luminance-match".into(),
        ])
        .unwrap();
        assert_eq!(
            args.out,
            PathBuf::from("research-output/synthetic-eye-luminance-match")
        );
        assert_eq!(args.experiment, Experiment::LuminanceMatch);
    }

    #[test]
    fn two_moment_match_has_an_independent_safe_default_output_directory() {
        let args = Args::parse([
            "--model".into(),
            "model.params".into(),
            "--experiment".into(),
            "two-moment-match".into(),
        ])
        .unwrap();
        assert_eq!(
            args.out,
            PathBuf::from("research-output/synthetic-eye-two-moment-match")
        );
        assert_eq!(args.experiment, Experiment::TwoMomentMatch);
    }
}
