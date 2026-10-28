# SRanibro application source — v0.1.11-beta

Application UI, configuration, eye processing, calibration, model-gaze integration and recorder source. See [SOURCE-LICENSE.md](SOURCE-LICENSE.md) for the source terms.

Windows requires stable Rust with the MSVC target and Windows SDK. From this directory:

```powershell
cargo test --locked --release --lib
cargo build --locked --release --bin sranibro-rs
cargo test --locked --release --features xr5-only --bin sranibro-xr5-dataset-collector
```

The result is `target/release/sranibro-rs.exe`. Available tracking paths depend on the configured adapters and runtime dependencies. For ready-to-run Windows packages, use the [official releases](https://github.com/challenger0303/SRanibro/releases). User models/runtimes are checked when using their respective tracking paths. Do not copy user settings into distributable packages.

The recorder defaults to a fixed point in SteamVR space while the user moves their head, and saves native Tobii reference directions/status/timing when available. Reference availability is not ground-truth fixation confidence. Hardware eye assignment, coordinate alignment and live accuracy remain separate checks.
