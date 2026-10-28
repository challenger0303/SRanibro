# SRanibro application source — v0.1.11-beta

See [SOURCE_BOUNDARY.md](SOURCE_BOUNDARY.md) for the narrow non-public Tobii access boundary and [SOURCE-LICENSE.md](SOURCE-LICENSE.md) for the existing source terms.

Windows requires stable Rust with the MSVC target and Windows SDK. From this directory:

```powershell
cargo test --locked --release --lib
cargo build --locked --release --bin sranibro-rs
cargo test --locked --release --features xr5-only --bin sranibro-xr5-dataset-collector
```

The result is `target/release/sranibro-rs.exe`. Direct EyeChip acquisition is intentionally unavailable in this public checkout. The published Windows download includes the existing private device-access backend. User models/runtimes are not required to compile; they are checked when using their respective tracking paths. Do not copy user settings into distributable packages.

The recorder defaults to a fixed point in SteamVR space while the user moves their head, and saves native Tobii reference directions/status/timing when available. Reference availability is not ground-truth fixation confidence. Hardware eye assignment, coordinate alignment and live accuracy remain separate checks.
