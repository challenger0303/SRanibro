# Application source boundary

This directory contains the SRanibro application source: UI, configuration, eyelid and eyebrow inference, post-processing, calibration, output, model-gaze integration, recorder, replay tests and supported SDK adapters.

At the owner's explicit request, the following Tobii forced-access implementation is excluded:

- `src/device/auth.rs`: private authentication material and response implementation (absent).
- The direct EyeChip authentication/subscription handshake in `src/device/vr4_adapter.rs` and the device initialization sequence in `src/device/usb.rs` (replaced with explicit unsupported errors).
- `src/platform.rs`: forced runtime/service/process handoff implementation (replaced with inert boundary functions).
- `src/bundled_tobii.rs` and all `private_*` payloads: embedded vendor-runtime reconstruction/extraction (provider returns None; no payloads are present).

Public builds refuse direct VR4/XR5 EyeChip acquisition before opening hardware. Their profiles, packet decoders, frame processing and recorder code remain reviewable. PSVR2, Varjo and ordinary user-supplied SDK integration are not removed. The official Windows binary retains its private device-access backend; the remainder is built from the same application revision. This checkout alone does not reproduce that excluded backend.

No DLL patch bytes, authentication secrets, vendor DLL, personal configuration, hardware serial, personal gaze mapping or eye recordings are included. SRanipal weights remain user-supplied. Previously released XR5/eyebrow models and their runtimes ship as binary-release assets under their accompanying terms.

The separate private data-collection, annotation and training research projects (`xr5-auto-annotation`, `xr5-eyelid-model`, `xr5-frontalization`) are not application build inputs and are not part of this application-source publication. Their participant records and annotations are not included. Compiled synthetic-eye tooling and the application's data-free model-gaze worker algorithms are included.

The existing application source license in SOURCE-LICENSE.md is retained. The sibling sranibro-core keeps its MIT license. Publishing source does not silently relicense either vendor components or the application.
