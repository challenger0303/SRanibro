# Experimental XR5 gaze worker source

The application integration is in `src/ml/xr5_gaze*.rs`. These Python files are the corresponding local worker algorithms. The preserved folder layout matches the worker's imports.

A compatible explicitly selected local `.xr5gaze` package supplies Python, dependency directories, model identity, mapping, device identity and file hashes. No personal mapping, hardware serial, trained model or user package is distributed here. Do not enable Model gaze without a compatible package. Personal-trial accuracy is not a population accuracy result. The Windows release defaults to native gaze; existing opt-in packages remain user-controlled.
