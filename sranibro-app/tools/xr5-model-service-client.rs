//! Explicit, collection-only client for the private SRanibro model service.
//!
//! The official service origin is baked into an official collector build. A
//! debug build may override it with `--service-url` for local end-to-end tests.
//! There is deliberately no training call and no model activation in this client.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::time::{SystemTime, UNIX_EPOCH};
use windows_sys::Win32::Foundation::GetLastError;
use windows_sys::Win32::Networking::WinHttp::{
    WinHttpCloseHandle, WinHttpConnect, WinHttpOpen, WinHttpOpenRequest, WinHttpQueryDataAvailable,
    WinHttpQueryHeaders, WinHttpReadData, WinHttpReceiveResponse, WinHttpSendRequest,
    WinHttpSetOption, WinHttpSetTimeouts, WinHttpWriteData, WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
    WINHTTP_FLAG_SECURE, WINHTTP_OPTION_REDIRECT_POLICY, WINHTTP_OPTION_REDIRECT_POLICY_NEVER,
    WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_QUERY_STATUS_CODE,
};
use windows_sys::Win32::Security::Cryptography::{
    BCryptGenRandom, BCRYPT_USE_SYSTEM_PREFERRED_RNG,
};

const MAX_JSON_RESPONSE: usize = 1024 * 1024;
const IDENTITY_PREFIX: &str = "c_";

#[derive(Clone, Debug)]
pub struct ServiceClient {
    origin: Origin,
    contributor_id: String,
}

#[derive(Clone, Debug)]
struct Origin {
    text: String,
    host: String,
    port: u16,
    secure: bool,
}

#[derive(Serialize)]
struct CreateRequest<'a> {
    schema_version: u8,
    contributor_id: &'a str,
    session_id: &'a str,
    device: &'static str,
    session_condition: &'a str,
    collector_version: &'static str,
    consent: Consent,
    archive: ArchiveDeclaration,
}

#[derive(Serialize)]
struct Consent {
    accepted: bool,
    biometric_data: bool,
    model_development: bool,
    model_distribution: bool,
    text_version: &'static str,
}

#[derive(Serialize)]
struct ArchiveDeclaration {
    sha256: String,
    size_bytes: u64,
}

#[derive(Deserialize)]
struct CreateResponse {
    contribution_id: String,
    state: String,
    upload_token: String,
    receipt_token: String,
    deletion_token: String,
}

#[derive(Deserialize)]
struct StatusResponse {
    contribution_id: String,
    state: String,
    sample_count: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UploadReceipt {
    pub schema_version: u8,
    pub service_origin: String,
    pub contribution_id: String,
    pub state: String,
    pub sample_count: Option<usize>,
    pub archive_sha256: String,
    pub receipt_token: String,
    pub deletion_token: String,
    pub uploaded_at_unix_ms: u128,
    pub deletion_note: String,
}

#[derive(Serialize, Deserialize)]
struct PendingUpload {
    schema_version: u8,
    archive_sha256: String,
    idempotency_key: String,
}

struct InternetHandle(*mut c_void);

impl InternetHandle {
    fn new(raw: *mut c_void, operation: &str) -> Result<Self, String> {
        if raw.is_null() {
            Err(winhttp_error(operation))
        } else {
            Ok(Self(raw))
        }
    }
}

impl Drop for InternetHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                WinHttpCloseHandle(self.0);
            }
        }
    }
}

impl ServiceClient {
    pub fn configured(mock: bool) -> Result<Option<Self>, String> {
        if mock {
            return Ok(None);
        }
        let compiled = option_env!("SRANIBRO_MODEL_SERVICE_URL")
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        #[cfg(debug_assertions)]
        let debug_override = command_line_service_url();
        #[cfg(not(debug_assertions))]
        let debug_override: Option<String> = None;
        let Some(value) = debug_override.or(compiled) else {
            return Ok(None);
        };
        let origin = Origin::parse(&value, cfg!(debug_assertions))?;
        let contributor_id = load_or_create_contributor_id()?;
        Ok(Some(Self {
            origin,
            contributor_id,
        }))
    }

    pub fn display_origin(&self) -> &str {
        &self.origin.text
    }

    pub fn upload(
        &self,
        archive_path: &Path,
        session_id: &str,
        session_condition: &str,
        mut progress: impl FnMut(u64, u64),
    ) -> Result<UploadReceipt, String> {
        if !archive_path.is_file() {
            return Err("saved recording no longer exists".into());
        }
        let size_bytes = archive_path
            .metadata()
            .map_err(|error| format!("cannot inspect recording: {error}"))?
            .len();
        if size_bytes == 0 || size_bytes > u32::MAX as u64 {
            return Err("recording size is outside the supported upload range".into());
        }
        let archive_sha256 = sha256_path(archive_path)?;
        let request = CreateRequest {
            schema_version: 1,
            contributor_id: &self.contributor_id,
            session_id,
            device: "pimax_xr5",
            session_condition,
            collector_version: env!("CARGO_PKG_VERSION"),
            consent: Consent {
                accepted: true,
                biometric_data: true,
                model_development: true,
                model_distribution: true,
                text_version: "xr5_dataset_upload_v1",
            },
            archive: ArchiveDeclaration {
                sha256: archive_sha256.clone(),
                size_bytes,
            },
        };
        let body = serde_json::to_vec(&request)
            .map_err(|error| format!("cannot encode upload request: {error}"))?;
        let idempotency = load_or_create_pending_upload(archive_path, &archive_sha256)?;
        let http = WinHttpClient::connect(&self.origin)?;
        let created_json = http.request_bytes(
            "POST",
            "/v1/contributions",
            &[
                ("Content-Type", "application/json".into()),
                ("Accept", "application/json".into()),
                ("Idempotency-Key", idempotency),
            ],
            &body,
        )?;
        let created: CreateResponse = serde_json::from_slice(&created_json)
            .map_err(|error| format!("service returned an invalid create receipt: {error}"))?;
        validate_contribution_id(&created.contribution_id)?;

        let status: StatusResponse = if matches!(created.state.as_str(), "quarantined" | "approved")
        {
            let path = format!("/v1/contributions/{}", created.contribution_id);
            let response = http.request_bytes(
                "GET",
                &path,
                &[
                    ("Accept", "application/json".into()),
                    ("Authorization", format!("Bearer {}", created.receipt_token)),
                ],
                &[],
            )?;
            serde_json::from_slice(&response)
                .map_err(|error| format!("service returned an invalid status receipt: {error}"))?
        } else if created.state == "awaiting_upload" {
            let path = format!("/v1/contributions/{}/archive", created.contribution_id);
            let response = http.put_file(
                &path,
                &[
                    ("Content-Type", "application/zip".into()),
                    ("Accept", "application/json".into()),
                    ("Authorization", format!("Bearer {}", created.upload_token)),
                    ("X-Content-SHA256", archive_sha256.clone()),
                ],
                archive_path,
                size_bytes,
                &mut progress,
            )?;
            serde_json::from_slice(&response)
                .map_err(|error| format!("service returned an invalid upload receipt: {error}"))?
        } else {
            return Err(format!(
                "service cannot accept this recording in state {}",
                created.state
            ));
        };
        if status.contribution_id != created.contribution_id
            || !matches!(status.state.as_str(), "quarantined" | "approved")
        {
            return Err(format!(
                "service did not quarantine the recording (state {})",
                status.state
            ));
        }

        let receipt = UploadReceipt {
            schema_version: 1,
            service_origin: self.origin.text.clone(),
            contribution_id: created.contribution_id,
            state: status.state,
            sample_count: status.sample_count,
            archive_sha256,
            receipt_token: created.receipt_token,
            deletion_token: created.deletion_token,
            uploaded_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            deletion_note: "Keep this receipt if you may ask the maintainer service to delete the stored raw recording. Already released models cannot be retroactively untrained.".into(),
        };
        save_receipt(archive_path, &receipt)?;
        remove_if_present(&pending_upload_path(archive_path))
            .map_err(|error| format!("cannot remove completed upload state: {error}"))?;
        Ok(receipt)
    }
}

impl Origin {
    fn parse(value: &str, allow_loopback_http: bool) -> Result<Self, String> {
        let value = value.trim().trim_end_matches('/');
        let (secure, authority) = if let Some(rest) = value.strip_prefix("https://") {
            (true, rest)
        } else if let Some(rest) = value.strip_prefix("http://") {
            (false, rest)
        } else {
            return Err("model service URL must begin with https://".into());
        };
        if authority.is_empty()
            || authority.contains('/')
            || authority.contains('?')
            || authority.contains('#')
            || authority.contains('@')
        {
            return Err("model service URL must be an origin without a path or login".into());
        }
        let (host, port) = split_authority(authority, secure)?;
        let loopback = matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1");
        if !secure && !(allow_loopback_http && loopback) {
            return Err("biometric uploads require HTTPS".into());
        }
        Ok(Self {
            text: value.to_owned(),
            host,
            port,
            secure,
        })
    }
}

fn split_authority(authority: &str, secure: bool) -> Result<(String, u16), String> {
    let default_port = if secure { 443 } else { 80 };
    if let Some(rest) = authority.strip_prefix('[') {
        let Some(end) = rest.find(']') else {
            return Err("invalid IPv6 service origin".into());
        };
        let host = &rest[..end];
        let suffix = &rest[end + 1..];
        let port = if suffix.is_empty() {
            default_port
        } else {
            suffix
                .strip_prefix(':')
                .ok_or("invalid IPv6 service port")?
                .parse()
                .map_err(|_| "invalid service port")?
        };
        return Ok((host.to_owned(), port));
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') => {
            (host, port.parse().map_err(|_| "invalid service port")?)
        }
        _ => (authority, default_port),
    };
    if host.is_empty()
        || !host
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b".-_".contains(&byte))
    {
        return Err("invalid model service host".into());
    }
    Ok((host.to_owned(), port))
}

struct WinHttpClient {
    origin: Origin,
    _session: InternetHandle,
    connection: InternetHandle,
}

impl WinHttpClient {
    fn connect(origin: &Origin) -> Result<Self, String> {
        let agent = wide("SRanibro-XR5-Dataset-Collector/1")?;
        let host = wide(&origin.host)?;
        let session = InternetHandle::new(
            unsafe {
                WinHttpOpen(
                    agent.as_ptr(),
                    WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                    null(),
                    null(),
                    0,
                )
            },
            "WinHttpOpen",
        )?;
        if unsafe { WinHttpSetTimeouts(session.0, 15_000, 15_000, 600_000, 60_000) } == 0 {
            return Err(winhttp_error("WinHttpSetTimeouts"));
        }
        let connection = InternetHandle::new(
            unsafe { WinHttpConnect(session.0, host.as_ptr(), origin.port, 0) },
            "WinHttpConnect",
        )?;
        Ok(Self {
            origin: origin.clone(),
            _session: session,
            connection,
        })
    }

    fn open_request(&self, method: &str, path: &str) -> Result<InternetHandle, String> {
        if !path.starts_with('/') || path.contains('\r') || path.contains('\n') {
            return Err("invalid service request path".into());
        }
        let method = wide(method)?;
        let path = wide(path)?;
        let request = InternetHandle::new(
            unsafe {
                WinHttpOpenRequest(
                    self.connection.0,
                    method.as_ptr(),
                    path.as_ptr(),
                    null(),
                    null(),
                    null(),
                    if self.origin.secure {
                        WINHTTP_FLAG_SECURE
                    } else {
                        0
                    },
                )
            },
            "WinHttpOpenRequest",
        )?;
        let redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
        if unsafe {
            WinHttpSetOption(
                request.0,
                WINHTTP_OPTION_REDIRECT_POLICY,
                &redirect_policy as *const _ as *const c_void,
                std::mem::size_of_val(&redirect_policy) as u32,
            )
        } == 0
        {
            return Err(winhttp_error("WinHttpSetOption(redirect policy)"));
        }
        Ok(request)
    }

    fn request_bytes(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, String)],
        body: &[u8],
    ) -> Result<Vec<u8>, String> {
        let request = self.open_request(method, path)?;
        let headers = headers_wide(headers)?;
        if body.len() > u32::MAX as usize {
            return Err("request body is too large".into());
        }
        if unsafe {
            WinHttpSendRequest(
                request.0,
                headers.as_ptr(),
                u32::MAX,
                body.as_ptr() as *const c_void,
                body.len() as u32,
                body.len() as u32,
                0,
            )
        } == 0
        {
            return Err(winhttp_error("WinHttpSendRequest"));
        }
        finish_response(&request)
    }

    fn put_file(
        &self,
        path: &str,
        headers: &[(&str, String)],
        file_path: &Path,
        total: u64,
        progress: &mut impl FnMut(u64, u64),
    ) -> Result<Vec<u8>, String> {
        let request = self.open_request("PUT", path)?;
        let headers = headers_wide(headers)?;
        if unsafe {
            WinHttpSendRequest(
                request.0,
                headers.as_ptr(),
                u32::MAX,
                null(),
                0,
                total as u32,
                0,
            )
        } == 0
        {
            return Err(winhttp_error("WinHttpSendRequest(upload)"));
        }
        let mut file = File::open(file_path)
            .map_err(|error| format!("cannot reopen saved recording: {error}"))?;
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut sent = 0u64;
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|error| format!("cannot read saved recording: {error}"))?;
            if count == 0 {
                break;
            }
            let mut offset = 0;
            while offset < count {
                let mut written = 0u32;
                if unsafe {
                    WinHttpWriteData(
                        request.0,
                        buffer[offset..count].as_ptr() as *const c_void,
                        (count - offset) as u32,
                        &mut written,
                    )
                } == 0
                {
                    return Err(winhttp_error("WinHttpWriteData"));
                }
                if written == 0 {
                    return Err("WinHTTP stopped while writing the recording".into());
                }
                offset += written as usize;
                sent += written as u64;
                progress(sent, total);
            }
        }
        if sent != total {
            return Err("saved recording changed during upload".into());
        }
        finish_response(&request)
    }
}

fn finish_response(request: &InternetHandle) -> Result<Vec<u8>, String> {
    if unsafe { WinHttpReceiveResponse(request.0, null_mut()) } == 0 {
        return Err(winhttp_error("WinHttpReceiveResponse"));
    }
    let mut status_code = 0u32;
    let mut status_size = std::mem::size_of_val(&status_code) as u32;
    if unsafe {
        WinHttpQueryHeaders(
            request.0,
            WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            null(),
            &mut status_code as *mut _ as *mut c_void,
            &mut status_size,
            null_mut(),
        )
    } == 0
    {
        return Err(winhttp_error("WinHttpQueryHeaders(status)"));
    }
    let mut response = Vec::new();
    loop {
        let mut available = 0u32;
        if unsafe { WinHttpQueryDataAvailable(request.0, &mut available) } == 0 {
            return Err(winhttp_error("WinHttpQueryDataAvailable"));
        }
        if available == 0 {
            break;
        }
        if response.len().saturating_add(available as usize) > MAX_JSON_RESPONSE {
            return Err("model service response exceeded 1 MiB".into());
        }
        let offset = response.len();
        response.resize(offset + available as usize, 0);
        let mut read = 0u32;
        if unsafe {
            WinHttpReadData(
                request.0,
                response[offset..].as_mut_ptr() as *mut c_void,
                available,
                &mut read,
            )
        } == 0
        {
            return Err(winhttp_error("WinHttpReadData"));
        }
        response.truncate(offset + read as usize);
        if read == 0 {
            break;
        }
    }
    if !(200..300).contains(&status_code) {
        let detail = String::from_utf8_lossy(&response);
        return Err(format!(
            "model service returned HTTP {status_code}: {}",
            detail.chars().take(400).collect::<String>()
        ));
    }
    Ok(response)
}

fn headers_wide(headers: &[(&str, String)]) -> Result<Vec<u16>, String> {
    let mut text = String::new();
    for (name, value) in headers {
        if name.contains(['\r', '\n', ':']) || value.contains(['\r', '\n']) {
            return Err("invalid HTTP header".into());
        }
        text.push_str(name);
        text.push_str(": ");
        text.push_str(value);
        text.push_str("\r\n");
    }
    wide(&text)
}

fn wide(value: &str) -> Result<Vec<u16>, String> {
    if value.contains('\0') {
        return Err("text contains a NUL character".into());
    }
    Ok(value.encode_utf16().chain(std::iter::once(0)).collect())
}

fn sha256_path(path: &Path) -> Result<String, String> {
    let mut input = File::open(path).map_err(|error| format!("cannot open recording: {error}"))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(|error| format!("cannot hash recording: {error}"))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex_digest(&digest.finalize()))
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn load_or_create_contributor_id() -> Result<String, String> {
    let directory = sranibro_rs::config::base_dir().join("model-service");
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("cannot create local service state: {error}"))?;
    let path = directory.join("xr5-contributor-id.txt");
    if let Ok(existing) = std::fs::read_to_string(&path) {
        let existing = existing.trim();
        if validate_contributor_id(existing).is_ok() {
            return Ok(existing.to_owned());
        }
    }
    let random = secure_random::<16>()?;
    let contributor_id = format!("{IDENTITY_PREFIX}{}", hex_digest(&random));
    let partial = path.with_extension("txt.partial");
    {
        let mut output = File::create(&partial)
            .map_err(|error| format!("cannot save contributor ID: {error}"))?;
        writeln!(output, "{contributor_id}")
            .map_err(|error| format!("cannot save contributor ID: {error}"))?;
        output
            .sync_all()
            .map_err(|error| format!("cannot commit contributor ID: {error}"))?;
    }
    std::fs::rename(&partial, &path)
        .map_err(|error| format!("cannot commit contributor ID: {error}"))?;
    Ok(contributor_id)
}

fn load_or_create_pending_upload(
    archive_path: &Path,
    archive_sha256: &str,
) -> Result<String, String> {
    let path = pending_upload_path(archive_path);
    if let Ok(body) = std::fs::read(&path) {
        if let Ok(existing) = serde_json::from_slice::<PendingUpload>(&body) {
            if existing.schema_version == 1
                && existing.archive_sha256 == archive_sha256
                && existing.idempotency_key.len() == 64
                && existing
                    .idempotency_key
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            {
                return Ok(existing.idempotency_key);
            }
        }
    }
    let state = PendingUpload {
        schema_version: 1,
        archive_sha256: archive_sha256.to_owned(),
        idempotency_key: hex_digest(&secure_random::<32>()?),
    };
    let body = serde_json::to_vec_pretty(&state)
        .map_err(|error| format!("cannot encode pending upload state: {error}"))?;
    write_atomic(&path, &body, "pending upload state")?;
    Ok(state.idempotency_key)
}

fn secure_random<const N: usize>() -> Result<[u8; N], String> {
    let mut random = [0u8; N];
    let status = unsafe {
        BCryptGenRandom(
            null_mut(),
            random.as_mut_ptr(),
            random.len() as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    if status != 0 {
        return Err(format!(
            "Windows secure random generation failed: {status:#x}"
        ));
    }
    Ok(random)
}

fn validate_contributor_id(value: &str) -> Result<(), String> {
    if value.len() != IDENTITY_PREFIX.len() + 32
        || !value.starts_with(IDENTITY_PREFIX)
        || !value[IDENTITY_PREFIX.len()..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("invalid pseudonymous contributor ID".into());
    }
    Ok(())
}

fn validate_contribution_id(value: &str) -> Result<(), String> {
    if value.len() != 36
        || value.bytes().enumerate().any(|(index, byte)| match index {
            8 | 13 | 18 | 23 => byte != b'-',
            _ => !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase(),
        })
    {
        return Err("service returned an invalid contribution ID".into());
    }
    Ok(())
}

fn save_receipt(archive_path: &Path, receipt: &UploadReceipt) -> Result<PathBuf, String> {
    let path = archive_path.with_extension("upload-receipt.json");
    let body = serde_json::to_vec_pretty(receipt)
        .map_err(|error| format!("cannot encode upload receipt: {error}"))?;
    write_atomic(&path, &body, "upload receipt")?;
    Ok(path)
}

fn pending_upload_path(archive_path: &Path) -> PathBuf {
    archive_path.with_extension("upload-state.json")
}

fn write_atomic(path: &Path, body: &[u8], label: &str) -> Result<(), String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("tmp");
    let partial = path.with_extension(format!("{extension}.partial"));
    {
        let mut output =
            File::create(&partial).map_err(|error| format!("cannot save {label}: {error}"))?;
        output
            .write_all(body)
            .and_then(|()| output.write_all(b"\n"))
            .and_then(|()| output.sync_all())
            .map_err(|error| format!("cannot save {label}: {error}"))?;
    }
    remove_if_present(path).map_err(|error| format!("cannot replace {label}: {error}"))?;
    std::fs::rename(&partial, &path).map_err(|error| format!("cannot commit {label}: {error}"))?;
    Ok(())
}

fn remove_if_present(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(debug_assertions)]
fn command_line_service_url() -> Option<String> {
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if let Some(value) = argument.strip_prefix("--service-url=") {
            return Some(value.to_owned());
        }
        if argument == "--service-url" {
            return arguments.next();
        }
    }
    None
}

fn winhttp_error(operation: &str) -> String {
    format!("{operation} failed with Windows error {}", unsafe {
        GetLastError()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[test]
    fn production_origin_requires_https() {
        assert!(Origin::parse("https://models.example.com", false).is_ok());
        assert!(Origin::parse("http://models.example.com", true).is_err());
        assert!(Origin::parse("http://127.0.0.1:8787", true).is_ok());
    }

    #[test]
    fn origin_rejects_paths_credentials_and_header_injection() {
        assert!(Origin::parse("https://models.example.com/v1", false).is_err());
        assert!(Origin::parse("https://user@models.example.com", false).is_err());
        assert!(Origin::parse("https://models.example.com\r\nX-Test: 1", false).is_err());
    }

    #[test]
    fn contributor_identifier_is_pseudonymous_and_strict() {
        assert!(validate_contributor_id("c_0123456789abcdef0123456789abcdef").is_ok());
        assert!(validate_contributor_id("serial-VRD02-P407V0D2").is_err());
    }

    #[test]
    fn explicit_upload_round_trip_uses_post_then_streaming_put() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let server_seen = seen.clone();
        let server = std::thread::spawn(move || {
            for request_number in 0..2 {
                let (stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let (line, headers, body) = read_test_request(stream.try_clone().unwrap());
                server_seen.lock().unwrap().push(line.clone());
                if request_number == 0 {
                    assert!(line.starts_with("POST /v1/contributions "));
                    assert!(headers.contains("idempotency-key:"));
                    assert!(String::from_utf8(body)
                        .unwrap()
                        .contains("\"model_development\":true"));
                    write_test_response(
                        stream,
                        201,
                        r#"{"contribution_id":"12345678-1234-1234-1234-123456789abc","state":"awaiting_upload","upload_token":"upload","receipt_token":"receipt","deletion_token":"delete"}"#,
                    );
                } else {
                    assert!(line.starts_with(
                        "PUT /v1/contributions/12345678-1234-1234-1234-123456789abc/archive "
                    ));
                    assert!(headers.contains("authorization: bearer upload"));
                    assert_eq!(body, b"standalone biometric archive fixture");
                    write_test_response(
                        stream,
                        200,
                        r#"{"contribution_id":"12345678-1234-1234-1234-123456789abc","state":"quarantined","sample_count":321,"created_at":"2026-08-02T00:00:00Z","updated_at":"2026-08-02T00:00:01Z"}"#,
                    );
                }
            }
        });

        let root = std::env::temp_dir().join(format!(
            "sranibro-upload-client-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let archive = root.join("recording.zip");
        std::fs::write(&archive, b"standalone biometric archive fixture").unwrap();
        let client = ServiceClient {
            origin: Origin::parse(&format!("http://127.0.0.1:{port}"), true).unwrap(),
            contributor_id: "c_0123456789abcdef0123456789abcdef".into(),
        };
        let mut progress = Vec::new();
        let receipt = client
            .upload(&archive, "s1786000000000", "normal", |sent, total| {
                progress.push((sent, total));
            })
            .unwrap();
        assert_eq!(receipt.state, "quarantined");
        assert_eq!(receipt.sample_count, Some(321));
        assert_eq!(seen.lock().unwrap().len(), 2);
        let payload_len = b"standalone biometric archive fixture".len() as u64;
        assert_eq!(progress.last().copied(), Some((payload_len, payload_len)));
        assert!(archive.with_extension("upload-receipt.json").is_file());
        assert!(!archive.with_extension("upload-state.json").exists());
        server.join().unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    fn read_test_request(stream: TcpStream) -> (String, String, Vec<u8>) {
        let mut reader = BufReader::new(stream);
        let mut first = String::new();
        reader.read_line(&mut first).unwrap();
        let mut headers = String::new();
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
            let lower = line.to_ascii_lowercase();
            if let Some(value) = lower.strip_prefix("content-length:") {
                content_length = value.trim().parse().unwrap();
            }
            headers.push_str(&lower);
        }
        let mut body = vec![0u8; content_length];
        reader.read_exact(&mut body).unwrap();
        (first.trim_end().into(), headers, body)
    }

    fn write_test_response(mut stream: TcpStream, status: u16, body: &str) {
        write!(
            stream,
            "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        stream.flush().unwrap();
    }
}
