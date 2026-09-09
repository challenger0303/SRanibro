//! Loopback-only HTTP/MJPEG output for the latest mapped eye-camera images.
//!
//! The server never queues camera frames: every client reads the newest telemetry
//! snapshot and skips duplicate generations. JPEG encoding happens only while a
//! client is connected and is capped at 30 Hz per stream.

use std::io::{self, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use jpeg_encoder::{ColorType, Encoder};

use crate::pipeline::{EyeFrame, Telemetry};

const STREAM_PERIOD: Duration = Duration::from_millis(33);
const MAX_CLIENTS: usize = 4;
const BOUNDARY: &str = "sranibro-eye-frame";

/// Running eye-image HTTP server. Dropping it closes the listener and asks every
/// active stream to stop; no frame history is retained.
pub struct EyeImageHttpServer {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    accept: Option<thread::JoinHandle<()>>,
}

impl EyeImageHttpServer {
    pub fn new(host: &str, port: u16, telemetry: Arc<Telemetry>) -> io::Result<Self> {
        let address = loopback_address(host, port)?;
        let listener = TcpListener::bind(address)?;
        let address = listener.local_addr()?;
        listener.set_nonblocking(true)?;

        let stop = Arc::new(AtomicBool::new(false));
        let clients = Arc::new(AtomicUsize::new(0));
        let accept_stop = stop.clone();
        let accept_clients = clients.clone();
        let accept = thread::Builder::new()
            .name("eye-image-http".into())
            .spawn(move || {
                while !accept_stop.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            if accept_clients.load(Ordering::Relaxed) >= MAX_CLIENTS {
                                reject_busy(stream);
                                continue;
                            }
                            accept_clients.fetch_add(1, Ordering::Relaxed);
                            spawn_client(
                                stream,
                                telemetry.clone(),
                                accept_stop.clone(),
                                accept_clients.clone(),
                            );
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(50));
                        }
                        Err(_) => thread::sleep(Duration::from_millis(200)),
                    }
                }
            })?;

        eprintln!("[eye-image] local preview listening on http://{address}/");
        Ok(Self {
            address,
            stop,
            accept: Some(accept),
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.address
    }
}

impl Drop for EyeImageHttpServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(accept) = self.accept.take() {
            let _ = accept.join();
        }
    }
}

fn loopback_address(host: &str, port: u16) -> io::Result<SocketAddr> {
    // Port 0 is accepted by the test-only ephemeral-server path below, but the product
    // UI constrains persisted ports to 1..=65535 so its displayed URL stays exact.
    let ip: IpAddr = host.trim().parse().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "eye-image bind address must be a numeric loopback address such as 127.0.0.1",
        )
    })?;
    if !ip.is_loopback() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "eye-image output is local-only; use 127.0.0.1 or ::1",
        ));
    }
    Ok(SocketAddr::new(ip, port))
}

fn spawn_client(
    stream: TcpStream,
    telemetry: Arc<Telemetry>,
    stop: Arc<AtomicBool>,
    clients: Arc<AtomicUsize>,
) {
    let client_counter = clients.clone();
    let spawned = thread::Builder::new()
        .name("eye-image-client".into())
        .spawn(move || {
            let _guard = ClientGuard(client_counter);
            serve_client(stream, &telemetry, &stop);
        });
    if spawned.is_err() {
        clients.fetch_sub(1, Ordering::Relaxed);
    }
}

struct ClientGuard(Arc<AtomicUsize>);

impl Drop for ClientGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

fn serve_client(mut stream: TcpStream, telemetry: &Telemetry, stop: &AtomicBool) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let path = match read_request_path(&mut stream) {
        Some(path) => path,
        None => {
            write_response(
                &mut stream,
                "403 Forbidden",
                "text/plain; charset=utf-8",
                b"Loopback Host/Origin required\n",
            );
            return;
        }
    };

    match path.as_str() {
        "/" | "/index.html" => write_index(&mut stream),
        "/left.jpg" => write_snapshot(&mut stream, telemetry, 0),
        "/right.jpg" => write_snapshot(&mut stream, telemetry, 1),
        "/left.mjpg" => write_stream(&mut stream, telemetry, stop, 0),
        "/right.mjpg" => write_stream(&mut stream, telemetry, stop, 1),
        _ => write_not_found(&mut stream),
    }
}

fn read_request_path(stream: &mut TcpStream) -> Option<String> {
    let mut request = [0u8; 4096];
    let mut used = 0usize;
    while used < request.len() {
        let read = stream.read(&mut request[used..]).ok()?;
        if read == 0 {
            break;
        }
        used += read;
        if request[..used]
            .windows(4)
            .any(|window| window == b"\r\n\r\n")
            || request[..used].windows(2).any(|window| window == b"\n\n")
        {
            break;
        }
    }
    let text = std::str::from_utf8(&request[..used]).ok()?;
    let mut lines = text.lines();
    let first_line = lines.next()?;
    let mut fields = first_line.split_whitespace();
    if fields.next()? != "GET" {
        return None;
    }
    let path = fields.next()?.split('?').next()?.to_owned();

    let mut host = None;
    let mut origin = None;
    for line in lines {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("host") {
            host = Some(value.trim());
        } else if name.eq_ignore_ascii_case("origin") {
            origin = Some(value.trim());
        }
    }
    if !host.is_some_and(loopback_authority) {
        return None;
    }
    if origin.is_some_and(|value| !loopback_origin(value)) {
        return None;
    }
    Some(path)
}

fn loopback_authority(value: &str) -> bool {
    let value = value.trim();
    if let Some(bracketed) = value.strip_prefix('[') {
        let Some(end) = bracketed.find(']') else {
            return false;
        };
        let host = &bracketed[..end];
        let suffix = &bracketed[end + 1..];
        if !suffix.is_empty()
            && suffix
                .strip_prefix(':')
                .is_none_or(|port| port.parse::<u16>().is_err())
        {
            return false;
        }
        return host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback());
    }

    let host = if value.parse::<IpAddr>().is_ok() {
        value
    } else if let Some((host, port)) = value.rsplit_once(':') {
        if port.parse::<u16>().is_err() {
            return false;
        }
        host
    } else {
        value
    };
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

fn loopback_origin(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    let rest = lower
        .strip_prefix("http://")
        .or_else(|| lower.strip_prefix("https://"));
    rest.and_then(|value| value.split('/').next())
        .is_some_and(loopback_authority)
}

fn write_index(stream: &mut TcpStream) {
    const BODY: &str = "<!doctype html><meta charset=utf-8><title>SRanibro eye images</title>\
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif}main{display:flex;gap:8px;padding:8px}\
figure{margin:0;flex:1}img{width:100%;height:auto;background:#000}figcaption{text-align:center;padding:4px}</style>\
<main><figure><img src=/left.mjpg><figcaption>Left</figcaption></figure>\
<figure><img src=/right.mjpg><figcaption>Right</figcaption></figure></main>";
    write_response(
        stream,
        "200 OK",
        "text/html; charset=utf-8",
        BODY.as_bytes(),
    );
}

fn write_not_found(stream: &mut TcpStream) {
    write_response(
        stream,
        "404 Not Found",
        "text/plain; charset=utf-8",
        b"Not found\n",
    );
}

fn reject_busy(mut stream: TcpStream) {
    write_response(
        &mut stream,
        "503 Service Unavailable",
        "text/plain; charset=utf-8",
        b"Too many eye-image clients\n",
    );
}

fn write_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
}

fn write_snapshot(stream: &mut TcpStream, telemetry: &Telemetry, eye: usize) {
    let Some(frame) = telemetry.stereo_frames()[eye].clone() else {
        write_response(
            stream,
            "503 Service Unavailable",
            "text/plain; charset=utf-8",
            b"No eye frame available yet\n",
        );
        return;
    };
    match encode_jpeg(&frame) {
        Ok(jpeg) => write_response(stream, "200 OK", "image/jpeg", &jpeg),
        Err(error) => write_response(
            stream,
            "500 Internal Server Error",
            "text/plain; charset=utf-8",
            error.to_string().as_bytes(),
        ),
    }
}

fn write_stream(stream: &mut TcpStream, telemetry: &Telemetry, stop: &AtomicBool, eye: usize) {
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary={BOUNDARY}\r\nCache-Control: no-store, no-cache, must-revalidate\r\nPragma: no-cache\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(header.as_bytes()).is_err() {
        return;
    }

    let mut last_generation = 0;
    while !stop.load(Ordering::Relaxed) {
        let frame = telemetry.stereo_frames()[eye].clone();
        if let Some(frame) = frame.filter(|frame| frame.generation != last_generation) {
            let Ok(jpeg) = encode_jpeg(&frame) else {
                return;
            };
            let part = format!(
                "--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\nX-Generation: {}\r\n\r\n",
                jpeg.len(),
                frame.generation
            );
            if stream.write_all(part.as_bytes()).is_err()
                || stream.write_all(&jpeg).is_err()
                || stream.write_all(b"\r\n").is_err()
            {
                return;
            }
            last_generation = frame.generation;
        }
        thread::sleep(STREAM_PERIOD);
    }
}

fn encode_jpeg(frame: &EyeFrame) -> io::Result<Vec<u8>> {
    let width = u16::try_from(frame.width)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "eye frame is too wide"))?;
    let height = u16::try_from(frame.height)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "eye frame is too tall"))?;
    let expected = usize::from(width) * usize::from(height);
    if frame.pixels.len() < expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "eye frame pixel buffer is truncated",
        ));
    }

    let mut jpeg = Vec::with_capacity(expected / 2);
    Encoder::new(&mut jpeg, 80)
        .encode(&frame.pixels[..expected], width, height, ColorType::Luma)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
    Ok(jpeg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_is_strictly_loopback() {
        assert_eq!(
            loopback_address("127.0.0.1", 5556).unwrap(),
            "127.0.0.1:5556".parse().unwrap()
        );
        assert!(loopback_address("::1", 5556).is_ok());
        assert!(loopback_address("0.0.0.0", 5556).is_err());
        assert!(loopback_address("192.168.1.2", 5556).is_err());
        assert!(loopback_address("localhost", 5556).is_err());
    }

    #[test]
    fn request_authority_rejects_dns_rebinding_and_remote_origins() {
        for local in [
            "localhost",
            "localhost:5556",
            "127.0.0.1",
            "127.0.0.1:5556",
            "[::1]",
            "[::1]:5556",
        ] {
            assert!(loopback_authority(local), "{local}");
        }
        assert!(!loopback_authority("eye.example"));
        assert!(!loopback_authority("localhost.example"));
        assert!(loopback_origin("http://127.0.0.1:5556"));
        assert!(!loopback_origin("https://eye.example"));
        assert!(!loopback_origin("null"));
    }

    #[test]
    fn grayscale_frame_encodes_as_jpeg_without_history() {
        let frame = EyeFrame {
            generation: 9,
            width: 4,
            height: 3,
            pixels: Arc::from([0, 32, 64, 96, 128, 160, 192, 224, 255, 192, 96, 0]),
        };
        let jpeg = encode_jpeg(&frame).unwrap();
        assert!(jpeg.starts_with(&[0xff, 0xd8]));
        assert!(jpeg.ends_with(&[0xff, 0xd9]));
    }

    #[test]
    fn truncated_frame_is_rejected() {
        let frame = EyeFrame {
            generation: 1,
            width: 4,
            height: 4,
            pixels: Arc::from([0; 15]),
        };
        assert_eq!(
            encode_jpeg(&frame).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn snapshot_is_served_and_drop_releases_the_port() {
        let telemetry = Telemetry::new(
            false,
            false,
            false,
            &crate::core::types::DeviceProfile::default(),
        );
        telemetry.frames.lock().unwrap()[0] = Some(EyeFrame {
            generation: 1,
            width: 4,
            height: 4,
            pixels: Arc::from([128; 16]),
        });

        let server = EyeImageHttpServer::new("127.0.0.1", 0, telemetry).unwrap();
        let address = server.local_addr();
        let mut client = TcpStream::connect(address).unwrap();
        client
            .write_all(b"GET /left.jpg HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        assert!(response.starts_with(b"HTTP/1.1 200 OK\r\n"));
        assert!(response
            .windows(b"Content-Type: image/jpeg".len())
            .any(|window| window == b"Content-Type: image/jpeg"));
        assert!(response.windows(2).any(|window| window == [0xff, 0xd8]));

        drop(server);
        TcpListener::bind(address).expect("server drop must release its listener");
    }

    #[test]
    fn remote_host_or_origin_cannot_read_eye_images() {
        let telemetry = Telemetry::new(
            false,
            false,
            false,
            &crate::core::types::DeviceProfile::default(),
        );
        let server = EyeImageHttpServer::new("127.0.0.1", 0, telemetry).unwrap();
        let address = server.local_addr();

        for request in [
            b"GET /left.jpg HTTP/1.1\r\nHost: eye.example\r\n\r\n".as_slice(),
            b"GET /left.jpg HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://eye.example\r\n\r\n"
                .as_slice(),
        ] {
            let mut client = TcpStream::connect(address).unwrap();
            client.write_all(request).unwrap();
            let mut response = Vec::new();
            client.read_to_end(&mut response).unwrap();
            assert!(
                response.starts_with(b"HTTP/1.1 403 Forbidden\r\n"),
                "{}",
                String::from_utf8_lossy(&response)
            );
        }

        drop(server);
    }
}
