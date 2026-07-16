//! Embed the Windows application icon (assets/sranibro.ico) into the exe so Explorer,
//! the taskbar, and Alt-Tab show it. Best-effort: if the resource compiler isn't found
//! the build still succeeds (just without the embedded icon) rather than failing.

fn main() {
    println!("cargo:rerun-if-changed=assets/sranibro.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/sranibro.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=app icon not embedded ({e})");
        }
    }
}
