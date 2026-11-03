
#![cfg(windows)]
#![allow(non_camel_case_types, non_upper_case_globals, dead_code)]

use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;

use windows_sys::Win32::Foundation::FreeLibrary;
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

#[repr(C)]
pub struct varjo_Session {
    _private: [u8; 0],
}

pub type varjo_Bool = i32;
pub type varjo_Error = i64;
pub type varjo_Nanoseconds = i64;
pub type varjo_StreamId = i64;
pub type varjo_BufferId = i64;
pub type varjo_ChannelFlag = u64;
pub type varjo_ChannelIndex = i64;
pub type varjo_StreamType = i64;
pub type varjo_BufferType = i64;
pub type varjo_DataFlag = u64;
pub type varjo_TextureFormat = i64;

pub const varjo_True: varjo_Bool = 1;
pub const varjo_False: varjo_Bool = 0;
pub const varjo_NoError: varjo_Error = 0;
pub const varjo_InvalidId: varjo_BufferId = -1;

pub const varjo_StreamType_DistortedColor: varjo_StreamType = 1;
pub const varjo_StreamType_EnvironmentCubemap: varjo_StreamType = 2;
pub const varjo_StreamType_EyeCamera: varjo_StreamType = 3;

pub const varjo_ChannelFlag_None: varjo_ChannelFlag = 0;
pub const varjo_ChannelFlag_Left: varjo_ChannelFlag = 1 << 0;
pub const varjo_ChannelFlag_Right: varjo_ChannelFlag = 1 << 1;
pub const varjo_ChannelFlag_All: varjo_ChannelFlag = u64::MAX;

pub const varjo_ChannelIndex_Left: varjo_ChannelIndex = 0;
pub const varjo_ChannelIndex_Right: varjo_ChannelIndex = 1;

pub const varjo_BufferType_CPU: varjo_BufferType = 1;
pub const varjo_BufferType_GPU: varjo_BufferType = 2;

pub const varjo_DataFlag_Buffer: varjo_DataFlag = 1 << 0;
pub const varjo_DataFlag_Intrinsics: varjo_DataFlag = 1 << 1;
pub const varjo_DataFlag_Extrinsics: varjo_DataFlag = 1 << 2;

pub const varjo_TextureFormat_INVALID: varjo_TextureFormat = 0;
pub const varjo_TextureFormat_Y8_UNORM: varjo_TextureFormat = 15;
pub const varjo_TextureFormat_NV12: varjo_TextureFormat = 13;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct varjo_Matrix {
    pub value: [f64; 16],
}

pub const varjo_GazeEyeStatus_Invalid: i64 = 0;
pub const varjo_GazeEyeStatus_Visible: i64 = 1;
pub const varjo_GazeEyeStatus_Compensated: i64 = 2;
pub const varjo_GazeEyeStatus_Tracked: i64 = 3;
pub const varjo_GazeStatus_Invalid: i64 = 0;
pub const varjo_GazeStatus_Adjust: i64 = 1;
pub const varjo_GazeStatus_Valid: i64 = 2;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct varjo_Ray {
    pub origin: [f64; 3],
    pub forward: [f64; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct varjo_Gaze {
    pub left_eye: varjo_Ray,
    pub right_eye: varjo_Ray,
    pub gaze: varjo_Ray,
    pub focus_distance: f64,
    pub stability: f64,
    pub capture_time: varjo_Nanoseconds,
    pub left_status: i64,
    pub right_status: i64,
    pub status: i64,
    pub frame_number: i64,
    pub left_pupil_size: f64,
    pub right_pupil_size: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct varjo_StreamConfig {
    pub stream_id: varjo_StreamId,
    pub channel_flags: varjo_ChannelFlag,
    pub stream_type: varjo_StreamType,
    pub buffer_type: varjo_BufferType,
    pub format: varjo_TextureFormat,
    pub stream_transform: varjo_Matrix,
    pub frame_rate: i32,
    pub width: i32,
    pub height: i32,
    pub row_stride: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct varjo_BufferMetadata {
    pub format: varjo_TextureFormat,
    pub buffer_type: varjo_BufferType,
    pub byte_size: i32,
    pub row_stride: i32,
    pub width: i32,
    pub height: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct varjo_EyeCameraFrameMetadata {
    pub timestamp: varjo_Nanoseconds,
    pub glint_mask_left: u32,
    pub glint_mask_right: u32,
}

#[repr(C)]
pub union varjo_StreamFrameMetadata {
    pub eye_camera: varjo_EyeCameraFrameMetadata,
    pub reserved: [i64; 64],
}

#[repr(C)]
pub struct varjo_StreamFrame {
    pub stream_type: varjo_StreamType,
    pub id: varjo_StreamId,
    pub frame_number: i64,
    pub channels: varjo_ChannelFlag,
    pub data_flags: varjo_DataFlag,
    pub hmd_pose: varjo_Matrix,
    pub metadata: varjo_StreamFrameMetadata,
}

pub const varjo_EventType_Visibility: i64 = 0x1;
pub const varjo_EventType_HeadsetStatus: i64 = 0x4;
pub const varjo_EventType_StandbyStatus: i64 = 0x6;
pub const varjo_EventType_Foreground: i64 = 0x7;
pub const varjo_EventType_DataStreamStop: i64 = 0xB;
pub const VARJO_EVENT_PAYLOAD_OFFSET: usize = 16;

pub type varjo_PropertyKey = i64;
pub const varjo_PropertyKey_UserPresence: varjo_PropertyKey = 0x2000;
pub const varjo_PropertyKey_HMDConnected: varjo_PropertyKey = 0xE001;

#[repr(C, align(8))]
pub struct varjo_Event {
    pub _buf: [u8; 512],
}
impl Default for varjo_Event {
    fn default() -> Self {
        Self { _buf: [0u8; 512] }
    }
}

const _: () = assert!(core::mem::size_of::<varjo_Matrix>() == 128);
const _: () = assert!(core::mem::size_of::<varjo_StreamFrameMetadata>() == 512);
const _: () = assert!(core::mem::size_of::<varjo_BufferMetadata>() == 32);
const _: () = assert!(core::mem::size_of::<varjo_StreamConfig>() == 184);
const _: () = assert!(core::mem::size_of::<varjo_StreamFrame>() == 680);
const _: () = assert!(core::mem::size_of::<varjo_Ray>() == 48);
const _: () = assert!(core::mem::size_of::<varjo_Gaze>() == 216);

pub type varjo_FrameListener = unsafe extern "C" fn(
    frame: *const varjo_StreamFrame,
    session: *mut varjo_Session,
    user_data: *mut c_void,
);

pub type PFN_varjo_IsAvailable = unsafe extern "C" fn() -> varjo_Bool;
pub type PFN_varjo_SessionInit = unsafe extern "C" fn() -> *mut varjo_Session;
pub type PFN_varjo_SessionShutDown = unsafe extern "C" fn(*mut varjo_Session);
pub type PFN_varjo_GetError = unsafe extern "C" fn(*mut varjo_Session) -> varjo_Error;
pub type PFN_varjo_GetDataStreamConfigCount = unsafe extern "C" fn(*mut varjo_Session) -> i32;
pub type PFN_varjo_GetDataStreamConfigs =
    unsafe extern "C" fn(*mut varjo_Session, *mut varjo_StreamConfig, i32);
pub type PFN_varjo_StartDataStream = unsafe extern "C" fn(
    *mut varjo_Session,
    varjo_StreamId,
    varjo_ChannelFlag,
    varjo_FrameListener,
    *mut c_void,
);
pub type PFN_varjo_StopDataStream = unsafe extern "C" fn(*mut varjo_Session, varjo_StreamId);
pub type PFN_varjo_PollEvent =
    unsafe extern "C" fn(*mut varjo_Session, *mut varjo_Event) -> varjo_Bool;
pub type PFN_varjo_GetBufferId = unsafe extern "C" fn(
    *mut varjo_Session,
    varjo_StreamId,
    i64,
    varjo_ChannelIndex,
) -> varjo_BufferId;
pub type PFN_varjo_GetBufferMetadata =
    unsafe extern "C" fn(*mut varjo_Session, varjo_BufferId) -> varjo_BufferMetadata;
pub type PFN_varjo_GetBufferCPUData =
    unsafe extern "C" fn(*mut varjo_Session, varjo_BufferId) -> *mut c_void;
pub type PFN_varjo_LockDataStreamBuffer = unsafe extern "C" fn(*mut varjo_Session, varjo_BufferId);
pub type PFN_varjo_UnlockDataStreamBuffer =
    unsafe extern "C" fn(*mut varjo_Session, varjo_BufferId);
pub type PFN_varjo_GazeInit = unsafe extern "C" fn(*mut varjo_Session);
pub type PFN_varjo_IsGazeAllowed = unsafe extern "C" fn(*mut varjo_Session) -> varjo_Bool;
pub type PFN_varjo_GetGaze = unsafe extern "C" fn(*mut varjo_Session) -> varjo_Gaze;
pub type PFN_varjo_SyncProperties = unsafe extern "C" fn(*mut varjo_Session);
pub type PFN_varjo_HasProperty =
    unsafe extern "C" fn(*mut varjo_Session, varjo_PropertyKey) -> varjo_Bool;
pub type PFN_varjo_GetPropertyBool =
    unsafe extern "C" fn(*mut varjo_Session, varjo_PropertyKey) -> varjo_Bool;

unsafe fn proc<T: Copy>(hmod: *mut c_void, name: &[u8]) -> Option<T> {
    debug_assert_eq!(*name.last().unwrap(), 0, "name must be NUL-terminated");
    GetProcAddress(hmod, name.as_ptr()).map(|f| std::mem::transmute_copy::<_, T>(&f))
}

pub struct VarjoLib {
    hmod: *mut c_void,
    pub is_available: PFN_varjo_IsAvailable,
    pub session_init: PFN_varjo_SessionInit,
    pub session_shutdown: PFN_varjo_SessionShutDown,
    pub get_error: PFN_varjo_GetError,
    pub cfg_count: PFN_varjo_GetDataStreamConfigCount,
    pub get_configs: PFN_varjo_GetDataStreamConfigs,
    pub start_stream: PFN_varjo_StartDataStream,
    pub stop_stream: PFN_varjo_StopDataStream,
    pub poll_event: PFN_varjo_PollEvent,
    pub get_buffer_id: PFN_varjo_GetBufferId,
    pub get_metadata: PFN_varjo_GetBufferMetadata,
    pub lock_buffer: PFN_varjo_LockDataStreamBuffer,
    pub get_cpu_data: PFN_varjo_GetBufferCPUData,
    pub unlock_buffer: PFN_varjo_UnlockDataStreamBuffer,
    pub gaze_init: Option<PFN_varjo_GazeInit>,
    pub is_gaze_allowed: Option<PFN_varjo_IsGazeAllowed>,
    pub get_gaze: Option<PFN_varjo_GetGaze>,
    pub sync_properties: Option<PFN_varjo_SyncProperties>,
    pub has_property: Option<PFN_varjo_HasProperty>,
    pub get_property_bool: Option<PFN_varjo_GetPropertyBool>,
}

impl VarjoLib {
    pub unsafe fn load(dll_path: &str) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};
        let wide: Vec<u16> = std::ffi::OsStr::new(dll_path)
            .encode_wide()
            .chain([0])
            .collect();
        let hmod = LoadLibraryW(wide.as_ptr());
        if hmod.is_null() {
            return Err(Error::new(
                ErrorKind::NotFound,
                format!("Varjo Base runtime load failed: {}", Error::last_os_error()),
            ));
        }
        macro_rules! load {
            ($name:literal, $t:ty) => {
                match proc::<$t>(hmod, concat!($name, "\0").as_bytes()) {
                    Some(f) => f,
                    None => {
                        FreeLibrary(hmod);
                        return Err(Error::new(
                            ErrorKind::NotFound,
                            concat!("Varjo Base runtime missing export ", $name),
                        ));
                    }
                }
            };
        }
        Ok(Self {
            hmod,
            is_available: load!("varjo_IsAvailable", PFN_varjo_IsAvailable),
            session_init: load!("varjo_SessionInit", PFN_varjo_SessionInit),
            session_shutdown: load!("varjo_SessionShutDown", PFN_varjo_SessionShutDown),
            get_error: load!("varjo_GetError", PFN_varjo_GetError),
            cfg_count: load!(
                "varjo_GetDataStreamConfigCount",
                PFN_varjo_GetDataStreamConfigCount
            ),
            get_configs: load!("varjo_GetDataStreamConfigs", PFN_varjo_GetDataStreamConfigs),
            start_stream: load!("varjo_StartDataStream", PFN_varjo_StartDataStream),
            stop_stream: load!("varjo_StopDataStream", PFN_varjo_StopDataStream),
            poll_event: load!("varjo_PollEvent", PFN_varjo_PollEvent),
            get_buffer_id: load!("varjo_GetBufferId", PFN_varjo_GetBufferId),
            get_metadata: load!("varjo_GetBufferMetadata", PFN_varjo_GetBufferMetadata),
            lock_buffer: load!("varjo_LockDataStreamBuffer", PFN_varjo_LockDataStreamBuffer),
            get_cpu_data: load!("varjo_GetBufferCPUData", PFN_varjo_GetBufferCPUData),
            unlock_buffer: load!(
                "varjo_UnlockDataStreamBuffer",
                PFN_varjo_UnlockDataStreamBuffer
            ),
            gaze_init: proc(hmod, b"varjo_GazeInit\0"),
            is_gaze_allowed: proc(hmod, b"varjo_IsGazeAllowed\0"),
            get_gaze: proc(hmod, b"varjo_GetGaze\0"),
            sync_properties: proc(hmod, b"varjo_SyncProperties\0"),
            has_property: proc(hmod, b"varjo_HasProperty\0"),
            get_property_bool: proc(hmod, b"varjo_GetPropertyBool\0"),
        })
    }
}

impl Drop for VarjoLib {
    fn drop(&mut self) {
        unsafe { FreeLibrary(self.hmod) };
    }
}
