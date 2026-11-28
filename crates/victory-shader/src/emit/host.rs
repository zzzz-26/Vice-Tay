//! Host D3D12 checks for emitted shaders: pipeline creation and, for
//! tests, drawing a full-screen triangle and reading the target back.

use std::ffi::CString;

use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::{Interface, PCSTR};

use super::HostShader;
use super::container::SigElement;
use crate::dxbc::{Container, FourCC};

pub struct Host {
    pub device: ID3D12Device,
}

/// What the stage declares that a pipeline must match.
fn sig(dxil: &[u8], fourcc: FourCC) -> Result<Vec<SigElement>, String> {
    let c = Container::parse(dxil).map_err(|e| e.to_string())?;
    match c.part(fourcc) {
        Some(p) => super::signature(p.data).map_err(|e| e.to_string()),
        None => Ok(Vec::new()),
    }
}

fn format(e: &SigElement) -> DXGI_FORMAT {
    let i = usize::from(e.cols.clamp(1, 4)) - 1;
    match e.comp {
        9 => [
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R32G32_FLOAT,
            DXGI_FORMAT_R32G32B32_FLOAT,
            DXGI_FORMAT_R32G32B32A32_FLOAT,
        ][i],
        4 => [
            DXGI_FORMAT_R32_SINT,
            DXGI_FORMAT_R32G32_SINT,
            DXGI_FORMAT_R32G32B32_SINT,
            DXGI_FORMAT_R32G32B32A32_SINT,
        ][i],
        _ => [
            DXGI_FORMAT_R32_UINT,
            DXGI_FORMAT_R32G32_UINT,
            DXGI_FORMAT_R32G32B32_UINT,
            DXGI_FORMAT_R32G32B32A32_UINT,
        ][i],
    }
}

fn target_format(e: &SigElement) -> DXGI_FORMAT {
    match e.comp {
        9 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        4 => DXGI_FORMAT_R32G32B32A32_SINT,
        _ => DXGI_FORMAT_R32G32B32A32_UINT,
    }
}

fn err(e: windows::core::Error) -> String {
    e.message().to_string()
}

fn software(desc: &DXGI_ADAPTER_DESC1) -> bool {
    desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
        || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c)
}

fn adapter_matches(requested: &str, index: u32, name: &str) -> bool {
    if let Ok(wanted) = requested.parse::<u32>() {
        wanted == index
    } else {
        !requested.is_empty() && name.to_lowercase().contains(&requested.to_lowercase())
    }
}

fn adapter_desc(device: &ID3D12Device) -> windows::core::Result<DXGI_ADAPTER_DESC1> {
    // SAFETY: plain adapter queries for a live device.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        factory
            .EnumAdapterByLuid::<IDXGIAdapter1>(device.GetAdapterLuid())?
            .GetDesc1()
    }
}

fn shader_model(device: &ID3D12Device) -> windows::core::Result<D3D_SHADER_MODEL> {
    let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_6,
    };
    // SAFETY: a feature structure of the requested type and size.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&raw mut sm).cast(),
            size_of_val(&sm) as u32,
        )?;
    }
    Ok(sm.HighestShaderModel)
}

/// Create the device used by the runtime and offline pipeline checks.
/// An explicit DXGI index or name in `VICTORY_D3D12_ADAPTER` takes priority.
/// Otherwise prefer a high-performance SM 6.6 adapter, then try DXGI's
/// default if there is no suitable enumerated adapter.
pub fn create_preferred_device() -> windows::core::Result<ID3D12Device> {
    let requested = std::env::var("VICTORY_D3D12_ADAPTER").unwrap_or_default();
    let requested = requested.trim();
    // SAFETY: plain adapter queries and device creation.
    unsafe {
        let create = |adapter: Option<&IDXGIAdapter1>| -> windows::core::Result<ID3D12Device> {
            let mut device = None;
            match adapter {
                Some(adapter) => D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?,
                None => D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?,
            }
            device.ok_or_else(windows::core::Error::empty)
        };
        if !requested.is_empty() {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            for index in 0.. {
                let Ok(adapter) = factory.EnumAdapters1(index) else {
                    break;
                };
                let desc = adapter.GetDesc1()?;
                let name = String::from_utf16_lossy(&desc.Description);
                if !software(&desc) && adapter_matches(requested, index, name.trim_end_matches('\0'))
                {
                    return create(Some(&adapter));
                }
            }
            return Err(windows::core::Error::new(
                DXGI_ERROR_NOT_FOUND,
                format!("VICTORY_D3D12_ADAPTER={requested} matched no hardware adapter"),
            ));
        }
        if let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory6>(DXGI_CREATE_FACTORY_FLAGS(0)) {
            for index in 0.. {
                let Ok(adapter) = factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                    index,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                ) else {
                    break;
                };
                if adapter.GetDesc1().is_ok_and(|desc| !software(&desc))
                    && let Ok(device) = create(Some(&adapter))
                    && shader_model(&device).is_ok_and(|sm| sm.0 >= D3D_SHADER_MODEL_6_6.0)
                {
                    return Ok(device);
                }
            }
        }
        create(None)
    }
}

#[cfg(test)]
#[test]
fn an_explicit_adapter_matches_an_index_or_name() {
    let nvidia = "NVIDIA GeForce RTX 3050 Laptop GPU";
    let integrated = "Intel(R) UHD Graphics";
    assert!(adapter_matches("1", 1, nvidia));
    assert!(!adapter_matches("1", 0, integrated));
    assert!(adapter_matches("geforce rtx 3050", 1, nvidia));
    assert!(!adapter_matches("nvidia", 0, integrated));
    assert!(!adapter_matches("", 0, integrated));
}

impl Host {
    #[cfg(test)]
    pub(super) fn check_hardware_validation(&self) {
        if std::env::var_os("VICTORY_D3D12_DEBUG").is_none() {
            return;
        }
        let queue: ID3D12InfoQueue = self.device.cast().expect("D3D12 debug layer");
        let (mut errors, mut warnings) = (0, 0);
        // SAFETY: storage is aligned for the message and sized for its inline text.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                queue.GetMessage(i, None, &mut size).unwrap();
                let mut storage = vec![0u64; size.div_ceil(8)];
                let message = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();
                queue.GetMessage(i, Some(message), &mut size).unwrap();
                let message = &*message;
                if message.Severity.0 <= D3D12_MESSAGE_SEVERITY_WARNING.0 {
                    let text = std::slice::from_raw_parts(message.pDescription, message.DescriptionByteLength);
                    eprintln!("D3D12 #{}: {}", message.ID.0, String::from_utf8_lossy(text).trim_end_matches('\0'));
                    if message.Severity == D3D12_MESSAGE_SEVERITY_WARNING { warnings += 1 } else { errors += 1 }
                }
            }
            assert_eq!(queue.GetNumMessagesDiscardedByMessageCountLimit(), 0);
        }
        eprintln!("D3D12 hardware validation: {errors} errors, {warnings} warnings");
        assert_eq!(errors, 0);
    }

    /// The preferred hardware adapter, or WARP with `warp` (or `VICTORY_WARP=1`).
    pub fn new(warp: bool) -> Result<Self, String> {
        let warp = warp || std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
        // SAFETY: plain D3D12/DXGI calls with valid out pointers.
        unsafe {
            // `VICTORY_D3D12_DEBUG`: the debug layer, whose messages a
            // refused pipeline's error then carries.
            if std::env::var_os("VICTORY_D3D12_DEBUG").is_some() {
                let mut debug: Option<ID3D12Debug> = None;
                if D3D12GetDebugInterface(&mut debug).is_ok()
                    && let Some(debug) = debug
                {
                    debug.EnableDebugLayer();
                    #[cfg(test)]
                    if !warp && std::env::var_os("VICTORY_NVIDIA_HARDWARE_TESTS").is_some_and(|v| v == "1") {
                        debug.cast::<ID3D12Debug1>().expect("GPU validation interface").SetEnableGPUBasedValidation(true);
                    }
                }
            }
            let mut device: Option<ID3D12Device> = None;
            if warp {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(err)?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter().map_err(err)?;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).map_err(err)?;
            } else {
                let hardware = create_preferred_device().map_err(err)?;
                // A machine without a graphics card gives the software
                // adapter as its default device (a hosted test runner,
                // whose first adapter is a display stub that cannot make a
                // device): that is no hardware device, and the tests that
                // ask for one skip it, as they do where there is none.
                if adapter_desc(&hardware).is_ok_and(|desc| software(&desc)) {
                    return Err("the default device is the software adapter".to_owned());
                }
                device = Some(hardware);
            }
            let device = device.ok_or("no device")?;
            let sm = shader_model(&device).map_err(err)?;
            if sm.0 < D3D_SHADER_MODEL_6_6.0 {
                return Err(format!("the device supports SM {:#x} only", sm.0));
            }
            Ok(Host { device })
        }
    }

    /// The name of the adapter that actually created the device.
    pub fn adapter_name(&self) -> Option<String> {
        let desc = adapter_desc(&self.device).ok()?;
        Some(String::from_utf16_lossy(&desc.Description).trim_end_matches('\0').to_owned())
    }

    /// The maximum wave size and whether the adapter is AMD RDNA, for
    /// offline shader translation followed by host pipeline creation.
    pub fn translation_caps(&self) -> Option<(u32, bool)> {
        let mut waves = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        // SAFETY: a feature structure of the requested type and size.
        unsafe {
            self.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut waves).cast(),
                size_of_val(&waves) as u32,
            )
        }
        .ok()?;
        let rdna = waves.WaveLaneCountMin == 32
            && waves.WaveLaneCountMax == 64
            && adapter_desc(&self.device).is_ok_and(|desc| desc.VendorId == 0x1002);
        Some((waves.WaveLaneCountMax, rdna))
    }

    /// The root signature translated shaders expect: user data per stage
    /// at `b0`, immediate samplers per stage at `s<n>`, heaps directly
    /// indexed.
    pub fn root_signature(
        &self,
        vs: &HostShader,
        ps: &HostShader,
    ) -> Result<ID3D12RootSignature, String> {
        self.root_signature_for(&[
            (vs, D3D12_SHADER_VISIBILITY_VERTEX),
            (ps, D3D12_SHADER_VISIBILITY_PIXEL),
        ])
    }

    /// The root signature for these stages: user data of stage k is root
    /// parameter k, root views follow.
    pub fn root_signature_for(
        &self,
        stages: &[(&HostShader, D3D12_SHADER_VISIBILITY)],
    ) -> Result<ID3D12RootSignature, String> {
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = stages
            .iter()
            .map(|&(s, vis)| D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: super::USER_DATA_SPACE,
                        Num32BitValues: s.user_data_dwords().max(1),
                    },
                },
                ShaderVisibility: vis,
            })
            .collect();
        // By slot, then kind: 0 CBV, 1 SRV, 2 UAV.
        let mut views: Vec<(u8, u8)> = stages
            .iter()
            .flat_map(|(s, _)| {
                let cbvs = s.root_cbvs.iter().map(|&r| (r, 0));
                let srvs = s.root_srvs.iter().map(|&r| (r, 1));
                cbvs.chain(srvs).chain(s.root_uavs.iter().map(|&r| (r, 2)))
            })
            .collect();
        views.sort_unstable();
        views.dedup();
        for (slot, kind) in views {
            params.push(D3D12_ROOT_PARAMETER1 {
                ParameterType: match kind {
                    1 => D3D12_ROOT_PARAMETER_TYPE_SRV,
                    2 => D3D12_ROOT_PARAMETER_TYPE_UAV,
                    _ => D3D12_ROOT_PARAMETER_TYPE_CBV,
                },
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: u32::from(slot),
                        RegisterSpace: super::ROOT_CBV_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
        // SAFETY: the feature query writes the structure of the given size.
        // Older runtimes may not recognize OPTIONS19; that means no support.
        let aniso_point_mip_supported = unsafe {
            self.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS19,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        }
        .is_ok()
            && options.AnisoFilterWithPointMipSupported.as_bool();
        let mut samplers = Vec::new();
        for &(s, vis) in stages {
            for (n, d) in s.static_samplers.iter().enumerate() {
                let [u, v, w] = d.address.map(|a| D3D12_TEXTURE_ADDRESS_MODE(a as i32));
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(d.filter_for_host(aniso_point_mip_supported) as i32),
                    AddressU: u,
                    AddressV: v,
                    AddressW: w,
                    MipLODBias: d.mip_lod_bias,
                    MaxAnisotropy: d.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(d.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(d.border as i32),
                    MinLOD: d.min_lod,
                    MaxLOD: d.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: super::SAMPLER_SPACE,
                    ShaderVisibility: vis,
                });
            }
        }
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: samplers.len() as u32,
                    pStaticSamplers: samplers.as_ptr(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                        | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                        | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
                },
            },
        };
        // SAFETY: the description points at live arrays.
        unsafe {
            let mut blob = None;
            let mut error = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error))
                .map_err(err)?;
            let blob: ID3DBlob = blob.ok_or("no root signature blob")?;
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            self.device.CreateRootSignature(0, bytes).map_err(err)
        }
    }

    /// A compute pipeline; user data is root parameter 0.
    pub fn compute_pipeline(
        &self,
        cs: &HostShader,
    ) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
        let root = self.root_signature_for(&[(cs, D3D12_SHADER_VISIBILITY_ALL)])?;
        let pso = self.compute_pipeline_with(&root, &cs.dxil)?;
        Ok((root, pso))
    }

    pub fn compute_pipeline_with(
        &self,
        root: &ID3D12RootSignature,
        cs: &[u8],
    ) -> Result<ID3D12PipelineState, String> {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the descriptor never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: cs.as_ptr().cast(),
                BytecodeLength: cs.len(),
            },
            ..Default::default()
        };
        // SAFETY: the description points at live data.
        unsafe { self.device.CreateComputePipelineState(&desc) }.map_err(|e| self.explained(e))
    }

    /// The error with the debug layer's messages since the last call, if
    /// the layer is on (`VICTORY_D3D12_DEBUG`).
    fn explained(&self, e: windows::core::Error) -> String {
        let mut text = err(e);
        let Ok(queue) = self.device.cast::<ID3D12InfoQueue>() else {
            return text;
        };
        // SAFETY: each message is read into a buffer of the size the queue
        // asks for, aligned for the message header.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut len = 0usize;
                let _ = queue.GetMessage(i, None, &mut len);
                let mut buffer = vec![0u64; len.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut len).is_ok() {
                    let m = &*message;
                    let bytes = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength.saturating_sub(1));
                    text.push_str(" | ");
                    text.push_str(&String::from_utf8_lossy(bytes));
                }
            }
            queue.ClearStoredMessages();
        }
        text
    }

    /// A graphics pipeline from a vertex and a pixel shader, with float or
    /// integer 32-bit targets as the pixel shader writes them.
    pub fn pipeline(
        &self,
        vs: &HostShader,
        ps: &HostShader,
    ) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
        let root = self.root_signature(vs, ps)?;
        let pso = self.pipeline_with(&root, &vs.dxil, &ps.dxil)?;
        Ok((root, pso))
    }

    /// A graphics pipeline with the given root signature; vertex inputs
    /// packed in register order.
    pub fn pipeline_with(
        &self,
        root: &ID3D12RootSignature,
        vs: &[u8],
        ps: &[u8],
    ) -> Result<ID3D12PipelineState, String> {
        self.pipeline_with_gs(root, vs, None, ps)
    }

    /// As [`Host::pipeline_with`], with a geometry shader between the two
    /// (the rectangle stage, `rect_list_gs`).
    pub fn pipeline_with_gs(
        &self,
        root: &ID3D12RootSignature,
        vs: &[u8],
        gs: Option<&[u8]>,
        ps: &[u8],
    ) -> Result<ID3D12PipelineState, String> {
        let inputs = sig(vs, FourCC::ISG1)?;
        let names: Vec<CString> = inputs
            .iter()
            .map(|e| CString::new(e.name.clone()).unwrap())
            .collect();
        let layout: Vec<D3D12_INPUT_ELEMENT_DESC> = inputs
            .iter()
            .zip(&names)
            .filter(|(e, _)| e.kind == 0)
            .map(|(e, n)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(n.as_ptr().cast()),
                SemanticIndex: e.index,
                Format: format(e),
                InputSlot: 0,
                AlignedByteOffset: D3D12_APPEND_ALIGNED_ELEMENT,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            })
            .collect();
        let outputs = sig(ps, FourCC::OSG1)?;
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the descriptor never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(root) },
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs.as_ptr().cast(),
                BytecodeLength: vs.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps.as_ptr().cast(),
                BytecodeLength: ps.len(),
            },
            GS: gs.map_or(D3D12_SHADER_BYTECODE::default(), |gs| D3D12_SHADER_BYTECODE {
                pShaderBytecode: gs.as_ptr().cast(),
                BytecodeLength: gs.len(),
            }),
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: layout.as_ptr(),
                NumElements: layout.len() as u32,
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        for rt in &mut desc.BlendState.RenderTarget {
            rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        }
        for o in outputs.iter().filter(|o| o.kind == 16) {
            let n = o.row.unwrap_or(0) as usize;
            desc.RTVFormats[n] = target_format(o);
            desc.NumRenderTargets = desc.NumRenderTargets.max(n as u32 + 1);
        }
        if outputs.iter().any(|o| matches!(o.kind, 17..=19)) {
            desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
            desc.DepthStencilState.DepthEnable = true.into();
            desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
            desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        }
        if outputs.iter().any(|o| o.kind == 20) {
            desc.DSVFormat = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
            desc.DepthStencilState.StencilEnable = true.into();
        }
        // SAFETY: the description points at live data.
        let pso = unsafe { self.device.CreateGraphicsPipelineState(&desc) };
        pso.map_err(err)
    }

    /// A mesh pipeline from a mesh and a pixel shader (user data of the mesh
    /// stage is root parameter 0, the pixel stage's 1).
    pub fn mesh_pipeline(
        &self,
        ms: &HostShader,
        ps: &HostShader,
    ) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
        let root = self.root_signature_for(&[
            (ms, D3D12_SHADER_VISIBILITY_MESH),
            (ps, D3D12_SHADER_VISIBILITY_PIXEL),
        ])?;
        let pso = self.mesh_pipeline_with(&root, &ms.dxil, &ps.dxil)?;
        Ok((root, pso))
    }

    /// A mesh pipeline with the given root signature: the pipeline-state
    /// stream form, with the targets [`Host::pipeline_with`] would use.
    pub fn mesh_pipeline_with(
        &self,
        root: &ID3D12RootSignature,
        ms: &[u8],
        ps: &[u8],
    ) -> Result<ID3D12PipelineState, String> {
        /// A stream subobject: its type, then its description, pointer-aligned.
        #[repr(C, align(8))]
        struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
        #[repr(C)]
        struct Stream {
            root: Sub<*mut std::ffi::c_void>,
            ms: Sub<D3D12_SHADER_BYTECODE>,
            ps: Sub<D3D12_SHADER_BYTECODE>,
            blend: Sub<D3D12_BLEND_DESC>,
            mask: Sub<u32>,
            raster: Sub<D3D12_RASTERIZER_DESC>,
            depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
            targets: Sub<D3D12_RT_FORMAT_ARRAY>,
            dsv: Sub<DXGI_FORMAT>,
            samples: Sub<DXGI_SAMPLE_DESC>,
        }
        let outputs = sig(ps, FourCC::OSG1)?;
        let mut blend = D3D12_BLEND_DESC::default();
        for rt in &mut blend.RenderTarget {
            rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        }
        let mut targets = D3D12_RT_FORMAT_ARRAY::default();
        for o in outputs.iter().filter(|o| o.kind == 16) {
            let n = o.row.unwrap_or(0) as usize;
            targets.RTFormats[n] = target_format(o);
            targets.NumRenderTargets = targets.NumRenderTargets.max(n as u32 + 1);
        }
        let mut depth = D3D12_DEPTH_STENCIL_DESC::default();
        let mut dsv = DXGI_FORMAT_UNKNOWN;
        if outputs.iter().any(|o| matches!(o.kind, 17..=19)) {
            dsv = DXGI_FORMAT_D32_FLOAT;
            depth.DepthEnable = true.into();
            depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
            depth.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        }
        if outputs.iter().any(|o| o.kind == 20) {
            dsv = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
            depth.StencilEnable = true.into();
        }
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: b.as_ptr().cast(),
            BytecodeLength: b.len(),
        };
        let mut stream = Stream {
            root: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
                root.as_raw(),
            ),
            ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, code(ms)),
            ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, code(ps)),
            blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, blend),
            mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, u32::MAX),
            raster: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
                D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
            ),
            depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, depth),
            targets: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
                targets,
            ),
            dsv: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
                dsv,
            ),
            samples: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
                DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
            ),
        };
        let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
            SizeInBytes: size_of::<Stream>(),
            pPipelineStateSubobjectStream: (&raw mut stream).cast(),
        };
        let device: ID3D12Device2 = self.device.cast().map_err(err)?;
        // SAFETY: the stream points at live data and outlives the call.
        unsafe { device.CreatePipelineState(&desc) }.map_err(err)
    }
}

/// A vertex shader that feeds `ps`'s inputs (zeros) and a position.
pub fn stub_vs(ps: &HostShader) -> Result<HostShader, String> {
    let inputs = sig(&ps.dxil, FourCC::ISG1)?;
    let mut outputs: Vec<SigElement> = inputs
        .into_iter()
        .filter(|e| matches!(e.kind, 0 | 3 | 6 | 7))
        .collect();
    if !outputs.iter().any(|e| e.kind == 3) {
        let row = outputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        outputs.push(SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        });
    }
    Ok(stub(false, outputs))
}

/// A pixel shader that reads nothing and writes target 0.
pub fn stub_ps() -> HostShader {
    let target = SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    stub(true, vec![target])
}

fn stub(pixel: bool, outputs: Vec<SigElement>) -> HostShader {
    use super::bitcode::{Attr, Term};
    let mut s = super::shader::Shader::new(pixel);
    let b = s.entry;
    for (id, e) in outputs.iter().enumerate() {
        for col in 0..e.cols {
            let (suffix, zero) = if e.comp == 9 {
                (".f32", s.m.f32(0))
            } else {
                (".i32", s.m.i32(0))
            };
            let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), zero];
            s.op(
                b,
                &format!("storeOutput{suffix}"),
                Attr::NoUnwind,
                s.t.void,
                5,
                &args,
            );
        }
    }
    s.outputs = outputs
        .into_iter()
        .map(|mut e| {
            e.usage = (1 << e.cols) - 1;
            e
        })
        .collect();
    s.m.terminate(b, Term::Ret);
    HostShader {
        stage: if pixel {
            crate::native::HwStage::Ps
        } else {
            crate::native::HwStage::Vs
        },
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    }
}

/// A root argument for [`Host::draw`].
#[cfg(test)]
pub enum Root {
    Constants(Vec<u32>),
    /// A table starting at this heap index.
    Table {
        sampler: bool,
        index: usize,
    },
    /// A root CBV (or with `srv`, a root SRV) over a buffer of these
    /// dwords ([`Host::dispatch`] only).
    View {
        srv: bool,
        data: Vec<u32>,
    },
}

/// Bindings for [`Host::draw`]: root arguments and the heap contents.
#[cfg(test)]
pub struct Draw<'a> {
    pub roots: Vec<(u32, Root)>,
    /// RGBA32F textures (width, height, texels) at view-heap indices 0..
    pub textures: &'a [(u32, u32, &'a [[f32; 4]])],
    /// Samplers at sampler-heap indices 0..
    pub samplers: &'a [D3D12_SAMPLER_DESC],
    /// Three vertices of stride bytes (none for position-free layouts).
    pub vertices: &'a [f32],
    pub stride: u32,
    pub size: (u32, u32),
    /// Mesh thread groups to dispatch instead of drawing (0: draw).
    pub mesh_groups: u32,
}

#[cfg(test)]
impl Host {
    /// Draws one triangle into an RGBA32F target and returns its texels.
    pub fn draw(
        &self,
        root: &ID3D12RootSignature,
        pso: &ID3D12PipelineState,
        d: &Draw,
    ) -> Result<Vec<[f32; 4]>, String> {
        let dev = &self.device;
        let (w, h) = d.size;
        // SAFETY: D3D12 calls with descriptions that outlive them; resources
        // are kept alive until the fence says the GPU is done.
        unsafe {
            let queue: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .map_err(err)?;
            let alloc: ID3D12CommandAllocator = dev
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(err)?;
            let list: ID3D12GraphicsCommandList = dev
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso)
                .map_err(err)?;
            let tex_desc = |w: u32, h: u32, flags| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(w),
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: flags,
                ..Default::default()
            };
            let buf_desc = |size: u64| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let heap = |t| D3D12_HEAP_PROPERTIES {
                Type: t,
                ..Default::default()
            };
            let make = |props: D3D12_HEAP_TYPE,
                        desc: &D3D12_RESOURCE_DESC,
                        state,
                        clear: Option<&D3D12_CLEAR_VALUE>| {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &heap(props),
                    D3D12_HEAP_FLAG_NONE,
                    desc,
                    state,
                    clear.map(|c| c as *const _),
                    &mut r,
                )
                .map_err(err)?;
                r.ok_or_else(|| "no resource".to_string())
            };
            let target = make(
                D3D12_HEAP_TYPE_DEFAULT,
                &tex_desc(w, h, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                Some(&D3D12_CLEAR_VALUE {
                    Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                    Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
                }),
            )?;
            let rtv_heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .map_err(err)?;
            let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            dev.CreateRenderTargetView(&target, None, rtv);
            let views: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: d.textures.len().max(1) as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .map_err(err)?;
            let sampler_heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                    NumDescriptors: d.samplers.len().max(1) as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .map_err(err)?;
            let view_size = dev
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let sampler_size =
                dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
            let mut keep = Vec::new();
            for (i, &(tw, th, texels)) in d.textures.iter().enumerate() {
                let tex = make(
                    D3D12_HEAP_TYPE_DEFAULT,
                    &tex_desc(tw, th, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                )?;
                let pitch = (tw as usize * 16).next_multiple_of(256);
                let upload = make(
                    D3D12_HEAP_TYPE_UPLOAD,
                    &buf_desc((pitch * th as usize) as u64),
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                )?;
                let mut p = std::ptr::null_mut();
                upload.Map(0, None, Some(&mut p)).map_err(err)?;
                for y in 0..th as usize {
                    let row = &texels[y * tw as usize..(y + 1) * tw as usize];
                    std::ptr::copy_nonoverlapping(
                        row.as_ptr().cast::<u8>(),
                        p.cast::<u8>().add(y * pitch),
                        row.len() * 16,
                    );
                }
                upload.Unmap(0, None);
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&tex),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&upload),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                                Width: tw,
                                Height: th,
                                Depth: 1,
                                RowPitch: pitch as u32,
                            },
                        },
                    },
                };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.ResourceBarrier(&[transition(
                    &tex,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                        | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
                let mut at = views.GetCPUDescriptorHandleForHeapStart();
                at.ptr += i * view_size;
                dev.CreateShaderResourceView(&tex, None, at);
                keep.push(tex);
                keep.push(upload);
            }
            for (i, s) in d.samplers.iter().enumerate() {
                let mut at = sampler_heap.GetCPUDescriptorHandleForHeapStart();
                at.ptr += i * sampler_size;
                dev.CreateSampler(s, at);
            }
            let vb = if d.vertices.is_empty() {
                None
            } else {
                let bytes = d.vertices.len() * 4;
                let vb = make(
                    D3D12_HEAP_TYPE_UPLOAD,
                    &buf_desc(bytes as u64),
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                )?;
                let mut p = std::ptr::null_mut();
                vb.Map(0, None, Some(&mut p)).map_err(err)?;
                std::ptr::copy_nonoverlapping(
                    d.vertices.as_ptr().cast::<u8>(),
                    p.cast::<u8>(),
                    bytes,
                );
                vb.Unmap(0, None);
                Some(vb)
            };
            list.SetDescriptorHeaps(&[Some(views.clone()), Some(sampler_heap.clone())]);
            list.SetGraphicsRootSignature(root);
            for (param, arg) in &d.roots {
                match arg {
                    Root::Constants(data) => list.SetGraphicsRoot32BitConstants(
                        *param,
                        data.len() as u32,
                        data.as_ptr().cast(),
                        0,
                    ),
                    Root::Table { sampler, index } => {
                        let (heap, size) = if *sampler {
                            (&sampler_heap, sampler_size)
                        } else {
                            (&views, view_size)
                        };
                        let mut at = heap.GetGPUDescriptorHandleForHeapStart();
                        at.ptr += (index * size) as u64;
                        list.SetGraphicsRootDescriptorTable(*param, at);
                    }
                    Root::View { .. } => return Err("root views in a draw".into()),
                }
            }
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: w as f32,
                Height: h as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                right: w as i32,
                bottom: h as i32,
                ..Default::default()
            }]);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            if let Some(vb) = &vb {
                list.IASetVertexBuffers(
                    0,
                    Some(&[D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: vb.GetGPUVirtualAddress(),
                        SizeInBytes: (d.vertices.len() * 4) as u32,
                        StrideInBytes: d.stride,
                    }]),
                );
            }
            if d.mesh_groups > 0 {
                let list6: ID3D12GraphicsCommandList6 = list.cast().map_err(err)?;
                list6.DispatchMesh(d.mesh_groups, 1, 1);
            } else {
                list.DrawInstanced(3, 1, 0, 0);
            }
            list.ResourceBarrier(&[transition(
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let pitch = (w as usize * 16).next_multiple_of(256);
            let readback = make(
                D3D12_HEAP_TYPE_READBACK,
                &buf_desc((pitch * h as usize) as u64),
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
            )?;
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                            Width: w,
                            Height: h,
                            Depth: 1,
                            RowPitch: pitch as u32,
                        },
                    },
                },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&target),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.Close().map_err(err)?;
            queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
            queue.Signal(&fence, 1).map_err(err)?;
            while fence.GetCompletedValue() < 1 {
                if dev.GetDeviceRemovedReason().is_err() {
                    return Err(format!(
                        "device removed: {}",
                        err(dev.GetDeviceRemovedReason().unwrap_err())
                    ));
                }
                std::thread::yield_now();
            }
            if let Err(e) = dev.GetDeviceRemovedReason() {
                return Err(format!("device removed: {}", err(e)));
            }
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).map_err(err)?;
            let mut out = Vec::with_capacity((w * h) as usize);
            for y in 0..h as usize {
                let row = std::slice::from_raw_parts(
                    p.cast::<u8>().add(y * pitch).cast::<[f32; 4]>(),
                    w as usize,
                );
                out.extend_from_slice(row);
            }
            readback.Unmap(0, None);
            drop(keep);
            Ok(out)
        }
    }
}

#[cfg(test)]
fn transition(
    r: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A view in the heap for [`Host::dispatch`].
#[cfg(test)]
pub enum View<'a> {
    /// An RGBA32F texture SRV (width, height, texels).
    Texture(u32, u32, &'a [[f32; 4]]),
    /// An RGBA32F cube SRV (face size, mip count, texels in D3D subresource order).
    Cube(u32, u16, &'a [[f32; 4]]),
    /// A raw buffer UAV of this many dwords, zeroed.
    RawUav(u32),
    /// An RGBA32F texture UAV (width, height), zeroed.
    TextureUav(u32, u32),
    /// An R32_UINT texture UAV (width, height), zeroed (atomics).
    UintTextureUav(u32, u32),
    /// A typed RGBA16F buffer UAV of this many elements, zeroed.
    HalfUav(u32),
    /// A typed buffer UAV pre-filled with `data` (`format`, bytes per element).
    TypedUav {
        data: &'a [u32],
        format: DXGI_FORMAT,
        elem_bytes: u32,
    },
    /// A constant buffer of these dwords.
    Cbv(&'a [u32]),
    /// A buffer SRV over these dwords.
    Buffer { data: &'a [u32], stride: Stride },
}

/// How [`View::Buffer`] is viewed.
#[cfg(test)]
#[derive(Clone, Copy)]
pub enum Stride {
    Raw,
    /// RGBA32F elements.
    Typed,
    /// Structured, elements of this many bytes.
    Structured(u32),
    /// Typed, elements in this format of this many bytes.
    Format(DXGI_FORMAT, u32),
}

/// Bindings for [`Host::dispatch`]: root arguments, the view heap in order,
/// the sampler heap, and the group counts.
#[cfg(test)]
pub struct Dispatch<'a> {
    pub roots: Vec<(u32, Root)>,
    pub views: &'a [View<'a>],
    pub samplers: &'a [D3D12_SAMPLER_DESC],
    pub groups: [u32; 3],
}

#[cfg(test)]
impl Host {
    /// Runs a compute pipeline and returns each UAV's contents in heap
    /// order (texels as float bits, row by row).
    pub fn dispatch(
        &self,
        root: &ID3D12RootSignature,
        pso: &ID3D12PipelineState,
        d: &Dispatch,
    ) -> Result<Vec<Vec<u32>>, String> {
        self.dispatch_repeated(root, pso, d, 1).map(|(output, _)| output)
    }

    pub(super) fn dispatch_repeated(
        &self,
        root: &ID3D12RootSignature,
        pso: &ID3D12PipelineState,
        d: &Dispatch,
        passes: u32,
    ) -> Result<(Vec<Vec<u32>>, f64), String> {
        assert!(passes > 0);
        let dev = &self.device;
        // SAFETY: D3D12 calls with descriptions that outlive them; resources
        // stay alive until the fence says the GPU is done.
        unsafe {
            let queue: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .map_err(err)?;
            let alloc: ID3D12CommandAllocator = dev
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(err)?;
            let list: ID3D12GraphicsCommandList = dev
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso)
                .map_err(err)?;
            let make = |heap: D3D12_HEAP_TYPE, desc: &D3D12_RESOURCE_DESC, state| {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    desc,
                    state,
                    None,
                    &mut r,
                )
                .map_err(err)?;
                r.ok_or_else(|| "no resource".to_string())
            };
            let sample = DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            };
            let texture_as = |w: u32, h: u32, flags, format| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(w),
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: sample,
                Flags: flags,
                ..Default::default()
            };
            let texture = |w, h, flags| texture_as(w, h, flags, DXGI_FORMAT_R32G32B32A32_FLOAT);
            let buffer = |size: u64, flags| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: sample,
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            // Texel words: 4 for RGBA32F, 1 for R32_UINT.
            let footprint_as =
                |r: &ID3D12Resource, w: u32, h: u32, words: u32| D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(r),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: if words == 1 {
                                    DXGI_FORMAT_R32_UINT
                                } else {
                                    DXGI_FORMAT_R32G32B32A32_FLOAT
                                },
                                Width: w,
                                Height: h,
                                Depth: 1,
                                RowPitch: (w * 4 * words).next_multiple_of(256),
                            },
                        },
                    },
                };
            let footprint = |r: &ID3D12Resource, w: u32, h: u32| footprint_as(r, w, h, 4);
            let subresource = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(r),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let heap = |ty, n: usize| -> Result<ID3D12DescriptorHeap, String> {
                dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: ty,
                    NumDescriptors: n.max(1) as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .map_err(err)
            };
            // An upload buffer holding `data` (sized for a constant buffer).
            let upload = |data: &[u32]| -> Result<ID3D12Resource, String> {
                let size = (data.len() * 4).next_multiple_of(256);
                let b = make(
                    D3D12_HEAP_TYPE_UPLOAD,
                    &buffer(size as u64, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )?;
                let mut p = std::ptr::null_mut();
                b.Map(0, None, Some(&mut p)).map_err(err)?;
                std::ptr::copy_nonoverlapping(data.as_ptr(), p.cast::<u32>(), data.len());
                b.Unmap(0, None);
                Ok(b)
            };
            let views = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, d.views.len())?;
            let samplers = heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, d.samplers.len())?;
            let view_size = dev
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let sampler_size =
                dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
            let mut keep = Vec::new();
            // UAVs: (resource, dwords or texture size, readback buffer).
            let mut uavs = Vec::new();
            for (i, v) in d.views.iter().enumerate() {
                let mut at = views.GetCPUDescriptorHandleForHeapStart();
                at.ptr += i * view_size;
                match *v {
                    View::Texture(..) | View::Cube(..) => {
                        let (w, h, mut texels, faces, mips) = match *v {
                            View::Texture(w, h, t) => (w, h, t, 1, 1),
                            View::Cube(n, mips, t) => (n, n, t, 6, mips),
                            _ => unreachable!(),
                        };
                        let mut desc = texture(w, h, D3D12_RESOURCE_FLAG_NONE);
                        desc.DepthOrArraySize = faces as u16;
                        desc.MipLevels = mips;
                        let tex = make(
                            D3D12_HEAP_TYPE_DEFAULT,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        for sub in 0..u32::from(mips) * faces {
                            let mip = sub % u32::from(mips);
                            let (w, h) = ((w >> mip).max(1), (h >> mip).max(1));
                            let pitch = (w as usize * 16).next_multiple_of(256);
                            let upload = make(
                                D3D12_HEAP_TYPE_UPLOAD,
                                &buffer((pitch * h as usize) as u64, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_GENERIC_READ,
                            )?;
                            let mut p = std::ptr::null_mut();
                            upload.Map(0, None, Some(&mut p)).map_err(err)?;
                            let (level, rest) = texels.split_at((w * h) as usize);
                            texels = rest;
                            for y in 0..h as usize {
                                let row = &level[y * w as usize..(y + 1) * w as usize];
                                std::ptr::copy_nonoverlapping(
                                    row.as_ptr().cast::<u8>(),
                                    p.cast::<u8>().add(y * pitch),
                                    row.len() * 16,
                                );
                            }
                            upload.Unmap(0, None);
                            let mut at_face = subresource(&tex);
                            at_face.Anonymous.SubresourceIndex = sub;
                            list.CopyTextureRegion(
                                &at_face,
                                0,
                                0,
                                0,
                                &footprint(&upload, w, h),
                                None,
                            );
                            keep.push(upload);
                        }
                        list.ResourceBarrier(&[transition(
                            &tex,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        )]);
                        let cube = D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                            ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                TextureCube: D3D12_TEXCUBE_SRV {
                                    MipLevels: u32::from(mips),
                                    ..Default::default()
                                },
                            },
                        };
                        dev.CreateShaderResourceView(
                            &tex,
                            (faces == 6).then_some(&cube as *const _),
                            at,
                        );
                        keep.push(tex);
                    }
                    View::RawUav(_) | View::HalfUav(_) => {
                        // (dwords, format, elements, flags)
                        let (n, format, elements, flags) = match *v {
                            View::HalfUav(e) => (
                                2 * e,
                                DXGI_FORMAT_R16G16B16A16_FLOAT,
                                e,
                                D3D12_BUFFER_UAV_FLAG_NONE,
                            ),
                            View::RawUav(n) => {
                                (n, DXGI_FORMAT_R32_TYPELESS, n, D3D12_BUFFER_UAV_FLAG_RAW)
                            }
                            _ => unreachable!(),
                        };
                        let bytes = u64::from(n) * 4;
                        let b = make(
                            D3D12_HEAP_TYPE_DEFAULT,
                            &buffer(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                            D3D12_RESOURCE_STATE_COMMON,
                        )?;
                        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: format,
                            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_UAV {
                                    NumElements: elements,
                                    Flags: flags,
                                    ..Default::default()
                                },
                            },
                        };
                        dev.CreateUnorderedAccessView(&b, None, Some(&desc), at);
                        let rb = make(
                            D3D12_HEAP_TYPE_READBACK,
                            &buffer(bytes, D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        uavs.push((b, None, n, rb));
                    }
                    View::TypedUav {
                        data,
                        format,
                        elem_bytes,
                    } => {
                        let n = data.len() as u32;
                        let bytes = u64::from(n) * 4;
                        let elements = (bytes / u64::from(elem_bytes)) as u32;
                        let b = make(
                            D3D12_HEAP_TYPE_DEFAULT,
                            &buffer(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        let up = upload(data)?;
                        list.CopyBufferRegion(&b, 0, &up, 0, bytes);
                        list.ResourceBarrier(&[transition(
                            &b,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        )]);
                        keep.push(up);
                        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: format,
                            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_UAV {
                                    NumElements: elements,
                                    Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                                    ..Default::default()
                                },
                            },
                        };
                        dev.CreateUnorderedAccessView(&b, None, Some(&desc), at);
                        let rb = make(
                            D3D12_HEAP_TYPE_READBACK,
                            &buffer(bytes, D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        uavs.push((b, None, n, rb));
                    }
                    View::TextureUav(w, h) => {
                        let t = make(
                            D3D12_HEAP_TYPE_DEFAULT,
                            &texture(w, h, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        )?;
                        dev.CreateUnorderedAccessView(&t, None, None, at);
                        let pitch = (w * 16).next_multiple_of(256);
                        let rb = make(
                            D3D12_HEAP_TYPE_READBACK,
                            &buffer(u64::from(pitch * h), D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        uavs.push((t, Some((w, h)), 4, rb));
                    }
                    View::UintTextureUav(w, h) => {
                        let t = make(
                            D3D12_HEAP_TYPE_DEFAULT,
                            &texture_as(
                                w,
                                h,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                DXGI_FORMAT_R32_UINT,
                            ),
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        )?;
                        dev.CreateUnorderedAccessView(&t, None, None, at);
                        let pitch = (w * 4).next_multiple_of(256);
                        let rb = make(
                            D3D12_HEAP_TYPE_READBACK,
                            &buffer(u64::from(pitch * h), D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        uavs.push((t, Some((w, h)), 1, rb));
                    }
                    View::Cbv(data) => {
                        let b = upload(data)?;
                        dev.CreateConstantBufferView(
                            Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC {
                                BufferLocation: b.GetGPUVirtualAddress(),
                                SizeInBytes: (data.len() * 4).next_multiple_of(256) as u32,
                            }),
                            at,
                        );
                        keep.push(b);
                    }
                    View::Buffer { data, stride } => {
                        let b = upload(data)?;
                        let (format, elements, stride, flags) = match stride {
                            Stride::Raw => (
                                DXGI_FORMAT_R32_TYPELESS,
                                data.len(),
                                0,
                                D3D12_BUFFER_SRV_FLAG_RAW,
                            ),
                            Stride::Typed => (
                                DXGI_FORMAT_R32G32B32A32_FLOAT,
                                data.len() / 4,
                                0,
                                D3D12_BUFFER_SRV_FLAG_NONE,
                            ),
                            Stride::Structured(s) => (
                                DXGI_FORMAT_UNKNOWN,
                                data.len() * 4 / s as usize,
                                s,
                                D3D12_BUFFER_SRV_FLAG_NONE,
                            ),
                            Stride::Format(f, bytes) => (
                                f,
                                data.len() * 4 / bytes as usize,
                                0,
                                D3D12_BUFFER_SRV_FLAG_NONE,
                            ),
                        };
                        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: format,
                            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_SRV {
                                    FirstElement: 0,
                                    NumElements: elements as u32,
                                    StructureByteStride: stride,
                                    Flags: flags,
                                },
                            },
                        };
                        dev.CreateShaderResourceView(&b, Some(&desc), at);
                        keep.push(b);
                    }
                }
            }
            for (i, s) in d.samplers.iter().enumerate() {
                let mut at = samplers.GetCPUDescriptorHandleForHeapStart();
                at.ptr += i * sampler_size;
                dev.CreateSampler(s, at);
            }
            list.SetDescriptorHeaps(&[Some(views.clone()), Some(samplers.clone())]);
            list.SetComputeRootSignature(root);
            for (param, arg) in &d.roots {
                match arg {
                    Root::Constants(data) => list.SetComputeRoot32BitConstants(
                        *param,
                        data.len() as u32,
                        data.as_ptr().cast(),
                        0,
                    ),
                    Root::Table { sampler, index } => {
                        let (h, size) = if *sampler {
                            (&samplers, sampler_size)
                        } else {
                            (&views, view_size)
                        };
                        let mut at = h.GetGPUDescriptorHandleForHeapStart();
                        at.ptr += (index * size) as u64;
                        list.SetComputeRootDescriptorTable(*param, at);
                    }
                    Root::View { srv, data } => {
                        let b = upload(data)?;
                        let va = b.GetGPUVirtualAddress();
                        if *srv {
                            list.SetComputeRootShaderResourceView(*param, va);
                        } else {
                            list.SetComputeRootConstantBufferView(*param, va);
                        }
                        keep.push(b);
                    }
                }
            }
            let [x, y, z] = d.groups;
            let timing = if passes > 1 {
                let mut query: Option<ID3D12QueryHeap> = None;
                dev.CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 2,
                        NodeMask: 0,
                    },
                    &mut query,
                ).map_err(err)?;
                let back = make(
                    D3D12_HEAP_TYPE_READBACK,
                    &buffer(16, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )?;
                let query = query.ok_or("no timestamp query heap")?;
                list.EndQuery(&query, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                Some((query, back))
            } else {
                None
            };
            let uav_order = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER::default()),
                },
                ..Default::default()
            };
            for pass in 0..passes {
                if pass != 0 {
                    list.ResourceBarrier(std::slice::from_ref(&uav_order));
                }
                list.Dispatch(x, y, z);
            }
            if let Some((query, back)) = &timing {
                list.EndQuery(query, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                list.ResolveQueryData(query, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, back, 0);
            }
            for (r, size, n, rb) in &uavs {
                // (Buffers were promoted from the common state by the dispatch.)
                list.ResourceBarrier(&[transition(
                    r,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                match size {
                    Some((w, h)) => list.CopyTextureRegion(
                        &footprint_as(rb, *w, *h, *n),
                        0,
                        0,
                        0,
                        &subresource(r),
                        None,
                    ),
                    None => list.CopyBufferRegion(rb, 0, r, 0, u64::from(*n) * 4),
                }
            }
            list.Close().map_err(err)?;
            queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
            queue.Signal(&fence, 1).map_err(err)?;
            while fence.GetCompletedValue() < 1 {
                if let Err(e) = dev.GetDeviceRemovedReason() {
                    return Err(format!("device removed: {}", err(e)));
                }
                std::thread::yield_now();
            }
            let mut out = Vec::new();
            for (_, size, n, rb) in &uavs {
                let mut p = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut p)).map_err(err)?;
                let words = match size {
                    Some((w, h)) => {
                        let words = *n as usize;
                        let pitch = (*w as usize * 4 * words).next_multiple_of(256);
                        (0..*h as usize)
                            .flat_map(|y| {
                                std::slice::from_raw_parts(
                                    p.cast::<u8>().add(y * pitch).cast::<u32>(),
                                    *w as usize * words,
                                )
                                .to_vec()
                            })
                            .collect()
                    }
                    None => std::slice::from_raw_parts(p.cast::<u32>(), *n as usize).to_vec(),
                };
                rb.Unmap(0, None);
                out.push(words);
            }
            drop(keep);
            let ns = if let Some((_, back)) = timing {
                let mut p = std::ptr::null_mut();
                back.Map(0, None, Some(&mut p)).map_err(err)?;
                let [start, end] = *p.cast::<[u64; 2]>();
                back.Unmap(0, Some(&D3D12_RANGE::default()));
                if end <= start {
                    return Err("timestamp batch did not advance".into());
                }
                (end - start) as f64 * 1e9 / queue.GetTimestampFrequency().map_err(err)? as f64
                    / f64::from(passes)
            } else {
                0.0
            };
            Ok((out, ns))
        }
    }
}
