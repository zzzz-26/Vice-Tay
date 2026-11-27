//! Two presented planes as one image: the console's display puts the second
//! (the interface) over the first (the scene), each scaled to the screen.
//! Grand Theft Auto VI presents so on Series S.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use crate::host::Host;

/// Whether a second presented plane is composed over the first
/// (`VICTORY_PLANES`; on by default, `0` turns it off, as in
/// avatarmanz/VIctory-7900XTX). It acts only when the title presents a
/// second plane.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PLANES").is_none_or(|v| v != "0"))
}

const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2), visibility=SHADER_VISIBILITY_PIXEL), RootConstants(num32BitConstants=1, b0, visibility=SHADER_VISIBILITY_PIXEL), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP)"
Texture2D<float4> scene : register(t0);
Texture2D<float4> over : register(t1);
SamplerState smooth_ : register(s0);
cbuffer Plane : register(b0) { uint mode; };
struct V { float4 at : SV_Position; float2 uv : TEXCOORD0; };
[RootSignature(RS)]
V vs(uint id : SV_VertexID) {
    V v;
    v.uv = float2((id << 1) & 2, id & 2);
    v.at = float4(v.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return v;
}
[RootSignature(RS)]
float4 ps(V v) : SV_Target {
    float4 s = scene.SampleLevel(smooth_, v.uv, 0);
    if (mode == 1) return float4(saturate(s.rgb), 1);
    if (mode == 2) return saturate(s);
    float4 o = over.SampleLevel(smooth_, v.uv, 0);
    return float4(saturate(o.rgb + s.rgb * (1 - saturate(o.a))), 1);
}
"#;

type View<'a> = (&'a ID3D12Resource, DXGI_FORMAT);

pub(crate) enum Picture<'a> {
    Composed(View<'a>, View<'a>),
    Scene(View<'a>),
    Overlay(View<'a>),
}

pub(crate) const UI_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
pub(crate) const UI_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0,
);

/// Two images allow the proxy to read the previous UI asynchronously while
/// the current Present records another. Present waits for the previous UI
/// composition before returning, so its image can be reused next time.
pub(crate) struct UiFrames {
    pub(crate) size: (u32, u32),
    pub(crate) images: [ID3D12Resource; 2],
}

impl UiFrames {
    pub(crate) fn new(host: &Host, size: (u32, u32)) -> Option<Self> {
        let create = || {
            let mut texture = None;
            // SAFETY: a default-heap RGBA render target for one UI image.
            unsafe {
                host.device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: u64::from(size.0),
                        Height: size.1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: UI_FORMAT,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                        ..Default::default()
                    },
                    UI_READ,
                    None,
                    &mut texture,
                ).ok()?;
            }
            texture
        };
        Some(Self { size, images: [create()?, create()?] })
    }
}

pub(crate) struct Composer {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    views: Vec<ID3D12DescriptorHeap>,
    step: u32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Composer {}
// SAFETY: as above.
unsafe impl Sync for Composer {}

fn compile(entry: PCSTR, target: PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; the out pointer is
    // ours.
    unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("compose"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            None,
        )?;
    }
    code.ok_or_else(windows::core::Error::empty)
}

impl Composer {
    pub(crate) fn new(host: &Host, format: DXGI_FORMAT) -> windows::core::Result<Self> {
        let (vs, ps) = (
            compile(s!("vs"), s!("vs_5_1"))?,
            compile(s!("ps"), s!("ps_5_1"))?,
        );
        // SAFETY: the blobs' own bytes, alive through the calls.
        unsafe {
            let bytes = |b: &ID3DBlob| {
                std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
            };
            let device = &host.device;
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes(&vs))?;
            let code = |b: &ID3DBlob| D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.GetBufferPointer(),
                BytecodeLength: b.GetBufferSize(),
            };
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs),
                PS: code(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.RTVFormats[0] = format;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
                D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let pso = device.CreateGraphicsPipelineState(&desc)?;
            Ok(Composer {
                root,
                pso,
                views: Vec::new(),
                step: device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            })
        }
    }

    /// Each presenter command-list slot owns its own pair of SRVs. The swap
    /// chain may have several outstanding presents; the slot is selected by
    /// the completion fence, and its descriptors are only overwritten on reuse.
    pub(crate) fn record(
        &mut self,
        host: &Host,
        slot: usize,
        list: &ID3D12GraphicsCommandList,
        picture: Picture<'_>,
        back: &ID3D12Resource,
        (width, height): (u32, u32),
    ) -> windows::core::Result<()> {
        let (mode, planes) = match picture {
            Picture::Composed(scene, overlay) => (0, [scene, overlay]),
            Picture::Scene(scene) => (1, [scene, scene]),
            Picture::Overlay(overlay) => (2, [overlay, overlay]),
        };
        while self.views.len() <= slot {
            let heap = unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 2,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })?
            };
            self.views.push(heap);
        }
        let views = &self.views[slot];
        // SAFETY: an open list; the views are made in slots no frame in flight
        // reads; live textures.
        unsafe {
            let (cpu, gpu) = (
                views.GetCPUDescriptorHandleForHeapStart(),
                views.GetGPUDescriptorHandleForHeapStart(),
            );
            for (i, plane) in planes.iter().enumerate() {
                let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu.ptr + (i * self.step as usize),
                };
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: plane.1,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MipLevels: 1,
                            ..Default::default()
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(Some(plane.0), Some(&desc), at);
            }
            let rtv = host.rtv(Some(back), None);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.SetDescriptorHeaps(&[Some(views.clone())]);
            list.SetGraphicsRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetGraphicsRootDescriptorTable(0, gpu);
            list.SetGraphicsRoot32BitConstant(1, mode, 0);
            list.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: width as f32,
                Height: height as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);
            list.DrawInstanced(3, 1, 0, 0);
        }
        Ok(())
    }
}

/// Select a legal float-sampled view of a display plane, preserving a typed
/// resource's format. Typeless resources use the corresponding display type.
pub(crate) fn view(d: &D3D12_RESOURCE_DESC) -> Option<DXGI_FORMAT> {
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || d.DepthOrArraySize != 1
        || d.SampleDesc.Count != 1
        || d.Flags.contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
    {
        return None;
    }
    match d.Format.0 {
        9 => Some(DXGI_FORMAT_R16G16B16A16_FLOAT),
        23 => Some(DXGI_FORMAT_R10G10B10A2_UNORM),
        27 => Some(DXGI_FORMAT_R8G8B8A8_UNORM),
        90 => Some(DXGI_FORMAT_B8G8R8A8_UNORM),
        10 | 11 | 13 | 24 | 26 | 28 | 29 | 31 | 67 | 87 | 91 => Some(d.Format),
        _ => None,
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod tests {
    use super::*;

    #[test]
    fn composes_scaled_premultiplied_plane_on_gpu() {
        let Some(host) = crate::host::host() else {
            eprintln!("skipping display-plane readback: no D3D12 device");
            return;
        };
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let mut compositor = Composer::new(host, format).unwrap();
        let list = host.new_list().unwrap();
        let texture = |width: u64| -> ID3D12Resource {
            let mut resource = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: width,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut resource,
                    )
                    .unwrap()
            };
            resource.unwrap()
        };
        // A half-red base at one resolution. The overlay at output resolution
        // contains transparent, half-opaque green and fully opaque blue pixels.
        let base = texture(1);
        let overlay = texture(3);
        let output = texture(3);
        let scene_output = texture(3);
        let ui_output = texture(3);
        let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let cmds = &list.list;
        unsafe {
            cmds.ClearRenderTargetView(host.rtv(Some(&base), None), &[0.5, 0.0, 0.0, 1.0], None);
            let overlay_rtv = host.rtv(Some(&overlay), None);
            let colors = [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.5],
                [0.0, 0.0, 1.0, 1.0],
            ];
            for (i, color) in colors.iter().enumerate() {
                let rect = windows::Win32::Foundation::RECT {
                    left: i as i32,
                    right: i as i32 + 1,
                    top: 0,
                    bottom: 1,
                };
                cmds.ClearRenderTargetView(overlay_rtv, color, Some(&[rect]));
            }
            cmds.ResourceBarrier(&[
                crate::host::transition(&base, target, read),
                crate::host::transition(&overlay, target, read),
            ]);
            compositor
                .record(
                    host,
                    0,
                    cmds,
                    Picture::Composed((&base, format), (&overlay, format)),
                    &output,
                    (3, 1),
                )
                .unwrap();
            compositor.record(host, 1, cmds, Picture::Scene((&base, format)), &scene_output, (3, 1)).unwrap();
            compositor.record(host, 2, cmds, Picture::Overlay((&overlay, format)), &ui_output, (3, 1)).unwrap();
            let outputs = [&output, &scene_output, &ui_output];
            cmds.ResourceBarrier(&outputs.map(|output| crate::host::transition(
                output, target, D3D12_RESOURCE_STATE_COPY_SOURCE,
            )));
            let readbacks = outputs.map(|output| crate::capture::read_back(&host.device, cmds, output).unwrap());
            list.close();
            host.execute(&[&list]);
            host.device.GetDeviceRemovedReason().unwrap();
            let expected = [
                [128, 0, 0, 255, 64, 128, 0, 255, 0, 0, 255, 255],
                [128, 0, 0, 255, 128, 0, 0, 255, 128, 0, 0, 255],
                [0, 0, 0, 0, 0, 128, 0, 128, 0, 0, 255, 255],
            ];
            for ((buffer, footprint), expected) in readbacks.into_iter().zip(expected) {
                let mut mapped = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut mapped)).unwrap();
                let start = mapped.cast::<u8>().add(footprint.Offset as usize);
                let pixels: [u8; 12] = std::slice::from_raw_parts(start, 12).try_into().unwrap();
                buffer.Unmap(0, None);
                for (actual, expected) in pixels.iter().zip(expected) {
                    assert!(actual.abs_diff(expected) <= 1, "unexpected plane {pixels:?}");
                }
                eprintln!("D3D12 display-plane readback RGBA: {pixels:?}");
            }
        }
    }
}
