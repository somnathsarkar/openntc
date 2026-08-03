#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <wrl.h>
using namespace Microsoft::WRL;

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>

#include <algorithm>
#include <chrono>

#if defined(_DEBUG)
#define VERIFY(hr) do { assert(!FAILED(hr)); } while(0)
#else
#define VERIFY(hr) do { (hr); } while(0)
#endif

const uint8_t g_numframes = 2;
uint32_t g_width = 1280;
uint32_t g_height = 720;

HWND g_hwnd;
RECT g_window_rect;

ComPtr<ID3D12Device2> g_device;
ComPtr<ID3D12CommandQueue> g_queue;
ComPtr<IDXGISwapChain4> g_swapchain;
ComPtr<ID3D12Resource> g_buffers[g_numframes];
ComPtr<ID3D12GraphicsCommandList> g_commandlist;
ComPtr<ID3D12CommandAllocator> g_commandallocators[g_numframes];
ComPtr<ID3D12DescriptorHeap> g_descriptorheap;
UINT g_descriptorsize;
UINT g_frame_i;
bool g_initialized;

ComPtr<ID3D12Fence> g_fence;
uint64_t g_fenceval = 0;
uint64_t g_framefenceval[g_numframes] = {};
HANDLE g_fence_event;

bool g_vsync = true;
bool g_gsync = false; // Tearing enabled?
bool g_fullscreen = false;

uint64_t SignalFence(ComPtr<ID3D12CommandQueue> command_queue, ComPtr<ID3D12Fence> fence, uint64_t* fenceval)
{
  (*fenceval)++;
  VERIFY(command_queue->Signal(fence.Get(), *fenceval));
  return *fenceval;
}

void WaitForFenceValue(ComPtr<ID3D12Fence> fence, uint64_t fenceval, HANDLE fenceevent, std::chrono::milliseconds duration_ms = std::chrono::milliseconds::max())
{
  if (fence->GetCompletedValue() < fenceval)
  {
    VERIFY(fence->SetEventOnCompletion(fenceval, fenceevent));
    ::WaitForSingleObject(fenceevent, static_cast<DWORD>(duration_ms.count()));
  }
}

void Update()
{
  static uint64_t framecounter = 0;
  static double elapsed_seconds = 0.0;
  static std::chrono::high_resolution_clock clock;
  static auto t0 = clock.now();

  framecounter++;
  auto t1 = clock.now();
  auto dT = t1 - t0;
  t0 = t1;

  elapsed_seconds += dT.count();
  if (elapsed_seconds > 1.0)
  {
    char buffer[500];
    auto fps = framecounter / elapsed_seconds;
    sprintf_s(buffer, 500, "FPS: %f\n", fps);
    OutputDebugStringA(buffer);
    framecounter = 0;
    elapsed_seconds = 0.0;
  }
}

void Render()
{
  auto command_allocator = g_commandallocators[g_frame_i];
  auto buffer = g_buffers[g_frame_i];
  command_allocator->Reset();
  g_commandlist->Reset(command_allocator.Get(), nullptr);

  {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = buffer.Get();
    barrier.Transition.Subresource = 0;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    g_commandlist->ResourceBarrier(1, &barrier);

    FLOAT clear_color[] = { 0.4f, 0.6f, 0.9f, 1.0f };
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = g_descriptorheap->GetCPUDescriptorHandleForHeapStart();
    rtv_handle.ptr += g_frame_i * g_descriptorsize;
    g_commandlist->ClearRenderTargetView(rtv_handle, clear_color, 0, nullptr);
  }

  {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = buffer.Get();
    barrier.Transition.Subresource = 0;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    g_commandlist->ResourceBarrier(1, &barrier);
    VERIFY(g_commandlist->Close());

    ID3D12CommandList* const command_lists[] = { g_commandlist.Get() };
    g_queue->ExecuteCommandLists(_countof(command_lists), command_lists);

    UINT sync_interval = g_vsync ? 1 : 0;
    UINT present_flags = (g_gsync && !g_vsync) ? DXGI_PRESENT_ALLOW_TEARING : 0;
    VERIFY(g_swapchain->Present(sync_interval, present_flags));
    g_framefenceval[g_frame_i] = SignalFence(g_queue, g_fence, &g_fenceval);

    g_frame_i = g_swapchain->GetCurrentBackBufferIndex();
    WaitForFenceValue(g_fence, g_framefenceval[g_frame_i], g_fence_event);
  }
}

void UpdateRenderTargetViews(ComPtr<ID3D12Device2> device, ComPtr<IDXGISwapChain4> swapchain, ComPtr<ID3D12DescriptorHeap> descriptor_heap)
{
  auto rtv_descriptor_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

  D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = descriptor_heap->GetCPUDescriptorHandleForHeapStart();

  for(int i = 0; i < g_numframes; i++)
  {
    ComPtr<ID3D12Resource> buffer;
    VERIFY(swapchain->GetBuffer(i, IID_PPV_ARGS(&buffer)));
    device->CreateRenderTargetView(buffer.Get(), nullptr, rtv_handle);
    g_buffers[i] = buffer;
    rtv_handle.ptr += rtv_descriptor_size;
  }

  g_descriptorsize = rtv_descriptor_size;
}

void Flush(ComPtr<ID3D12CommandQueue> command_queue, ComPtr<ID3D12Fence> fence, uint64_t* fenceval, HANDLE fenceevent)
{
  uint64_t signalval = SignalFence(command_queue, fence, fenceval);
  WaitForFenceValue(fence, signalval, fenceevent);
}

void Resize(uint32_t width, uint32_t height)
{
  if (g_width != width || g_height != height)
  {
    g_width = std::max(1u, width);
    g_height = std::max(1u, height);

    Flush(g_queue, g_fence, &g_fenceval, g_fence_event);

    for (int i = 0; i < g_numframes; i++)
    {
      g_buffers[i].Reset();
      g_framefenceval[i] = g_framefenceval[g_frame_i];
    }

    DXGI_SWAP_CHAIN_DESC swapchain_desc = {};
    VERIFY(g_swapchain->GetDesc(&swapchain_desc));
    VERIFY(g_swapchain->ResizeBuffers(g_numframes, g_width, g_height, swapchain_desc.BufferDesc.Format, swapchain_desc.Flags));
    g_frame_i = g_swapchain->GetCurrentBackBufferIndex();
    UpdateRenderTargetViews(g_device, g_swapchain, g_descriptorheap);
  }
}

void SetFullscreen(bool fullscreen)
{
  if (g_fullscreen != fullscreen)
  {
    g_fullscreen = fullscreen;

    if (g_fullscreen)
    {
      ::GetWindowRect(g_hwnd, &g_window_rect);
      UINT window_style = WS_OVERLAPPEDWINDOW & ~(WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX);
      ::SetWindowLongA(g_hwnd, GWL_STYLE, window_style);

      HMONITOR hmonitor = ::MonitorFromWindow(g_hwnd, MONITOR_DEFAULTTONEAREST);
      MONITORINFOEXA monitor_info = {};
      monitor_info.cbSize = sizeof(MONITORINFOEXA);
      ::GetMonitorInfoA(hmonitor, &monitor_info);

      ::SetWindowPos(
        g_hwnd, 
        HWND_TOP, 
        monitor_info.rcMonitor.left, 
        monitor_info.rcMonitor.top, 
        monitor_info.rcMonitor.right - monitor_info.rcMonitor.left, 
        monitor_info.rcMonitor.bottom - monitor_info.rcMonitor.top, 
        SWP_FRAMECHANGED | SWP_NOACTIVATE);

      ::ShowWindow(g_hwnd, SW_MAXIMIZE);
    }
  }
}

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  if (g_initialized)
  {
    switch (uMsg)
    {
      case WM_PAINT:
        Update();
        Render();
        break;

      case WM_SYSKEYDOWN:
      case WM_KEYDOWN:
      {
        bool alt = (::GetAsyncKeyState(VK_MENU) & 0x8000) != 0;

        switch (wParam)
        {
          case 'V':
            g_vsync = !g_vsync;
            break;
          case VK_ESCAPE:
            ::PostQuitMessage(0);
            break;
          case VK_RETURN:
            if (alt)
            {
          case VK_F11:
              SetFullscreen(!g_fullscreen);
            }
            break;
        }
      }
      break;

      case WM_SYSCHAR:
      break;

      case WM_SIZE:
      {
        RECT client_rect = {};
        ::GetClientRect(g_hwnd, &client_rect);
        int width = client_rect.right - client_rect.left;
        int height = client_rect.bottom - client_rect.top;
        Resize(width, height);
      }
      break;

      case WM_DESTROY:
        ::PostQuitMessage(0);
        break;

      default:
        return ::DefWindowProcA(hWnd, uMsg, wParam, lParam);
    }
  }
  return ::DefWindowProcA(hWnd, uMsg, wParam, lParam);
}

void EnableDebugLayer()
{
#if defined(_DEBUG)
  ComPtr<ID3D12Debug> debug_interface;
  VERIFY(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_interface)));
  debug_interface->EnableDebugLayer();
#endif
}

HWND CreateAndShowWindow(HINSTANCE hInstance, const char* class_name, const char* title)
{
  WNDCLASSEXA wndcls = {0};
  wndcls.cbSize = sizeof(WNDCLASSEXA);
  wndcls.style = CS_HREDRAW | CS_VREDRAW;
  wndcls.lpfnWndProc = &WndProc;
  wndcls.cbClsExtra = 0;
  wndcls.cbWndExtra = 0;
  wndcls.hInstance = hInstance;
  wndcls.hIcon = ::LoadIcon(hInstance, NULL);
  wndcls.hCursor = ::LoadCursor(NULL, IDC_ARROW);
  wndcls.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  wndcls.lpszClassName = class_name;
  wndcls.hIconSm = ::LoadIcon(hInstance, NULL);

  static ATOM atom = ::RegisterClassExA(&wndcls);
  assert(atom > 0);

  HWND hWnd = ::CreateWindowExA(
    NULL,
    class_name,
    title,
    WS_OVERLAPPEDWINDOW,
    100,
    100,
    g_width,
    g_height,
    NULL,
    NULL,
    hInstance,
    NULL);

  assert(hWnd);

  return hWnd;
}

ComPtr<IDXGIAdapter4> GetAdapter()
{
  ComPtr<IDXGIFactory4> dxgiFactory;
  UINT create_factory_flags = 0;
#if defined(_DEBUG)
  create_factory_flags = DXGI_CREATE_FACTORY_DEBUG;
#endif
  VERIFY(CreateDXGIFactory2(create_factory_flags, IID_PPV_ARGS(&dxgiFactory)));

  ComPtr<IDXGIAdapter1> dxgi_adapter1;
  ComPtr<IDXGIAdapter4> dxgi_adapter4;
  SIZE_T max_dedicated_vram = 0;
  for (UINT i = 0; dxgiFactory->EnumAdapters1(i, &dxgi_adapter1) != DXGI_ERROR_NOT_FOUND; i++)
  {
    DXGI_ADAPTER_DESC1 dxgi_adapter_desc1;
    dxgi_adapter1->GetDesc1(&dxgi_adapter_desc1);

    if ((dxgi_adapter_desc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
        SUCCEEDED(D3D12CreateDevice(dxgi_adapter1.Get(), D3D_FEATURE_LEVEL_12_2, __uuidof(ID3D12Device), nullptr)) &&
        dxgi_adapter_desc1.DedicatedVideoMemory > max_dedicated_vram)
    {
      max_dedicated_vram = dxgi_adapter_desc1.DedicatedVideoMemory;
      VERIFY(dxgi_adapter1.As(&dxgi_adapter4));
    }
  }
  return dxgi_adapter4;
}

ComPtr<ID3D12Device2> CreateDevice(ComPtr<IDXGIAdapter4> adapter)
{
  ComPtr<ID3D12Device2> device2;
  VERIFY(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_2, IID_PPV_ARGS(&device2)));

#if defined(_DEBUG)
  ComPtr<ID3D12InfoQueue> info_queue;
  if (SUCCEEDED(device2.As(&info_queue)))
  {
    info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
    info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
    info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);

    D3D12_MESSAGE_SEVERITY severities[] =
    {
      D3D12_MESSAGE_SEVERITY_INFO
    };

    D3D12_MESSAGE_ID deny_ids[] = {
      D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
      D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,
      D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,
    };

    D3D12_INFO_QUEUE_FILTER new_filter = {};
    new_filter.DenyList.NumSeverities = _countof(severities);
    new_filter.DenyList.pSeverityList = severities;
    new_filter.DenyList.NumIDs = _countof(deny_ids);
    new_filter.DenyList.pIDList = deny_ids;

    VERIFY(info_queue->PushStorageFilter(&new_filter));
  }
#endif
  return device2;
}

ComPtr<ID3D12CommandQueue> CreateCommandQueue(ComPtr<ID3D12Device2> device, D3D12_COMMAND_LIST_TYPE type)
{
  ComPtr<ID3D12CommandQueue> command_queue;

  D3D12_COMMAND_QUEUE_DESC desc = {};
  desc.Type = type;
  desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
  desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  desc.NodeMask = 0;

  VERIFY(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&command_queue)));
  return command_queue;
}

bool CheckTearingSupport()
{
  BOOL allow_tearing = FALSE;

  ComPtr<IDXGIFactory4> factory4;
  if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory4))))
  {
    ComPtr<IDXGIFactory5> factory5;
    if (SUCCEEDED(factory4.As(&factory5)))
    {
      if (FAILED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow_tearing, sizeof(allow_tearing))))
      {
        allow_tearing = FALSE;
      }
    }
  }

  return (allow_tearing == TRUE);
}

ComPtr<IDXGISwapChain4> CreateSwapChain(HWND hwnd, ComPtr<ID3D12CommandQueue> command_queue, uint32_t width, uint32_t height, uint32_t buffer_count)
{
  ComPtr<IDXGISwapChain4> dxgi_swapchain4;
  ComPtr<IDXGIFactory4> dxgi_factory4;
  UINT create_factory_flags = 0;
#if defined(_DEBUG)
  create_factory_flags = DXGI_CREATE_FACTORY_DEBUG;
#endif
  VERIFY(CreateDXGIFactory2(create_factory_flags, IID_PPV_ARGS(&dxgi_factory4)));

  DXGI_SWAP_CHAIN_DESC1 swapchain_desc = {0};
  swapchain_desc.Width = width;
  swapchain_desc.Height = height;
  swapchain_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  swapchain_desc.Stereo = FALSE;
  swapchain_desc.SampleDesc = {1, 0};
  swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  swapchain_desc.BufferCount = buffer_count;
  swapchain_desc.Scaling = DXGI_SCALING_STRETCH;
  swapchain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  swapchain_desc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
  swapchain_desc.Flags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

  ComPtr<IDXGISwapChain1> swapchain1;
  VERIFY(dxgi_factory4->CreateSwapChainForHwnd(command_queue.Get(), g_hwnd, &swapchain_desc, nullptr, nullptr, &swapchain1));
  VERIFY(dxgi_factory4->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER));
  VERIFY(swapchain1.As(&dxgi_swapchain4));

  return dxgi_swapchain4;
}

ComPtr<ID3D12DescriptorHeap> CreateDescriptorHeap(ComPtr<ID3D12Device2> device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t num_descriptors)
{
  ComPtr<ID3D12DescriptorHeap> descriptor_heap;

  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.NumDescriptors = num_descriptors;
  desc.Type = type;

  VERIFY(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&descriptor_heap)));
  return descriptor_heap;
}

ComPtr<ID3D12CommandAllocator> CreateCommandAllocator(ComPtr<ID3D12Device> device, D3D12_COMMAND_LIST_TYPE type)
{
  ComPtr<ID3D12CommandAllocator> command_allocator;
  VERIFY(device->CreateCommandAllocator(type, IID_PPV_ARGS(&command_allocator)));

  return command_allocator;
}

ComPtr<ID3D12GraphicsCommandList> CreateCommandList(ComPtr<ID3D12Device2> device, ComPtr<ID3D12CommandAllocator> command_allocator, D3D12_COMMAND_LIST_TYPE type)
{
  ComPtr<ID3D12GraphicsCommandList> command_list;
  VERIFY(device->CreateCommandList(0, type, command_allocator.Get(), nullptr, IID_PPV_ARGS(&command_list)));
  VERIFY(command_list->Close());

  return command_list;
}

ComPtr<ID3D12Fence> CreateFence(ComPtr<ID3D12Device2> device)
{
  ComPtr<ID3D12Fence> fence;
  VERIFY(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
  return fence;
}

HANDLE CreateEventHandle()
{
  HANDLE fence_event;
  fence_event = ::CreateEventA(nullptr, FALSE, FALSE, NULL);
  assert(fence_event);
  return fence_event;
}

int CALLBACK wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow)
{
  SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

  const char* wndclass_name = "openntc-gui-window-class";

  EnableDebugLayer();
  g_gsync = CheckTearingSupport();
  g_hwnd = CreateAndShowWindow(hInstance, wndclass_name, "openntc-gui");

  ::GetWindowRect(g_hwnd, &g_window_rect);

  ComPtr<IDXGIAdapter4> dxgi_adapter4 = GetAdapter();
  g_device = CreateDevice(dxgi_adapter4);
  g_queue = CreateCommandQueue(g_device, D3D12_COMMAND_LIST_TYPE_DIRECT);
  g_swapchain = CreateSwapChain(g_hwnd, g_queue, g_width, g_height, g_numframes);
  g_frame_i = g_swapchain->GetCurrentBackBufferIndex();
  g_descriptorheap = CreateDescriptorHeap(g_device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, g_numframes);
  g_descriptorsize = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
  UpdateRenderTargetViews(g_device, g_swapchain, g_descriptorheap);

  for (int i = 0; i < g_numframes; i++)
  {
    g_commandallocators[i] = CreateCommandAllocator(g_device, D3D12_COMMAND_LIST_TYPE_DIRECT);
  }
  g_commandlist = CreateCommandList(g_device, g_commandallocators[g_frame_i], D3D12_COMMAND_LIST_TYPE_DIRECT);

  g_fence = CreateFence(g_device);
  g_fence_event = CreateEventHandle();

  g_initialized = true;
  ::ShowWindow(g_hwnd, SW_SHOW);
  MSG msg = {};
  while (msg.message != WM_QUIT)
  {
    if (::PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE))
    {
      ::TranslateMessage(&msg);
      ::DispatchMessage(&msg);
    }
  }

  Flush(g_queue, g_fence, &g_fenceval, g_fence_event);
  ::CloseHandle(g_fence_event);

  return 0;
}