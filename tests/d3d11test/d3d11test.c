/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * d3d11test.exe - minimal D3D11 flip-model swapchain app for testing lsfg-metal under
 * Wine + D3DMetal (or DXVK/MoltenVK). Clears to a cycling color and draws nothing else.
 *
 *   x86_64-w64-mingw32-gcc -O2 -o d3d11test.exe d3d11test.c -ld3d11 -ldxgi -luser32 -lgdi32 -luuid
 *   d3d11test.exe [seconds] [fps]
 */

#define COBJMACROS
#define INITGUID
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

int main(int argc, char** argv) {
    const double seconds = argc > 1 ? atof(argv[1]) : 6.0;
    const double fps = argc > 2 ? atof(argv[2]) : 30.0;

    WNDCLASSW wc = {0};
    wc.lpfnWndProc = wndproc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.lpszClassName = L"d3d11test";
    RegisterClassW(&wc);
    HWND hwnd = CreateWindowW(L"d3d11test", L"d3d11test", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        100, 100, 960, 540, NULL, NULL, wc.hInstance, NULL);

    DXGI_SWAP_CHAIN_DESC sd = {0};
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 960;
    sd.BufferDesc.Height = 540;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    ID3D11Device* device = NULL;
    ID3D11DeviceContext* ctx = NULL;
    IDXGISwapChain* swapchain = NULL;
    HRESULT hr = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0, NULL, 0,
        D3D11_SDK_VERSION, &sd, &swapchain, &device, NULL, &ctx);
    if (FAILED(hr)) { printf("D3D11CreateDeviceAndSwapChain failed: 0x%08lx\n", hr); return 1; }

    ID3D11Texture2D* backbuffer = NULL;
    IDXGISwapChain_GetBuffer(swapchain, 0, &IID_ID3D11Texture2D, (void**)&backbuffer);
    ID3D11RenderTargetView* rtv = NULL;
    ID3D11Device_CreateRenderTargetView(device, (ID3D11Resource*)backbuffer, NULL, &rtv);

    LARGE_INTEGER freq, start, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    unsigned frames = 0;
    double next = 0;
    for (;;) {
        MSG msg;
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        QueryPerformanceCounter(&now);
        const double t = (double)(now.QuadPart - start.QuadPart) / (double)freq.QuadPart;
        if (t > seconds) break;
        if (t < next) { Sleep(1); continue; }
        next += 1.0 / fps;

        const float h = (float)fmod(frames * 0.02, 1.0);
        const float color[4] = { 0.5f + 0.5f * sinf(h * 6.283f), 0.5f + 0.5f * sinf(h * 6.283f + 2.1f),
                                 0.5f + 0.5f * sinf(h * 6.283f + 4.2f), 1.0f };
        ID3D11DeviceContext_OMSetRenderTargets(ctx, 1, &rtv, NULL);
        ID3D11DeviceContext_ClearRenderTargetView(ctx, rtv, color);
        IDXGISwapChain_Present(swapchain, 1, 0);
        frames++;
    }
    QueryPerformanceCounter(&now);
    const double t = (double)(now.QuadPart - start.QuadPart) / (double)freq.QuadPart;
    printf("d3d11test: %u frames in %.1fs (%.1f fps)\n", frames, t, frames / t);
    fflush(stdout);
    return 0;
}
