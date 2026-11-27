# Starts the title with the settings this fork is played with (README.md, "Getting started").
#   scripts\play.ps1 -Exe <path of your own copy's executable>
# -Safe         without the newer, faster ways (the first thing to try when something is wrong)
# -RayTracing   the title's own ray-traced indirect light: lit interiors, about a third slower
# -Windowed     a window instead of the whole screen (full screen is the default)
# -NoPanel      without the performance panel (on by default, F3 hides it while playing)
# -SkipShaders  start even when the shader cache is cold (otherwise every shader is translated first)
# -Console      series-x (the default) or series-s: the console the title takes this PC for
# -FrameGen     AMD frame generation
# -Fsr / -Dlss  PC upscaling in Series S (quality, balanced or performance)
# -Fsr4K        PC FidelityFX at 3840x2160 (quality, balanced or performance)
# -Dlss4K       NVIDIA DLSS at 3840x2160 (quality, balanced or performance); FSR if it cannot start
# -Build        target\diag (the default, the build all measurements are of) or target\release
# -Machine      auto (the default: by the graphics card), 7900xtx, 4070, 4090 or laptop
#               (laptop: an NVIDIA PC with 16 GB of RAM and a 6-8 GB card, docs/laptop-16gb.md)
# -More         other settings: -More @{VICTORY_PANEL='1'} or -More @{VICTORY_AMBIENT='1.0'}
# On a smaller NVIDIA PC, compare the same Series S scene after a fresh start
# with one override at a time, e.g. -More @{VICTORY_LAND_THREADS='3'} for
# copy-back work, or -More @{VICTORY_THIN_THREADS='6'} for list translation.
# Run it from a normal window, not as administrator: the driver then keeps its shader cache elsewhere.
param(
    [Parameter(Mandatory)][string]$Exe,
    [switch]$Safe,
    [switch]$RayTracing,
    [switch]$Windowed,
    [switch]$NoPanel,
    [switch]$SkipShaders,
    [ValidateSet('series-x', 'series-s')][string]$Console = 'series-x',
    [switch]$FrameGen,
    [ValidateSet('quality', 'balanced', 'performance')][string]$Fsr,
    [ValidateSet('quality', 'balanced', 'performance')][string]$Dlss,
    [ValidateSet('quality', 'balanced', 'performance')][string]$Fsr4K,
    [ValidateSet('quality', 'balanced', 'performance')][string]$Dlss4K,
    [ValidateSet('diag', 'release')][string]$Build = 'diag',
    [ValidateSet('auto', '7900xtx', '4070', '4090', 'laptop')][string]$Machine = 'auto',
    [hashtable]$More = @{}
)
$settings = @{
    VICTORY_TEXTURE_NO_WAIT = '1'; VICTORY_RELAY = '2'; VICTORY_WIDE_BACK = '1'; VICTORY_CARD_REHASH_RUNS = '1'
    VICTORY_JOIN_COPIES = '1'; VICTORY_INDIRECT_WINDOW = '1'; VICTORY_RELAY_COPY = '1'; VICTORY_CARD_STEPS = '2'
    VICTORY_TITLE_PRIORITIES = '0'
    VICTORY_SHOW_AFTER = '1'; VICTORY_AMBIENT = '0.8'; VICTORY_D3D12_DRED = 'light'
}
$ram = (Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue).TotalPhysicalMemory
$controllers = @(if ($Machine -in 'auto', '7900xtx') {
    Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
})
$hasNvidia = [bool]($controllers.PNPDeviceID -match 'VEN_10DE')
if ($Machine -eq 'auto') {
    # Use the existing vendor defaults. The 4070 / 7800X3D profile is explicit.
    $Machine = if ($hasNvidia) { '4090' } else { '7900xtx' }
}
if ($Machine -eq '7900xtx') {
    # i9-12900K, RX 7900 XTX, 64 GB: the page-mapping lock was fastest with one
    # thread, and the AMD driver lowers its clocks between submissions.
    $settings.VICTORY_MAP_THREADS = '1'; $settings.VICTORY_CARD_AWAKE = '1'

    # Issue #138 reported device resets in relay modes 1 and 2 on these drivers.
    if (-not $hasNvidia -and -not $More.ContainsKey('VICTORY_RELAY')) {
        $affected = $controllers | Where-Object {
            $_.PNPDeviceID -match 'VEN_1002' -and $_.Name -match '\bRX\s+6700\s+XT\b' -and
                $_.DriverVersion -in '32.0.21045.26001', '32.0.32029.2010'
        } | Select-Object -First 1
        if ($affected) {
            $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
            if (($os.BuildNumber -as [int]) -ge 22000 -and $os.Caption -match '\bLTSC\b') {
                $settings.VICTORY_RELAY = '0'
                Write-Host "Using relay mode 0 for the RX 6700 XT on Windows 11 LTSC (driver $($affected.DriverVersion), issue #138)."
            }
        }
    }
} else {
    # RTX 4090, Ryzen 9 9950X3D (16 cores of one kind, two caches), 48 GB.
    # NVIDIA sets its clocks by the driver's power mode: no spinner on the
    # compute queue. 16 cores record and land, not 8.
    $settings.VICTORY_THIN_THREADS = '12'; $settings.VICTORY_UNDER_THREADS = '8'; $settings.VICTORY_LAND_THREADS = '6'
    if (-not $Safe) {
        # Pages are mapped by several threads only while the title streams
        # (not the default: one run behind it).
        $settings.VICTORY_MAP_THREADS = 'auto'
        # A fill of the constant-buffer ring that a small copy at its place writes
        # into refused the whole submission to the card steps; in list order it is
        # taken (submit thread 104.7 -> 50.9 ms a frame, 12.1 -> 14.3 fps here).
        $settings.VICTORY_PLACED_ORDER = '1'
    }
}
if ($Machine -eq '4070') {
    # Ryzen 7 7800X3D: eight physical cores, versus the 4090 test PC's sixteen.
    # Keep the same rendering profile; spend VRAM on
    # shader buffers before textures consume it. No approximate shader modes.
    $settings.VICTORY_THIN_THREADS = '8'; $settings.VICTORY_UNDER_THREADS = '6'; $settings.VICTORY_LAND_THREADS = '4'
    $settings.VICTORY_MAP_THREADS = '1'; $settings.VICTORY_PIPELINE_WORKERS = '3'
    if (-not $Safe) {
        $settings.VICTORY_CARD_RESERVE_BUDGETS = '1'
        $settings.VICTORY_CARD_BUFFERS_MIN_KB = '64'; $settings.VICTORY_CARD_BUFFERS_MB = '1536'
        $settings.VICTORY_CARD_WINDOWS_MB = '1024'
    }
}
if ($Machine -eq 'laptop') {
    # i7-13700H, RTX 3050 6 GB Laptop GPU, 16 GB (docs/laptop-16gb.md); asked for explicitly,
    # the measurements cover that one machine. On top of the NVIDIA settings above, which the
    # measured runs had: what makes the world load fit in 16 GB. These stay under -Safe too,
    # since without them the world load runs out of RAM.
    # Game files read past Windows' file cache, title buffers in its private memory used in
    # place, windows outside buffers cut on 64 KiB with their tails, and up to 3 GB of them in
    # the card's memory (Resizable BAR) instead of RAM. The card's own copies of title buffers
    # get 256 MB, which leaves the card room for those windows.
    $settings.VICTORY_DSTORAGE_IO = 'direct'; $settings.VICTORY_INPLACE = '2'
    $settings.VICTORY_WINDOW_KB = '64'; $settings.VICTORY_WINDOW_TAILS = '1'
    $settings.VICTORY_WINDOWS_BAR_MB = '3072'; $settings.VICTORY_CARD_BUFFERS_MB = '256'
    $settings.VICTORY_LEVELS_ON_DEMAND = '1'
    # A ceiling on the game's working set: past it Windows trims the game's own pages, not the
    # rest of the PC's (16 GB: 4,500 MB, which leaves about 11.6 GB for Windows, the memory the
    # card shares and the desktop; with no ceiling free RAM ran out before the world loaded).
    if ($ram) {
        $settings.VICTORY_WORKING_SET_MB = [string][math]::Max(3072, [int64]($ram / 1MB) - 11624)
    }
    # A 1080p output drawn at half size by the title's own upscaler: the card stays at about
    # 4.2 GB of a 5.2 GB budget.
    $settings.VICTORY_OUTPUT = '1920x1080'; $settings.VICTORY_UPSCALER = '6'; $settings.VICTORY_RENDER_SCALE = '0.5'
    # No RAM throttle and no change to the title's watchdog here: the world load fits with the
    # settings above (docs/laptop-16gb.md); for longer sessions see #96 and #126 there.
}
# -Safe leaves out every way still on trial, and the newest faster ones.
if ($Safe) {
    $settings.VICTORY_UNDER_WIDE = '0'; $settings.VICTORY_LAND_ECHO = '0'
} else {
    $settings.VICTORY_THIN = '1'; $settings.VICTORY_JOIN_PARTS = '8'; $settings.VICTORY_UPLOAD_AHEAD = '1'
    # A dispatch does not end a batch of argument clamps before the title waits
    # for it: off by default, as other ways the title may wait are not known.
    $settings.VICTORY_INDIRECT_BATCH = '2'
}
if ($RayTracing) { $settings.VICTORY_BVH_BUILDER = '1' }
# The whole screen, with no frame, as on the console.
if (-not $Windowed) { $settings.VICTORY_FULLSCREEN = '1' }
# The performance panel (FPS, frame times, GPU, CPU, VRAM, RAM) in the upper left
# corner, so clips and screenshots show what the build runs at and on what PC.
if (-not $NoPanel) { $settings.VICTORY_PANEL = '1' }
if ($FrameGen) { $settings.VICTORY_FRAMEGEN = '1' }
if ($Fsr -or $Dlss) { $settings.Remove('VICTORY_UPSCALER') }
if ($Fsr) { $settings.VICTORY_FSR = $Fsr.ToLowerInvariant() }
if ($Dlss) { $settings.VICTORY_DLSS = $Dlss.ToLowerInvariant() }
if ($Fsr4K) { $settings.VICTORY_FSR_4K = $Fsr4K.ToLowerInvariant() }
if ($Dlss4K) { $settings.VICTORY_DLSS_4K = $Dlss4K.ToLowerInvariant() }
foreach ($k in $More.Keys) { $settings[$k] = $More[$k] }

Write-Host "Settings for the $Machine$(if ($Safe) { ', safe' })$(if ($RayTracing) { ', with ray tracing' })."
if ($Console -eq 'series-s' -and $Machine -in '4090', 'laptop') {
    Write-Host ("Series S requests {0} list translation threads, {1} for memory updates, and {2} for landing." -f
        $settings.VICTORY_THIN_THREADS, $settings.VICTORY_UNDER_THREADS, $settings.VICTORY_LAND_THREADS)
}
if ($ram -and $ram -lt 60GB) {
    Write-Host ("Windows reports {0:N1} GiB of physical RAM. Keep a system-managed page file. See docs/measurements.md for measured Series S memory use." -f ($ram / 1GB))
}

$victory = Join-Path $PSScriptRoot "..\target\$Build\victory.exe"
if (-not (Test-Path -LiteralPath $victory -PathType Leaf)) { Write-Host "No $victory. Build it first (README.md)."; exit 1 }
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) { Write-Host "No $Exe."; exit 1 }
$victory = (Resolve-Path -LiteralPath $victory).Path
Write-Host "Vice-Tay executable $victory"
try {
    $checksum = Get-FileHash -LiteralPath $victory -Algorithm SHA256 -ErrorAction Stop
    Write-Host "Vice-Tay SHA-256 $($checksum.Hash)"
} catch {
    Write-Warning "Could not read the Vice-Tay executable checksum: $($_.Exception.Message)"
}
# (Other settings left over in this window go; where your files are stays.)
$paths = 'VICTORY_CONFIG', 'VICTORY_GAME_DIR', 'VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR', 'VICTORY_FIDELITYFX_DIR', 'VICTORY_FFX_FG_DIR', 'VICTORY_DLSS_DIR'
$inherited = @(Get-ChildItem Env:VICTORY_* | Where-Object { $_.Name -notin $paths })
foreach ($entry in $inherited) { Remove-Item "Env:$($entry.Name)" }
foreach ($k in $settings.Keys) { Set-Item "Env:$k" $settings[$k] }
$discarded = $inherited | Where-Object {
    $_.Name -notin $More.Keys -and -not $_.Value.Equals([Environment]::GetEnvironmentVariable($_.Name))
} | Sort-Object Name
if ($discarded) {
    Write-Host "Discarded inherited values for $($discarded.Name -join ', '). Use -More to pass them explicitly."
}
$Exe = (Resolve-Path -LiteralPath $Exe).Path
Push-Location (Join-Path $PSScriptRoot '..')   # victory.toml is looked for here
try {
    if (-not $SkipShaders) {
        $config = & $victory config
        if ($LASTEXITCODE -eq 0) {
            $cache = $config | Select-String '^cache_dir\s*:\s*(.+)$' | Select-Object -First 1
            if ($cache) {
                $shaders = Join-Path $cache.Matches[0].Groups[1].Value 'shaders'
                $entries = Get-ChildItem -LiteralPath $shaders -File -Recurse -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -cmatch '^[0-9a-f]{62}$' } | Select-Object -First 1000
                if (@($entries).Count -lt 1000) {
                    $source = Split-Path -LiteralPath $Exe
                    $game = $config | Select-String '^game_dir\s*:\s*(.+)$' | Select-Object -First 1
                    if ($game -and (Test-Path -LiteralPath $game.Matches[0].Groups[1].Value -PathType Container)) {
                        $source = $game.Matches[0].Groups[1].Value
                    }
                    Write-Host "The shader cache has fewer than 1,000 entries. Translating shaders before launch. Use -SkipShaders to start without this step."
                    & $victory shaders $source
                    if ($LASTEXITCODE -ne 0) { Write-Host "victory shaders ended with exit code $LASTEXITCODE; starting anyway." }
                }
            }
        }
    }
    & $victory run $Exe --console $Console --execute
} finally { Pop-Location }
