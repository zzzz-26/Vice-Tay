# Summarize a diagnostics capture created with VICTORY_FAULT_BOOK.
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)][string[]]$Path,
    [ValidateRange(1, [long]::MaxValue)][long]$Frequency = 10000000
)

foreach ($item in $Path) {
    $file = (Get-Item -LiteralPath $item -ErrorAction Stop).FullName
    $reader = [IO.BinaryReader]::new([IO.File]::OpenRead($file))
    try {
        # QPC ticks, three unsigned words, thread ID, event kind.
        $recordSize = 40
        $length = $reader.BaseStream.Length
        if ($length % $recordSize -ne 0) { throw "Incomplete fault record in $file." }

        $frames = 0
        $first = 0
        $last = 0
        $waitNs = 0
        $holdNs = 0
        $byBlock = @{}
        $byThread = @{}
        $events = @{}
        for ($offset = 0L; $offset -lt $length; $offset += $recordSize) {
            $ticks = $reader.ReadInt64()
            $block = $reader.ReadUInt64()
            $detail = $reader.ReadUInt64()
            $held = $reader.ReadUInt64()
            $threadId = $reader.ReadUInt32()
            $event = $reader.ReadUInt32()
            $events[$event] += 1
            if ($event -eq 0) {
                if ($frames -eq 0) { $first = $ticks }
                $last = $ticks
                $frames += 1
            }
            if ($event -eq 1) {
                $wait = $detail -band [uint64]4294967295
                $waitNs += $wait
                $holdNs += $held
                $byBlock[$block] += 1
                $byThread[$threadId] += $wait + $held
            }
        }

        $frameDivisor = [Math]::Max(1, $frames)
        $faults = [long]$events[[uint32]1]
        $busiestThreads = $byThread.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 5
        $hottestBlocks = $byBlock.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 5
        [pscustomobject]@{
            File = $file
            Records = [long]($length / $recordSize)
            Frames = $frames
            MeanFrameMs = $(if ($frames -gt 1) { [Math]::Round(($last - $first) * 1000.0 / $Frequency / ($frames - 1), 2) })
            GuardFaults = $faults
            FaultsPerFrame = $(if ($frames) { [Math]::Round($faults / $frameDivisor, 2) })
            WaitMsPerFrame = $(if ($frames) { [Math]::Round($waitNs / 1e6 / $frameDivisor, 3) })
            HoldMsPerFrame = $(if ($frames) { [Math]::Round($holdNs / 1e6 / $frameDivisor, 3) })
            BusiestThreads = @($busiestThreads | ForEach-Object { 'thread {0}: {1:N3} ms' -f $_.Key, ($_.Value / 1e6) })
            HottestBlocks = @($hottestBlocks | ForEach-Object { '0x{0:x}: {1} faults' -f $_.Key, $_.Value })
        }
    } finally {
        $reader.Dispose()
    }
}
