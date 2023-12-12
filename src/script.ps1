# Loop through 5000 iterations
$maxIterations = 5000
$timeOutInSeconds = 1
for ($i = 1; $i -le $maxIterations; $i++) {
    # Start a new process (Python script) and wait for it to finish
    $process = Start-Process -FilePath "python3" -ArgumentList "generate-synthetic-images.py" -PassThru -WindowStyle Hidden

    
    $timeoutReached = $false
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while (-not $process.HasExited -and $timer.Elapsed.TotalSeconds -lt $timeOutInSeconds) {
        Start-Sleep -Seconds 1
    }

    # Check if the process is still running
    if (-not $process.HasExited) {
        Write-Host "Timeout reached. Terminating process."
        
        # Kill the process forcefully
        $process.Kill()
        $timeoutReached = $true
    }

    $timer.Stop()

    # Output the result based on whether the process completed or was terminated due to a timeout
    if (-not $timeoutReached) {
        Write-Host "Iteration $i completed successfully."
    } else {
        Write-Host "Iteration $i terminated due to timeout."
    }
}