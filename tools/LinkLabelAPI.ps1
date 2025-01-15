param (
    [Parameter(Mandatory=$true)][string]$token,
    [string]$outdir = "clips"
)

$base_url = "http://141.37.157.50:28080"

# $project = Invoke-WebRequest -Uri "$base_url/api/projects/5" -Headers @{Authorization="Token $token"}
$projectTasks = Invoke-WebRequest -Uri "$base_url/api/tasks/" -Headers @{Authorization="Token $token"} -Body @{project=5}
$jsonProjectTasks = $projectTasks | ConvertFrom-Json

If(!(test-path -PathType container $outdir))
{
    New-Item -Path outdir -ItemType Directory
}

foreach($task in $jsonProjectTasks.tasks) {
    $img = $task.data.img
    $filename = Split-Path $task.data.img -Leaf  # Returns "file.txt"
    $outPath = "$outdir/$filename"
    
    Invoke-WebRequest -Uri "$base_url$img" -Headers @{Authorization="Token $token"} -OutFile $outPath
}