# One-time interactive owner enrollment. No secrets are printed, logged or sent to GitHub.
# Run inside your own Windows PowerShell terminal after creating a standard OpenRouter key.
$ErrorActionPreference = 'Stop'
$root = 'https://openrouter.ai/api/v1'
$destination = Join-Path $env:LOCALAPPDATA 'MusicJapan\APIForge\secrets\openrouter-free.token'
$allowed = @(
  'nvidia/nemotron-3-ultra-550b-a55b:free',
  'thinkingmachines/inkling:free',
  'poolside/laguna-s-2.1:free'
)
if (Test-Path -LiteralPath $destination) {
  throw 'An existing OpenRouter key is present: refusing to overwrite'
}
$catalog = Invoke-RestMethod -Uri ($root + '/models') -TimeoutSec 30
foreach ($name in $allowed) {
  $m = @($catalog.data | Where-Object { $_.id -ceq $name })
  if ($m.Count -ne 1) { throw ('Missing free model: ' + $name) }
  $prices = $m[0].pricing
  if ($null -eq $prices) { throw 'Price not verifiable' }
  foreach ($field in @('prompt','completion')) {
    if ($null -eq $prices.$field) { throw 'Missing price field' }
  }
  foreach ($field in $prices.PSObject.Properties) {
    if ($null -eq $field.Value) { throw 'Unverifiable price field' }
    try { $cost = [decimal]$field.Value }
    catch { throw 'Unverifiable price field' }
    if ($cost -ne 0) { throw 'Nonzero model pricing: abort' }
  }
}
$parent = Split-Path -Parent $destination
if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
  throw 'Existing owner credential directory not found'
}
Write-Host 'All three free model prices verified at $0. Enter your own OpenRouter API key below.'
$secure = Read-Host 'OpenRouter standard API key (hidden)' -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$plain = $null
$tmp = $null
try {
  $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
  if (($plain -notmatch '^sk-or-v1-[a-zA-Z0-9_-]{20,}$') -or $plain.Length -gt 512) {
    throw 'Key format invalid'
  }
  $info = Invoke-RestMethod -Uri ($root + '/key') -Headers @{Authorization=('Bearer '+$plain)} -TimeoutSec 20
  if ($null -eq $info.data -or $info.data.is_management_key -eq $true) {
    throw 'Standard API key validation failed'
  }
  $tmp = Join-Path $parent ('openrouter-free-' + [guid]::NewGuid().ToString('N') + '.tmp')
  [IO.File]::WriteAllText($tmp, $plain, (New-Object Text.UTF8Encoding($false)))
  $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User
  $acl = New-Object Security.AccessControl.FileSecurity
  $acl.SetOwner($sid)
  $acl.SetAccessRuleProtection($true, $false)
  $rights = [Security.AccessControl.FileSystemRights]::Read -bor [Security.AccessControl.FileSystemRights]::Write
  $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
      $sid, $rights, [Security.AccessControl.AccessControlType]::Allow)))
  [IO.File]::SetAccessControl($tmp, $acl)
  [IO.File]::Move($tmp, $destination)
  $tmp = $null
  Write-Host 'OpenRouter free key authenticated and enrolled with owner-only file ACL.'
  Write-Host 'Run: python "$env:LOCALAPPDATA\MusicJapan\OpenRouterFree\openrouter_free_client.py" --check-key'
} finally {
  if ($tmp -and (Test-Path -LiteralPath $tmp)) { Remove-Item -LiteralPath $tmp -Force }
  [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
  $plain = $null
}
