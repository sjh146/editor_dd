# Flask 서버 외부 접속 진단 스크립트

Write-Host "=== Flask 서버 외부 접속 진단 ===" -ForegroundColor Cyan
Write-Host ""

# 1. 서버 실행 상태 확인
Write-Host "1. 서버 실행 상태 확인..." -ForegroundColor Yellow
$connections = Get-NetTCPConnection -LocalPort 5001 -ErrorAction SilentlyContinue
if ($connections) {
    Write-Host "   ✓ 서버가 포트 5001에서 실행 중입니다" -ForegroundColor Green
    $connections | Format-Table LocalAddress, LocalPort, State -AutoSize
} else {
    Write-Host "   ✗ 서버가 포트 5001에서 실행되지 않습니다" -ForegroundColor Red
    Write-Host "   → 'python app.py'로 서버를 실행하세요" -ForegroundColor Yellow
}
Write-Host ""

# 2. 로컬 IP 주소 확인
Write-Host "2. 로컬 IP 주소 확인..." -ForegroundColor Yellow
$localIPs = Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
    $_.IPAddress -notlike "127.*" -and 
    $_.IPAddress -notlike "169.254.*"
} | Select-Object IPAddress, InterfaceAlias
if ($localIPs) {
    Write-Host "   로컬 IP 주소:" -ForegroundColor Green
    $localIPs | Format-Table -AutoSize
} else {
    Write-Host "   ✗ 로컬 IP 주소를 찾을 수 없습니다" -ForegroundColor Red
}
Write-Host ""

# 3. 방화벽 규칙 확인
Write-Host "3. 방화벽 규칙 확인..." -ForegroundColor Yellow
$firewallRules = Get-NetFirewallPortFilter | Where-Object {$_.LocalPort -eq 5001} | Get-NetFirewallRule
if ($firewallRules) {
    Write-Host "   ✓ 방화벽 규칙이 설정되어 있습니다:" -ForegroundColor Green
    $firewallRules | Format-Table DisplayName, Enabled, Direction, Action -AutoSize
    
    $disabledRules = $firewallRules | Where-Object {$_.Enabled -eq $false}
    if ($disabledRules) {
        Write-Host "   ⚠ 비활성화된 규칙이 있습니다:" -ForegroundColor Yellow
        $disabledRules | Format-Table DisplayName, Enabled -AutoSize
    }
} else {
    Write-Host "   ✗ 포트 5001에 대한 방화벽 규칙이 없습니다" -ForegroundColor Red
    Write-Host "   → 방화벽 규칙을 추가하세요:" -ForegroundColor Yellow
    Write-Host "   New-NetFirewallRule -DisplayName 'Flask Server Port 5001' -Direction Inbound -LocalPort 5001 -Protocol TCP -Action Allow" -ForegroundColor Gray
}
Write-Host ""

# 4. 공인 IP 주소 확인
Write-Host "4. 공인 IP 주소 확인..." -ForegroundColor Yellow
try {
    $publicIP = (Invoke-WebRequest -Uri "https://api.ipify.org" -UseBasicParsing -TimeoutSec 5).Content
    Write-Host "   공인 IP 주소: $publicIP" -ForegroundColor Green
    Write-Host "   → 외부에서 접속 시 사용할 주소: http://$publicIP`:5001" -ForegroundColor Cyan
} catch {
    Write-Host "   ⚠ 공인 IP 주소를 확인할 수 없습니다 (인터넷 연결 확인 필요)" -ForegroundColor Yellow
}
Write-Host ""

# 5. 포트 포워딩 확인 안내
Write-Host "5. 라우터 포트포워딩 설정 확인..." -ForegroundColor Yellow
Write-Host "   ⚠ 라우터에서 포트포워딩 설정이 필요합니다:" -ForegroundColor Yellow
Write-Host "   - 외부 포트: 5001" -ForegroundColor Gray
Write-Host "   - 내부 IP: 192.168.75.48 (또는 위에서 확인한 로컬 IP)" -ForegroundColor Gray
Write-Host "   - 내부 포트: 5001" -ForegroundColor Gray
Write-Host "   - 프로토콜: TCP" -ForegroundColor Gray
Write-Host ""

# 6. 로컬 접속 테스트
Write-Host "6. 로컬 접속 테스트..." -ForegroundColor Yellow
$testIP = ($localIPs | Select-Object -First 1).IPAddress
if ($testIP) {
    try {
        $response = Invoke-WebRequest -Uri "http://$testIP`:5001" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        Write-Host "   ✓ 로컬 네트워크에서 접속 가능합니다 (http://$testIP`:5001)" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠ 로컬 네트워크에서 접속 테스트 실패: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "   → 서버가 실행 중인지 확인하세요" -ForegroundColor Gray
    }
}
Write-Host ""

Write-Host "=== 진단 완료 ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "외부 접속이 안 되는 경우 체크리스트:" -ForegroundColor Yellow
Write-Host "1. 라우터 관리 페이지에서 포트포워딩 설정 확인" -ForegroundColor White
Write-Host "2. 공인 IP 주소로 접속 시도 (위에서 확인한 IP)" -ForegroundColor White
Write-Host "3. 방화벽 규칙이 활성화되어 있는지 확인" -ForegroundColor White
Write-Host "4. ISP(인터넷 제공업체)가 포트를 막지 않는지 확인" -ForegroundColor White

