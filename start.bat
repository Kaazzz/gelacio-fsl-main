@echo off
echo ===============================================
echo    CheXCA - Chest X-ray AI Diagnosis System
echo ===============================================
echo.

echo Starting Backend Server...
start "CheXCA Backend" cmd /k "cd backend && python main.py"

timeout /t 3 /nobreak >nul

echo Starting Frontend Application...
start "CheXCA Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ===============================================
echo Both servers are starting...
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press Ctrl+C in each window to stop the servers.
echo ===============================================
