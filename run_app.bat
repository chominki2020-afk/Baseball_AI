@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo 볼배합 추천 앱을 켜는 중... (처음엔 설치 때문에 1~2분 걸려요)
python -m pip install --quiet streamlit plotly anthropic
python -m streamlit run app.py
