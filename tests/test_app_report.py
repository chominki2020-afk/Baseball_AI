import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # 모델 파일(.pkl)을 저장소 루트에서 읽음

"""스카우팅 리포트 화면: 투수 1명으로 리포트를 만들고 내려받기 버튼이 생기는지 (빠른 확인용)"""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=175)
at.run()
at.sidebar.radio[0].set_value("📋 스카우팅 리포트").run()
print("화면 exc", [e.value for e in at.exception])
ms = at.sidebar.multiselect(key="r_p")
ms.set_value(ms.value[:1]).run()
at.button[0].click().run()
print("생성 exc", [e.value for e in at.exception])
html = at.session_state["rep_html"][1] if "rep_html" in at.session_state else ""
print("리포트 길이", len(html), "타자 페이지", html.count('class="page"') - 1)
