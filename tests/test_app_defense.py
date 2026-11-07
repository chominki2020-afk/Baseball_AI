import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # 모델 파일(.pkl)을 저장소 루트에서 읽음

"""수비팀 화면 (기본 화면): 추천 카드에 스윙·헛스윙 확률(앙상블)이 나오는지, 다른 화면들이 오류 없이 뜨는지"""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=170)
at.run()
print("수비팀 exc", [e.value for e in at.exception])
cards = [m.value for m in at.markdown if "헛스윙" in m.value and "스윙" in m.value]
print("추천 카드", len(cards), cards[0][:160] if cards else "")
for mode in ["♟️ 게임이론 볼배합", "⚔️ 공격팀"]:
    at.sidebar.radio[0].set_value(mode).run()
    print(mode, "exc", [e.value for e in at.exception])
