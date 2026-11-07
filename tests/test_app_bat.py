import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # 모델 파일(.pkl)을 저장소 루트에서 읽음

from streamlit.testing.v1 import AppTest
at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=170)
at.run()
at.sidebar.radio[0].set_value("🏏 타자 개선").run()
print("exc", [e.value for e in at.exception])
print([m.value[:100] for m in at.markdown if "border-left" in m.value][:3])
for nm in ["Judge, Aaron", "Kwan, Steven"]:
    at.sidebar.selectbox(key="bat_p").set_value(nm).run()
    print(nm, "exc", [e.value for e in at.exception])
for k in ["style", "xwcon", "miss"]:
    at.sidebar.radio[1].set_value(k).run(); print(k, "exc", [e.value for e in at.exception])
