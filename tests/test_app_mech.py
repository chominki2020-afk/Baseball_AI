import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # 모델 파일(.pkl)을 저장소 루트에서 읽음

from streamlit.testing.v1 import AppTest
at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=170)
at.run()
at.sidebar.radio[0].set_value("🛠️ 투수 개선").run()
print("exc", [e.value for e in at.exception])
print("md", [m.value[:90] for m in at.markdown][:12])
print("info", [i.value[:120] for i in at.info])
for nm in ["Glasnow, Tyler", "Hader, Josh"]:
    at.sidebar.selectbox(key="mech_p").set_value(nm).run()
    print(nm, "exc", [e.value for e in at.exception])
    print([m.value[:110] for m in at.markdown if "border-left" in m.value or "⭐" in m.value][:5])
r = at.sidebar.radio[1]
print("opts", r.options)
r.set_value(r.options[1]).run()
print("all exc", [e.value for e in at.exception], len(at.markdown))
