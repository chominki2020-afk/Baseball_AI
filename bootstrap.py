"""
모델 파일이 없으면 GitHub Release에서 자동으로 받기 (Streamlit Cloud 같은 새 환경에서 앱이 바로 켜지게)

  - 저장소에는 코드만 있고, 학습된 모델(.pkl/.npz)은 Release의 baseball-ai-models.zip에 있음
  - 다른 곳에서 받으려면 환경변수 BASEBALL_AI_MODELS_URL 로 주소를 바꿀 수 있음
"""
import io
import os
import urllib.request
import zipfile

MODELS_URL = os.environ.get(
    "BASEBALL_AI_MODELS_URL",
    "https://github.com/chominki2020-afk/Baseball_AI/releases/latest/download/baseball-ai-models.zip")
REQUIRED = ["matchup_model.pkl", "swing_model.pkl", "offense_model.pkl", "bip_model.pkl", "physics.pkl"]


def missing(root="."):
    return [f for f in REQUIRED if not os.path.exists(os.path.join(root, f))]


def ensure_models(root=".", url=MODELS_URL, log=print):
    """필수 모델이 하나라도 없으면 zip을 받아 풀기. 이미 있는 파일은 덮어쓰지 않음. 받은 파일 수를 돌려줌"""
    if not missing(root):
        return 0
    log(f"모델 파일 받는 중: {url}")
    with urllib.request.urlopen(url, timeout=180) as r:
        data = r.read()
    n = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if not base or base.startswith(".") or not base.endswith((".pkl", ".npz")):
                continue                                    # 모델 파일만 (경로 조작 방지)
            dest = os.path.join(root, base)
            if os.path.exists(dest):
                continue
            with z.open(name) as src, open(dest, "wb") as out:
                out.write(src.read())
            n += 1
    log(f"모델 {n}개 준비 완료")
    return n


if __name__ == "__main__":
    ensure_models()
