"""
전체 파이프라인을 순서대로 실행 (데이터 받기 → 모델 학습 → 검증)

  python pipeline.py --list            # 단계 목록
  python pipeline.py                   # 처음부터 끝까지 (몇 시간 걸림, 데이터만 약 850MB)
  python pipeline.py --from batter     # 특정 단계부터
  python pipeline.py --only game       # 한 단계만

원칙: 2024로 학습하고 2025로 평가한다. 하이퍼파라미터도 2024 안에서만 고른다.
"""
import argparse
import subprocess
import sys
import time

D24 = "statcast_2024-04-01_2024-09-30.csv"
D25 = "statcast_2025-03-27_2025-09-28.csv"

# (이름, 설명, 실행할 명령들, 만들어지는 파일)
STEPS = [
    ("data24", "2024 Statcast 받기 + Stuff+", [["stuff_plus.py", "--start", "2024-04-01", "--end", "2024-09-30"]], D24),
    ("data25", "2025 Statcast 받기 (검증용)", [["stuff_plus.py", "--start", "2025-03-27", "--end", "2025-09-28"]], D25),
    ("matchup", "매치업 모델 (구종×코스 실점)", [["build_matchup.py", "--data", D24]], "matchup_model.pkl"),
    ("swing", "스윙·헛스윙 모델", [["build_swing.py"]], "swing_model.pkl"),
    ("offense", "공격(타자 시점) 모델", [["build_offense.py"]], "offense_model.pkl"),
    ("bip", "타구 방향·질 모델", [["build_bip.py"]], "bip_model.pkl"),
    ("physics", "투구 궤적 물리", [["build_physics.py"]], "physics.pkl"),
    ("bio", "선수 신체조건 (MLB Stats API, 선택)", [["fetch_bio.py"]], "player_bio.csv"),
    ("mech_data", "투수 리포트·공 샘플", [["build_mechanics_data.py"]], "pitcher_report.pkl"),
    ("release_move", "릴리스 → 무브 관계 학습", [["build_release_move.py"]], "release_move.pkl"),
    ("val_release", "검증: 릴리스를 바꾼 투수들 (사례 1)", [["validate_release.py", "prep"], ["validate_release.py", "eval"]],
     "val_release_summary.csv"),
    ("personal", "공 모양 모델 v2 (직구 기준 상대값)", [["build_personal.py", "move"], ["build_personal.py", "model"]],
     "swing_model_v2.pkl"),
    ("val_personal", "검증: 개인화 무브", [["validate_personal.py"]], "val_personal_summary.csv"),
    ("batter", "타자 스윙 모델 + 검증 (사례 2·3)", [["build_batter_swing.py", "val"], ["build_batter_swing.py", "save"]],
     "batter_swing.pkl"),
    ("embed", "선수 임베딩 신경망 (PyTorch 필요, 사례 5)", [["build_embed.py", "val"], ["build_embed.py", "save"]],
     "embed_model.npz"),
    ("game", "게임이론: 노림 증거 + β 역추정 (사례 4)", [["build_game.py", "exploit"], ["build_game.py", "mix"]],
     "game_mix.pkl"),
    ("minimax", "미니맥스 검증 (사례 4)", [["build_minimax.py"]], "minimax.pkl"),
    ("backtest", "코스 추천 2025 백테스트", [["backtest.py", "--loop"], ["backtest.py", "--report"]], "results/backtest_2025.txt"),
    ("ensemble", "앙상블(신경망+LightGBM) 추천 백테스트", [["backtest_ensemble.py"], ["backtest_ensemble.py", "--report"]],
     "val_ensemble_summary.csv"),
    ("command", "투수별 제구 오차 추정 + 검증 (사례 7)", [["build_command.py"]], "val_command_summary.csv"),
    ("command_bt", "제구 가정 백테스트 (사례 7)", [["backtest_command.py"], ["backtest_command.py", "--report"]],
     "val_command_backtest.csv"),
    ("scouting", "스카우팅 리포트: 팀별 라인업·선발 표 + 샘플 리포트", [["scouting.py", "--teams"],
     ["scouting.py", "--opp", "NYY", "--us", "PIT", "--out", "docs/sample_scouting_PIT_vs_NYY.html"]],
     "docs/sample_scouting_PIT_vs_NYY.html"),
    ("high_fastball", "높은 패스트볼 × 타자 유형 검증", [["test_high_fastball.py", "--data", D25, "--split", "2025-07-01"]],
     "results/high_fastball_2025.txt"),
]


def run(cmd):
    if cmd[1:] == ["--loop"]:                     # backtest.py: 시간 제한 때문에 끝날 때까지 반복
        while True:
            out = subprocess.run([sys.executable, cmd[0]], capture_output=True, text=True, encoding="utf-8").stdout
            print(out.strip().splitlines()[-1] if out.strip() else "")
            if "추천 계산 완료" in out:
                return
    if cmd[1:] == ["--report"] and cmd[0] == "backtest.py":
        with open("results/backtest_2025.txt", "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, *cmd], stdout=f, check=True)
        return
    subprocess.run([sys.executable, *cmd], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--only")
    a = ap.parse_args()
    names = [s[0] for s in STEPS]
    if a.list:
        for i, (n, desc, _, out) in enumerate(STEPS, 1):
            print(f"{i:>2}. {n:<14}{desc}  → {out}")
        sys.exit()
    todo = STEPS
    if a.only:
        todo = [s for s in STEPS if s[0] == a.only]
    elif a.start:
        todo = STEPS[names.index(a.start):]
    for n, desc, cmds, out in todo:
        t0 = time.time()
        print(f"\n=== [{n}] {desc}")
        for c in cmds:
            run(c)
        print(f"    완료 ({time.time() - t0:.0f}초) → {out}")
