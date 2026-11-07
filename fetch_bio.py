"""
투수 신체조건(키·몸무게·나이) 받기 — MLB 공식 Stats API

실행: python fetch_bio.py   (인터넷 필요, 1분 이내)
결과: player_bio.csv  (pitcher, name, height_in, weight_lb, birth_date, throws)
"""
import json
import time
import urllib.request

import pandas as pd

URL = "https://statsapi.mlb.com/api/v1/people?personIds={}"


def height_in(text):
    """'6\\' 6"' → 78 (인치)"""
    try:
        ft, inch = text.replace('"', "").split("'")
        return int(ft) * 12 + int(inch.strip() or 0)
    except Exception:
        return None


if __name__ == "__main__":
    ids = pd.read_csv("statcast_2024-04-01_2024-09-30.csv", usecols=["pitcher"])["pitcher"].unique().tolist()
    print(f"투수 {len(ids)}명 정보 받는 중...")
    rows = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        req = urllib.request.Request(URL.format(",".join(map(str, chunk))),
                                     headers={"User-Agent": "baseball-portfolio-project"})
        with urllib.request.urlopen(req, timeout=30) as r:
            people = json.load(r).get("people", [])
        for p in people:
            rows.append({"pitcher": p["id"], "name": p.get("fullName"),
                         "height_in": height_in(p.get("height", "")), "weight_lb": p.get("weight"),
                         "birth_date": p.get("birthDate"), "throws": p.get("pitchHand", {}).get("code")})
        print(f"  {min(i + 100, len(ids))} / {len(ids)}")
        time.sleep(0.3)
    out = pd.DataFrame(rows)
    out.to_csv("player_bio.csv", index=False, encoding="utf-8-sig")
    print(f"저장 완료: player_bio.csv ({len(out)}명, 키 있는 투수 {out['height_in'].notna().sum()}명)")
