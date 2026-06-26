#!/usr/bin/env python3
"""
Mundial 2026 - Modelo de prediccion por simulacion Monte Carlo.

Basado en el enfoque del pipeline de Kaggle (kaggle/fifa_wc2026_pipeline.py,
dataset: rauffauzanrambe/fifa-world-cup-2026-prediction-system): fuerza del
equipo a partir de rating historico (Elo) + calidad de plantilla, goles por
proceso de Poisson, y simulacion completa del torneo (12 grupos, mejores
terceros, dieciseisavos -> final) condicionada a los resultados reales
ya jugados.

Salida: predictions.json
  - champion: probabilidad de ganar el Mundial por seleccion
  - reach: probabilidad de alcanzar cada ronda
  - slots: para cada cruce del cuadro (R32-0 ... F-0), top de equipos con
    probabilidad de ESTAR en ese cruce y de PASAR de ronda
  - matches: probabilidad 1X2 de los proximos cruces con equipos definidos

Uso:
  python model/predict.py                # usa data/results.json si existe
  python model/predict.py --fetch        # descarga resultados reales del API
  python model/predict.py --sims 20000
"""
import json, math, random, sys, os, urllib.request
from collections import defaultdict
from datetime import date

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_API = "https://canaldocente.es/mundial/api.php?action=results_get"

# ---------------- parametros del modelo (identicos a la web) ----------------
P = dict(wElo=0.65, homeAdv=40, expWC=30, pressure=30, climate=25, luck=0.9)
WC_CHAMPS = ["Brasil", "Alemania", "Argentina", "Francia", "Uruguay", "Inglaterra", "España"]
EUR = ["Alemania", "España", "Francia", "Inglaterra", "Países Bajos", "Portugal", "Bélgica",
       "Croacia", "Suiza", "Austria", "Noruega", "Escocia", "Chequia", "Turquía", "Suecia", "Bosnia"]
FAVS = ["España", "Argentina", "Francia", "Inglaterra", "Brasil", "Portugal", "Colombia", "Países Bajos"]
HOSTS = ["México", "Estados Unidos", "Canadá"]
GROUPS = "ABCDEFGHIJKL"
KO = ["R32", "R16", "QF", "SF", "F"]

TEAMS = json.load(open(os.path.join(BASE, "data", "teams.json"), encoding="utf-8"))


def strength(t):
    d = TEAMS[t]
    s = P["wElo"] * d["elo"] + (1 - P["wElo"]) * (1350 + (d["xi"] - 65) * 32)
    if t in WC_CHAMPS: s += P["expWC"] * 0.6
    if t in HOSTS: s += P["homeAdv"]
    if t in EUR: s -= P["climate"] * 0.5
    if t in FAVS: s -= P["pressure"] * 0.25
    return s


def exp_win(a, b):
    return 1.0 / (1.0 + 10 ** ((strength(b) - strength(a)) / 400.0))


def poisson(lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def sim_match(a, b, ko):
    ea = exp_win(a, b)
    ga = poisson(0.35 + 2.5 * ea * P["luck"])
    gb = poisson(0.35 + 2.5 * (1 - ea) * P["luck"])
    pens = None
    if ko and ga == gb:
        if random.random() < 0.45:
            if random.random() < ea: ga += 1
            else: gb += 1
        else:
            pens = "a" if random.random() < (0.5 + (ea - 0.5) * 0.4) else "b"
    return ga, gb, pens


def sim_group(teams, real_ms):
    st = {t: dict(t=t, pts=0, gf=0, gc=0) for t in teams}
    mds = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]
    for md in mds:
        for i, j in md:
            rm = None
            for m in (real_ms or []):
                if m.get("ga") in ("", None): continue
                if {m["a"], m["b"]} == {teams[i], teams[j]}:
                    rm = m; break
            if rm:
                ga, gb = (int(rm["ga"]), int(rm["gb"])) if rm["a"] == teams[i] else (int(rm["gb"]), int(rm["ga"]))
            else:
                ga, gb, _ = sim_match(teams[i], teams[j], False)
            st[teams[i]]["gf"] += ga; st[teams[i]]["gc"] += gb
            st[teams[j]]["gf"] += gb; st[teams[j]]["gc"] += ga
            if ga > gb: st[teams[i]]["pts"] += 3
            elif gb > ga: st[teams[j]]["pts"] += 3
            else: st[teams[i]]["pts"] += 1; st[teams[j]]["pts"] += 1
    return sorted(st.values(), key=lambda r: (-r["pts"], -(r["gf"] - r["gc"]), -r["gf"], random.random()))


def build_r32(first, second, T):
    return [
        (first["A"], T[0]), (second["A"], second["B"]), (first["C"], T[1]), (second["C"], second["D"]),
        (first["E"], T[2]), (second["E"], second["F"]), (first["G"], T[3]), (second["G"], second["H"]),
        (first["B"], T[4]), (first["I"], second["L"]), (first["D"], T[5]), (first["J"], second["K"]),
        (first["F"], T[6]), (first["K"], second["J"]), (first["H"], T[7]), (first["L"], second["I"]),
    ]


def _asdict(v):
    # PHP/JSON serializa un array asociativo vacio como [] (lista), no {} (objeto).
    # El API devuelve "ko": [] mientras no hay eliminatorias jugadas; normalizamos
    # a dict para poder indexar con .get() sin romper.
    return v if isinstance(v, dict) else {}


def sim_world_cup(real):
    real = real if isinstance(real, dict) else {}
    real_groups = _asdict(real.get("groups"))
    real_ko = _asdict(real.get("ko"))
    groups = {g: [t for t in TEAMS if TEAMS[t]["g"] == g] for g in GROUPS}
    first, second, thirds = {}, {}, []
    for g in GROUPS:
        tb = sim_group(groups[g], real_groups.get(g))
        first[g], second[g] = tb[0]["t"], tb[1]["t"]
        thirds.append((tb[2]["pts"], tb[2]["gf"] - tb[2]["gc"], random.random(), tb[2]["t"]))
    thirds.sort(reverse=True)
    T = [x[3] for x in thirds[:8]]
    cur = build_r32(first, second, T)
    rounds = {}
    for ri, rid in enumerate(KO):
        res, winners = [], []
        for idx, (a, b) in enumerate(cur):
            fx = real_ko.get(f"{rid}-{idx}")
            if fx and fx.get("a") and fx.get("ga") not in ("", None):
                a, b = fx["a"], fx["b"]
                ga, gb = int(fx["ga"]), int(fx["gb"])
                w = a if ga > gb else b if gb > ga else (fx.get("pens") or a)
            else:
                ga, gb, pens = sim_match(a, b, True)
                w = a if ga > gb else b if gb > ga else (a if pens == "a" else b)
            res.append((a, b, w)); winners.append(w)
        rounds[rid] = res
        if len(winners) == 1:
            return winners[0], rounds
        cur = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


def main():
    sims = 10000
    real = {}
    if "--sims" in sys.argv:
        sims = int(sys.argv[sys.argv.index("--sims") + 1])
    rj = os.path.join(BASE, "data", "results.json")
    if "--fetch" in sys.argv:
        try:
            with urllib.request.urlopen(RESULTS_API, timeout=20) as r:
                data = json.load(r)
            if data.get("ok") and isinstance(data.get("data"), dict) and data["data"].get("groups"):
                real = data["data"]
                json.dump(real, open(rj, "w", encoding="utf-8"), ensure_ascii=False)
                print("Resultados reales descargados del API")
        except Exception as e:
            print("Aviso: no se pudieron descargar resultados:", e)
    if not real and os.path.exists(rj):
        real = json.load(open(rj, encoding="utf-8"))

    champ = defaultdict(int)
    reach = {rid: defaultdict(int) for rid in KO}
    slot_in = defaultdict(lambda: defaultdict(int))
    slot_adv = defaultdict(lambda: defaultdict(int))
    for _ in range(sims):
        winner, rounds = sim_world_cup(real)
        champ[winner] += 1
        for rid, res in rounds.items():
            for idx, (a, b, w) in enumerate(res):
                sid = f"{rid}-{idx}"
                slot_in[sid][a] += 1; slot_in[sid][b] += 1
                slot_adv[sid][w] += 1
                reach[rid][a] += 1; reach[rid][b] += 1

    def top(d, n=8):
        return {k: round(v / sims, 4) for k, v in sorted(d.items(), key=lambda x: -x[1])[:n]}

    out = {
        "updated": date.today().isoformat(),
        "n_sims": sims,
        "conditioned_on_results": bool(real),
        "champion": top(champ, 48),
        "reach": {rid: top(reach[rid], 48) for rid in KO},
        "slots": {sid: {"in": top(slot_in[sid]), "advance": top(slot_adv[sid])} for sid in slot_in},
    }
    json.dump(out, open(os.path.join(BASE, "predictions.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    fav = max(champ, key=champ.get)
    print(f"OK: {sims} simulaciones. Favorito: {fav} ({champ[fav]/sims:.1%})")


if __name__ == "__main__":
    main()
