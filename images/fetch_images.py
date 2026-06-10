#!/usr/bin/env python3
"""
Mundial 2026 - Recuperacion de imagenes libres de derechos de los jugadores.

Estrategia (Wikidata + Wikimedia Commons):
 1. Para cada jugador del catalogo del album (data/catalog.json) se busca su
    entidad en Wikidata (wbsearchentities, es/en).
 2. Se valida que la entidad es un futbolista (P106 = Q937857) y que tiene
    imagen (P18) alojada en Wikimedia Commons.
 3. Se recupera el autor y la licencia de la imagen (API de Commons,
    extmetadata) para cumplir la atribucion exigida por las licencias
    Creative Commons (CC BY / CC BY-SA).

Salida: images.json
  { "<equipo>|<jugador>": {"u": url_miniatura_300px, "a": "autor - licencia"} }
Las claves estan normalizadas igual que en la web (minusculas, sin acentos,
no-alfanumerico -> '-') para emparejar con las cartas.

Uso: python images/fetch_images.py [--limit N]
"""
import json, os, re, sys, time, unicodedata, urllib.parse, urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WD_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "mundial2026-album/1.0 (https://github.com/upocuantitativo/mundial2026)"}

EN2ES = {"Algeria":"Argelia","Argentina":"Argentina","Australia":"Australia","Austria":"Austria",
 "Belgium":"Bélgica","Brazil":"Brasil","Canada":"Canadá","Cape Verde":"Cabo Verde","Colombia":"Colombia",
 "Croatia":"Croacia","Ecuador":"Ecuador","Egypt":"Egipto","England":"Inglaterra","France":"Francia",
 "Germany":"Alemania","Ghana":"Ghana","Haiti":"Haití","Iran":"Irán","Ivory Coast":"Costa de Marfil",
 "Japan":"Japón","Jordan":"Jordania","Korea":"Corea del Sur","Korea Republic":"Corea del Sur",
 "Mexico":"México","Morocco":"Marruecos","Netherlands":"Países Bajos","New Zealand":"Nueva Zelanda",
 "Norway":"Noruega","Panama":"Panamá","Paraguay":"Paraguay","Portugal":"Portugal","Qatar":"Qatar",
 "Saudi Arabia":"Arabia Saudita","Scotland":"Escocia","Senegal":"Senegal","South Africa":"Sudáfrica",
 "Spain":"España","Sweden":"Suecia","Switzerland":"Suiza","Tunisia":"Túnez","Turkey":"Turquía",
 "United States":"Estados Unidos","Uruguay":"Uruguay","Uzbekistan":"Uzbekistán"}


def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def get(url, params):
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(url + "?" + qs, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


def search_player(name):
    """Devuelve lista de QIDs candidatos para un nombre."""
    qids = []
    for lang in ("es", "en"):
        try:
            d = get(WD_API, dict(action="wbsearchentities", search=name, language=lang,
                                 uselang=lang, format="json", limit=4, type="item"))
            qids += [it["id"] for it in d.get("search", [])]
        except Exception:
            pass
    seen, out = set(), []
    for q in qids:
        if q not in seen:
            seen.add(q); out.append(q)
    return out[:6]


def entities(qids):
    if not qids: return {}
    d = get(WD_API, dict(action="wbgetentities", ids="|".join(qids),
                         props="claims", format="json"))
    return d.get("entities", {})


def first_footballer_image(ents, qids):
    """Primer candidato que sea futbolista (P106 Q937857) con imagen P18."""
    for q in qids:
        e = ents.get(q, {})
        claims = e.get("claims", {})
        occs = [c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                for c in claims.get("P106", [])]
        if "Q937857" not in occs:  # futbolista
            continue
        imgs = claims.get("P18", [])
        if not imgs:
            continue
        fn = imgs[0].get("mainsnak", {}).get("datavalue", {}).get("value")
        if fn:
            return fn
    return None


def commons_meta(filenames):
    """autor + licencia por fichero (lotes de 50)."""
    meta = {}
    for i in range(0, len(filenames), 50):
        batch = filenames[i:i + 50]
        try:
            d = get(COMMONS_API, dict(action="query", titles="|".join("File:" + f for f in batch),
                                      prop="imageinfo", iiprop="extmetadata", format="json"))
            for page in d.get("query", {}).get("pages", {}).values():
                title = page.get("title", "").replace("File:", "")
                em = (page.get("imageinfo") or [{}])[0].get("extmetadata", {})
                artist = re.sub(r"<[^>]+>", "", em.get("Artist", {}).get("value", "")).strip()[:80]
                lic = em.get("LicenseShortName", {}).get("value", "")
                meta[title] = (artist, lic)
        except Exception:
            pass
        time.sleep(0.3)
    return meta


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    catalog = json.load(open(os.path.join(BASE, "data", "catalog.json"), encoding="utf-8"))
    players = []
    for c in catalog:
        team_en, name = c["t"], c["p"]
        if "Team Crest" in name:
            continue
        team = "Curazao" if team_en.startswith("Cura") else EN2ES.get(team_en)
        if not team:
            continue
        players.append((team, name))
    players = list(dict.fromkeys(players))
    if limit:
        players = players[:limit]
    print(f"{len(players)} jugadores a buscar en Wikidata...")

    out_path = os.path.join(BASE, "images.json")
    out = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    found = {}
    for n, (team, name) in enumerate(players, 1):
        key = f"{norm(team)}|{norm(name)}"
        if key in out:
            continue
        try:
            qids = search_player(name)
            fn = first_footballer_image(entities(qids), qids)
            if fn:
                found[key] = fn
                print(f"  [{n}/{len(players)}] {name} ({team}) -> {fn[:50]}")
        except Exception as e:
            print(f"  [{n}/{len(players)}] {name}: error {e}")
        time.sleep(0.15)

    meta = commons_meta(list(found.values()))
    for key, fn in found.items():
        artist, lic = meta.get(fn, ("", ""))
        url = ("https://commons.wikimedia.org/wiki/Special:FilePath/"
               + urllib.parse.quote(fn) + "?width=300")
        attrib = " - ".join(x for x in (artist, lic) if x)
        out[key] = {"u": url, "a": attrib}

    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"OK: {len(out)} imagenes en images.json (+{len(found)} nuevas)")


if __name__ == "__main__":
    main()
