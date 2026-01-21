"""Extrait les citations (commendations) de Halo 5 : Guardians depuis WikiHalo.

But:
- Générer un JSON exploitable par l'app (nom, catégorie, description, paliers, image).
- Fonctionne sans dépendances externes (stdlib seulement).

Remarque:
- Le projet actuel n'expose pas (encore) de données de "citations/commendations" dans la DB.
  Ce script sert donc à constituer un référentiel (assets + libellés) côté repo.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

BASE_URL = "https://wiki.halo.fr"
DEFAULT_URL = "https://wiki.halo.fr/index.php?title=Citations_de_Halo_5_:_Guardians&printable=yes"


def _normalize_name(s: str) -> str:
    base = " ".join(str(s or "").strip().lower().split())
    return "".join(ch for ch in unicodedata.normalize("NFKD", base) if not unicodedata.combining(ch))


def _load_exclusions(path: str | None) -> tuple[set[str], set[str]]:
    """Charge une liste d'exclusion optionnelle.

    Formats acceptés:
    - Liste JSON: ["basename.png", "Nom de citation", ...]
    - Dict JSON: {"image_basenames": [...], "names": [...], "items": [...]} (items traité comme noms)
    """
    if not path:
        return set(), set()
    p = Path(path)
    if not p.exists():
        return set(), set()

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return set(), set()

    image_basenames: set[str] = set()
    names: set[str] = set()

    def _consume(values: Any, *, as_image: bool) -> None:
        if not isinstance(values, list):
            return
        for v in values:
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s:
                continue
            if as_image:
                image_basenames.add(Path(s).name)
            else:
                names.add(_normalize_name(s))

    if isinstance(raw, list):
        # Heuristique: si ça ressemble à un nom de fichier, c'est une image, sinon un nom.
        for v in raw:
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s:
                continue
            if "." in Path(s).name:
                image_basenames.add(Path(s).name)
            else:
                names.add(_normalize_name(s))
        return image_basenames, names

    if isinstance(raw, dict):
        _consume(raw.get("image_basenames"), as_image=True)
        _consume(raw.get("names"), as_image=False)
        _consume(raw.get("items"), as_image=False)
        return image_basenames, names

    return set(), set()


def _fetch_html(url: str) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OpenSpartan-graph/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OpenSpartan-graph/1.0",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=30) as resp:
        data = resp.read()
    out_path.write_bytes(data)


_TAG_RE = re.compile(r"<[^>]+>", flags=re.S)
_BR_RE = re.compile(r"<br\s*/?>", flags=re.I)


def _strip_tags(html_fragment: str) -> str:
    txt = _BR_RE.sub("\n", html_fragment)
    txt = _TAG_RE.sub("", txt)
    txt = unescape(txt)
    return txt.replace("\xa0", " ").strip()


def _compact_spaces(s: str) -> str:
    # Conserve les retours lignes mais normalise les espaces.
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    return s.strip()


_NOTE_MARK_RE = re.compile(r"\[\s*Note\s*[^\]]*\]", flags=re.I)


def _remove_note_marks(s: str) -> str:
    return _NOTE_MARK_RE.sub("", s).strip()


_BRACKET_ANNOT_RE = re.compile(r"\[[^\]]+\]", flags=re.S)


def _strip_bracket_annotations(s: str) -> str:
    """Supprime les annotations éditoriales type [sic], [note], etc."""
    return _BRACKET_ANNOT_RE.sub("", s or "").strip()


_TRAILING_PAREN_RE = re.compile(r"^(?P<before>.*)\((?P<inside>[^()]*)\)\s*$", flags=re.S)


def _looks_like_french(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    if any(ch in s for ch in "àâäçéèêëîïôöùûüœ'’"):
        return True
    low = s.lower()
    # Mots très fréquents dans les descriptions FR.
    return any(
        k in low
        for k in [
            "tuez",
            "tuer",
            "eliminez",
            "éliminez",
            "gagner",
            "gagnez",
            "victoire",
            "obtenir",
            "obtenez",
            "protéger",
            "protegez",
            "défendez",
            "defendez",
            "terminez",
            "matchmaking",
            "partie",
            "coéquipier",
            "equipe",
        ]
    )


def _clean_description(desc: str) -> str:
    """Nettoie la description.

    - Retire les annotations entre crochets (ex: [sic]).
    - Si la chaîne contient une traduction entre parenthèses en fin de phrase,
      conserve la partie entre parenthèses si elle ressemble à du FR.
    """
    desc = _compact_spaces(_strip_bracket_annotations(desc))
    m = _TRAILING_PAREN_RE.match(desc)
    if not m:
        return desc

    inside = _compact_spaces(m.group("inside") or "")
    inside = re.sub(r"^\s*Obtener\b", "Obtenir", inside, flags=re.I)
    inside = re.sub(r"^\s*Obtener\s+", "Obtenir ", inside, flags=re.I)
    inside = _compact_spaces(inside)

    if _looks_like_french(inside):
        return inside
    return desc


@dataclass(frozen=True)
class CommendationTier:
    tier: int
    target: str
    reward: str


@dataclass(frozen=True)
class Commendation:
    category: str
    name: str
    description: str
    image_url: Optional[str]
    file_page_url: Optional[str]
    tiers: list[CommendationTier]


def _parse_int_loose(s: str) -> Optional[int]:
    s = (s or "").strip()
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


_CITATION_ICON_RE = re.compile(
    r"<img[^>]+src=\"(?P<src>/images/[^\"]*H5G_citation_[^\"]+?\.png)\"",
    flags=re.I,
)


def _extract_all_citation_icon_urls(html_text: str, base_url: str = BASE_URL) -> set[str]:
    urls: set[str] = set()
    for m in _CITATION_ICON_RE.finditer(html_text):
        src = re.sub(r"\s+", "", m.group("src") or "")
        if not src:
            continue
        urls.add(urljoin(base_url, src))
    return urls


def _extract_all_citation_icon_map(html_text: str, base_url: str = BASE_URL) -> dict[str, str]:
    """Retourne {"H5G_citation_...png": "https://.../images/<hash>/H5G_citation_...png"}."""
    out: dict[str, str] = {}
    for url in _extract_all_citation_icon_urls(html_text, base_url=base_url):
        fname = (url or "").split("/")[-1]
        if fname and fname not in out:
            out[fname] = url
    return out


_TITLE_H4_RE = re.compile(
    r"<h4>.*?<span[^>]*class=\"mw-headline\"[^>]*>\s*(?P<title>.*?)\s*</span>.*?</h4>",
    flags=re.S,
)
_TITLE_DL_RE = re.compile(r"<dl>\s*<dt>(?P<title>.*?)</dt>\s*</dl>", flags=re.S)


def _nearest_title_before(html_text: str, pos: int) -> Optional[str]:
    # Cherche dans une fenêtre pour éviter un scan complet.
    start = max(0, pos - 6000)
    window = html_text[start:pos]
    best: Optional[tuple[int, str]] = None

    for rx in (_TITLE_H4_RE, _TITLE_DL_RE):
        for m in rx.finditer(window):
            title = _compact_spaces(_strip_tags(m.group("title") or ""))
            if not title:
                continue
            p = start + m.start()
            if best is None or p > best[0]:
                best = (p, title)

    return best[1] if best else None


_H2_RE = re.compile(
    r"<h2>\s*<span[^>]*class=\"mw-headline\"[^>]*>\s*(?P<title>.*?)\s*</span>\s*</h2>",
    flags=re.S,
)

# Un bloc "citation" est celui qui a une table de paliers (wikitable).
# La page utilise plusieurs encodages pour le titre:
# - <dl><dt>Tueur d'Élites</dt></dl>
# - <h4><span class="mw-headline" id="...">Player vs Everything</span></h4>
_CITATION_BLOCK_RE = re.compile(
    r"(?:"
    r"<dl>\s*<dt>(?P<name_dl>.*?)</dt>\s*</dl>"
    r"|"
    r"<h4>.*?<span[^>]*class=\"mw-headline\"[^>]*>(?P<name_h4>.*?)</span>.*?</h4>"
    r")"
    r"\s*<p>(?P<p>(?:(?!<p>).)*?)</p>\s*"
    r"(?P<table><table[^>]*class=\"wikitable\".*?</table>)",
    flags=re.S,
)

_IMG_RE = re.compile(r"<img[^>]+src=\"(?P<src>[^\"]+)\"", flags=re.I)
_FILE_PAGE_RE = re.compile(r"<a[^>]+href=\"(?P<href>/Fichier:[^\"]+)\"", flags=re.I)

# Lignes de paliers: <th scope="row">1</th><td>10</td><td>100 EXP</td>
_TIER_ROW_RE = re.compile(
    r"<tr>\s*<th[^>]*scope=\"row\"[^>]*>\s*(?P<tier>\d+)\s*</th>\s*"
    r"<td>\s*(?P<target>.*?)\s*</td>\s*"
    r"<td>\s*(?P<reward>.*?)\s*</td>\s*</tr>",
    flags=re.S,
)


def _category_at(pos: int, headings: list[tuple[int, str]]) -> str:
    # headings: [(start_index, title), ...] trié.
    last = "(inconnu)"
    for hpos, title in headings:
        if hpos <= pos:
            last = title
        else:
            break
    return last


def parse_commendations(html_text: str, base_url: str = BASE_URL) -> list[Commendation]:
    headings = [(m.start(), _compact_spaces(_strip_tags(m.group("title")))) for m in _H2_RE.finditer(html_text)]
    headings.sort(key=lambda x: x[0])

    out: list[Commendation] = []
    for m in _CITATION_BLOCK_RE.finditer(html_text):
        start = m.start()
        category = _category_at(start, headings)

        name_html = m.group("name_dl") or m.group("name_h4") or ""
        name = _compact_spaces(_strip_tags(name_html))

        p_html = m.group("p")
        p_text = _remove_note_marks(_compact_spaces(_strip_tags(p_html)))

        img = _IMG_RE.search(p_html)
        if img:
            raw_src = img.group("src")
            raw_src = re.sub(r"\s+", "", raw_src)
            image_url = urljoin(base_url, raw_src)
        else:
            image_url = None

        file_page = _FILE_PAGE_RE.search(p_html)
        if file_page:
            raw_href = file_page.group("href")
            raw_href = re.sub(r"\s+", "", raw_href)
            file_page_url = urljoin(base_url, raw_href)
        else:
            file_page_url = None

        table_html = m.group("table")
        tiers: list[CommendationTier] = []
        for rm in _TIER_ROW_RE.finditer(table_html):
            tiers.append(
                CommendationTier(
                    tier=int(_strip_tags(rm.group("tier")) or "0"),
                    target=_remove_note_marks(_compact_spaces(_strip_tags(rm.group("target")))),
                    reward=_remove_note_marks(_compact_spaces(_strip_tags(rm.group("reward")))),
                )
            )

        # La description (dans p_text) contient typiquement juste la phrase "Tuez ...".
        description = p_text

        out.append(
            Commendation(
                category=category,
                name=name,
                description=description,
                image_url=image_url,
                file_page_url=file_page_url,
                tiers=tiers,
            )
        )

    # Dédup au cas où (certaines pages peuvent répéter des blocs via colonnes)
    unique: dict[tuple[str, str], Commendation] = {}
    for c in out:
        unique[(c.category, c.name)] = c
    return list(unique.values())


def find_missing_commendations(html_text: str, extracted: list[Commendation], base_url: str = BASE_URL) -> list[dict[str, Any]]:
    icon_map = _extract_all_citation_icon_map(html_text, base_url=base_url)

    def _basename(url: str) -> str:
        url = (url or "").strip()
        return url.split("/")[-1]

    all_files = set(icon_map.keys())
    extracted_files = {_basename(c.image_url) for c in extracted if c.image_url}

    missing_files = sorted(all_files - extracted_files)

    # Index headings once.
    headings = [(m.start(), _compact_spaces(_strip_tags(m.group("title")))) for m in _H2_RE.finditer(html_text)]
    headings.sort(key=lambda x: x[0])

    out: list[dict[str, Any]] = []
    for fname in missing_files:
        url = icon_map.get(fname) or f"{base_url}/images/{fname}"
        pos = html_text.find(fname)

        category = _category_at(pos if pos >= 0 else 10**18, headings)
        guess = _nearest_title_before(html_text, pos) if pos >= 0 else None
        out.append({"category": category, "name_guess": guess, "image_file": fname, "image_url": url})
    return out


def _image_basename_from_item(item: dict[str, Any]) -> Optional[str]:
    for k in ("image_url", "image_path", "image_file"):
        v = item.get(k)
        if not isinstance(v, str) or not v.strip():
            continue
        return v.strip().split("/")[-1].split("\\")[-1]
    return None


def _load_existing_items_for_merge(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Charge un JSON existant et retourne un index stable pour merge.

    Returns:
        (index_by_image_basename, raw_items)
    """
    if not path.exists():
        return {}, []

    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}, []

    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}, []

    idx: dict[str, dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        key = _image_basename_from_item(it)
        if key:
            idx[key] = it
    return idx, [it for it in items if isinstance(it, dict)]


def _merge_existing_item(
    generated: dict[str, Any],
    existing: dict[str, Any],
    *,
    preserve_fields: set[str],
) -> dict[str, Any]:
    # 1) Préserve explicitement certains champs (typiquement édités à la main)
    for k in preserve_fields:
        if k in existing:
            generated[k] = existing.get(k)

    # 2) Conserve toutes les clés inconnues ajoutées manuellement
    for k, v in existing.items():
        if k not in generated:
            generated[k] = v

    return generated


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrait les citations Halo 5 (commendations) depuis WikiHalo")
    ap.add_argument("--url", default=DEFAULT_URL, help="URL source (défaut: page WikiHalo imprimable)")
    ap.add_argument(
        "--input-html",
        default=None,
        help="HTML local à parser (si fourni, n'effectue pas de fetch HTTP)",
    )
    ap.add_argument(
        "--allow-network",
        action="store_true",
        help=(
            "Autorise les accès réseau (fetch HTTP et/ou téléchargement d'images). "
            "Par défaut, le script refuse toute connexion réseau."
        ),
    )
    ap.add_argument(
        "--output",
        default=str(Path("data") / "wiki" / "halo5_commendations_fr.json"),
        help="Chemin du JSON de sortie",
    )
    ap.add_argument(
        "--missing-output",
        default=str(Path("data") / "wiki" / "halo5_commendations_missing.json"),
        help="Chemin du JSON listant les citations présentes sur la page mais non extraites",
    )
    ap.add_argument(
        "--exclude",
        default=str(Path("data") / "wiki" / "halo5_commendations_exclude.json"),
        help=(
            "Chemin d'une blacklist JSON (optionnelle) pour exclure certaines citations (par nom et/ou basename d'icône). "
            "Formats: liste JSON ou dict {image_basenames: [...], names: [...]}"
        ),
    )
    ap.add_argument(
        "--no-merge-existing",
        action="store_true",
        help="N'effectue pas de fusion avec un JSON existant (écrase tout).",
    )
    ap.add_argument(
        "--clean-output",
        action="store_true",
        help="Produit un JSON 'propre' (sans URLs externes/champs redondants, nettoie [sic], garde la traduction FR).",
    )
    ap.add_argument(
        "--download-images",
        action="store_true",
        help="Télécharge les icônes en local et ajoute image_path (relatif) dans le JSON",
    )
    ap.add_argument(
        "--images-dir",
        default=str(Path("static") / "commendations" / "h5g"),
        help="Dossier local où stocker les icônes (si --download-images)",
    )
    args = ap.parse_args()

    if (not args.input_html) and (not bool(args.allow_network)):
        print("ERROR: accès réseau désactivé. Fournis --input-html ou utilise --allow-network.")
        return 2
    if bool(args.download_images) and (not bool(args.allow_network)):
        print("ERROR: --download-images nécessite --allow-network (téléchargement réseau).")
        return 2

    if args.input_html:
        html_text = Path(args.input_html).read_text(encoding="utf-8", errors="replace")
        source_url = args.url
        fetched_at = None
    else:
        html_text = _fetch_html(args.url)
        source_url = args.url
        fetched_at = datetime.now(timezone.utc).isoformat()

    comms = parse_commendations(html_text)
    missing = find_missing_commendations(html_text, comms)

    # Merge optionnel: conserve les champs édités à la main d'un JSON existant.
    out_path = Path(args.output)
    merge_existing = (not bool(args.no_merge_existing)) and out_path.exists()
    existing_by_img: dict[str, dict[str, Any]] = {}
    existing_items: list[dict[str, Any]] = []
    if merge_existing:
        existing_by_img, existing_items = _load_existing_items_for_merge(out_path)

    excluded_images, excluded_names = _load_exclusions(args.exclude)

    images_dir = Path(args.images_dir)
    payload: dict[str, Any] = {
        "source_url": source_url,
        "fetched_at": fetched_at,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": 0,
        "missing_count": len(missing),
        "items": [],
    }

    preserve_fields = {"category", "name", "description"}

    generated_items: list[dict[str, Any]] = []
    for c in sorted(comms, key=lambda x: (x.category, x.name)):
        img_key = (c.image_url or "").strip().split("/")[-1]
        if img_key and (Path(img_key).name in excluded_images):
            continue
        if _normalize_name(c.name) in excluded_names:
            continue

        item: dict[str, Any] = {
            "category": c.category,
            "name": c.name,
            "description": c.description,
            "image_url": c.image_url,
            "file_page_url": c.file_page_url,
            "max_tier": (max([t.tier for t in c.tiers]) if c.tiers else None),
            "tiers": [
                {
                    "tier": t.tier,
                    "target": t.target,
                    "target_count": _parse_int_loose(t.target),
                    "reward": t.reward,
                }
                for t in sorted(c.tiers, key=lambda x: x.tier)
            ],
        }

        if merge_existing:
            key = _image_basename_from_item(item)
            if key and key in existing_by_img:
                item = _merge_existing_item(item, existing_by_img[key], preserve_fields=preserve_fields)

        generated_items.append(item)

    # Conserve d'éventuels items ajoutés manuellement (présents dans le JSON existant mais absents de l'extraction)
    if merge_existing and existing_items:
        gen_keys = {
            _image_basename_from_item(it)
            for it in generated_items
            if isinstance(it, dict) and _image_basename_from_item(it)
        }
        for it in existing_items:
            key = _image_basename_from_item(it)
            if key and key not in gen_keys:
                if excluded_images and (Path(key).name in excluded_images):
                    continue
                if excluded_names and _normalize_name(str(it.get("name") or "")) in excluded_names:
                    continue
                generated_items.append(it)

    payload["count"] = len(generated_items)

    payload["items"] = sorted(
        generated_items,
        key=lambda x: (
            str(x.get("category") or ""),
            str(x.get("name") or ""),
        ),
    )

    # Calcule le seuil "maître" (dernier palier) par citation.
    for item in payload["items"]:
        tiers = item.get("tiers") or []
        last_cnt = None
        if tiers:
            last = sorted(tiers, key=lambda x: int(x.get("tier") or 0))[-1]
            last_cnt = last.get("target_count")
        item["master_count"] = last_cnt
        # Le label est purement UI. On ne le met pas dans le JSON "clean".
        item["master_rank_label"] = "Maître" if last_cnt is not None else None

    # Téléchargement local des icônes.
    if args.download_images:
        for item in payload["items"]:
            url = item.get("image_url")
            if not url:
                continue
            fname = url.split("/")[-1]
            # Normalise les noms Windows (certains contiennent %XX)
            img_out_path = images_dir / fname
            if not img_out_path.exists():
                _download_file(url, img_out_path)
            # Chemin relatif utilisable par Streamlit
            item["image_path"] = str(Path("static") / "commendations" / "h5g" / fname).replace("\\", "/")

    # Nettoyage final si demandé: on retire les champs externes/redondants.
    if bool(args.clean_output):
        clean_items: list[dict[str, Any]] = []
        for it in payload["items"]:
            if not isinstance(it, dict):
                continue

            name = _compact_spaces(_strip_bracket_annotations(str(it.get("name") or "")))
            desc = _clean_description(str(it.get("description") or ""))
            cat = _compact_spaces(_strip_bracket_annotations(str(it.get("category") or "")))

            tiers_in = it.get("tiers") or []
            tiers_out: list[dict[str, Any]] = []
            if isinstance(tiers_in, list):
                for t in tiers_in:
                    if not isinstance(t, dict):
                        continue
                    tiers_out.append(
                        {
                            "tier": t.get("tier"),
                            "target_count": t.get("target_count"),
                            "reward": t.get("reward"),
                        }
                    )

            clean_items.append(
                {
                    "category": cat,
                    "name": name,
                    "description": desc,
                    "tiers": tiers_out,
                    "master_count": it.get("master_count"),
                    "image_path": it.get("image_path"),
                }
            )

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(clean_items),
            "items": clean_items,
        }

    # Écrit la liste des manquants (même si vide)
    miss_path = Path(args.missing_output)
    miss_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge optionnel du fichier des "missing" (utile si tu as ajouté des annotations dessus)
    miss_items = missing
    if not bool(args.no_merge_existing) and miss_path.exists():
        existing_missing_by_img, existing_missing_items = _load_existing_items_for_merge(miss_path)
        if existing_missing_items:
            merged: list[dict[str, Any]] = []
            seen: set[str] = set()
            for it in miss_items:
                key = _image_basename_from_item(it)
                if key and key in existing_missing_by_img:
                    it = _merge_existing_item(it, existing_missing_by_img[key], preserve_fields=set())
                if key:
                    seen.add(key)
                merged.append(it)
            for it in existing_missing_items:
                key = _image_basename_from_item(it)
                if key and key not in seen:
                    merged.append(it)
            miss_items = merged

    miss_payload = {
        "source_url": source_url,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "missing_count": len(miss_items),
        "items": miss_items,
    }

    if bool(args.clean_output):
        # Fichier "missing" nettoyé aussi (structure simple)
        miss_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "missing_count": len(miss_items),
            "items": miss_items,
        }
    miss_path.write_text(json.dumps(miss_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with_images = sum(1 for c in comms if c.image_url)
    print(f"OK: {len(comms)} citations extraites ({with_images} avec image)")
    print(f"Manquantes (présentes sur la page mais non extraites): {len(missing)}")
    print(f"JSON: {out_path}")
    print(f"Missing JSON: {miss_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
