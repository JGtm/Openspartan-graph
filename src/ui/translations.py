"""Traductions UI (FR).

Ici on centralise les mappings de libellés (playlist, mode/pair) afin de:
- réduire la liste de valeurs distinctes dans l'UI
- afficher des labels en français quand on les connaît

Ce fichier n'a pas vocation à être exhaustif; on garde des fallbacks sûrs.
"""

from __future__ import annotations


PLAYLIST_FR: dict[str, str] = {
    "Quick Play": "Partie rapide",
    "Ranked Arena": "Arène classée",
    "Ranked Slayer": "Assassin classé",
}


# NOTE: ce mapping cible surtout MatchStats.PlaylistMapModePairs (pair_name)
# Exemple: "Arena:CTF on Aquarius" -> "Arène : Capture du drapeau"
PAIR_FR: dict[str, str] = {
    # Variantes sans carte (ex: payloads partiels / UI)
    "Arena:CTF": "Arène : Capture du drapeau",
    "Arena:King of the Hill": "Arène : Roi de la colline",
    "Arena:Neutral Flag CTF": "Arène : Drapeau neutre",
    "Arena:Oddball": "Arène : Oddball",
    "Arena:Team Slayer": "Arène : Assassin en équipe",

    "Arena:CTF on Absolution": "Arène : Capture du drapeau",
    "Arena:CTF on Aquarius": "Arène : Capture du drapeau",
    "Arena:CTF on Banished Narrows": "Arène : Capture du drapeau",
    "Arena:CTF on Catalyst": "Arène : Capture du drapeau",
    "Arena:CTF on Chasm": "Arène : Capture du drapeau",
    "Arena:CTF on Cliffhanger": "Arène : Capture du drapeau",
    "Arena:CTF on Cliffside": "Arène : Capture du drapeau",
    "Arena:CTF on Critical Dewpoint": "Arène : Capture du drapeau",
    "Arena:CTF on Detachment": "Arène : Capture du drapeau",
    "Arena:CTF on Domicile": "Arène : Capture du drapeau",
    "Arena:CTF on Dredge": "Arène : Capture du drapeau",
    "Arena:CTF on Dynasty": "Arène : Capture du drapeau",
    "Arena:CTF on Elevation": "Arène : Capture du drapeau",
    "Arena:CTF on Empyrean": "Arène : Capture du drapeau",
    "Arena:CTF on Forbidden": "Arène : Capture du drapeau",
    "Arena:CTF on Forest": "Arène : Capture du drapeau",
    "Arena:CTF on Fortress": "Arène : Capture du drapeau",
    "Arena:CTF on Isolation": "Arène : Capture du drapeau",
    "Arena:CTF on Nemesis": "Arène : Capture du drapeau",
    "Arena:CTF on Origin": "Arène : Capture du drapeau",
    "Arena:CTF on Shirov": "Arène : Capture du drapeau",
    "Arena:CTF on Snowbound": "Arène : Capture du drapeau",
    "Arena:CTF on Starboard": "Arène : Capture du drapeau",
    "Arena:CTF on Takamanohara": "Arène : Capture du drapeau",

    "Arena:King of the Hill on Absolution": "Arène : Roi de la colline",
    "Arena:King of the Hill on Banished Narrows": "Arène : Roi de la colline",
    "Arena:King of the Hill on Behemoth": "Arène : Roi de la colline",
    "Arena:King of the Hill on Catalyst": "Arène : Roi de la colline",
    "Arena:King of the Hill on Chasm": "Arène : Roi de la colline",
    "Arena:King of the Hill on Ecotone": "Arène : Roi de la colline",
    "Arena:King of the Hill on Elevation": "Arène : Roi de la colline",
    "Arena:King of the Hill on Empyrean": "Arène : Roi de la colline",
    "Arena:King of the Hill on Forbidden": "Arène : Roi de la colline",
    "Arena:King of the Hill on Goliath": "Arène : Roi de la colline",
    "Arena:King of the Hill on Illusion": "Arène : Roi de la colline",
    "Arena:King of the Hill on Live Fire": "Arène : Roi de la colline",
    "Arena:King of the Hill on Nemesis": "Arène : Roi de la colline",
    "Arena:King of the Hill on Opulence": "Arène : Roi de la colline",
    "Arena:King of the Hill on Salvation": "Arène : Roi de la colline",
    "Arena:King of the Hill on Shogun": "Arène : Roi de la colline",
    "Arena:King of the Hill on Snowbound": "Arène : Roi de la colline",
    "Arena:King of the Hill on Vagabond": "Arène : Roi de la colline",

    "Arena:Neutral Flag CTF on Aquarius": "Arène : Drapeau neutre",
    "Arena:Neutral Flag CTF on Bazaar": "Arène : Drapeau neutre",
    "Arena:Neutral Flag CTF on Behemoth": "Arène : Drapeau neutre",
    "Arena:Neutral Flag CTF on Detachment": "Arène : Drapeau neutre",
    "Arena:Neutral Flag CTF on Forest": "Arène : Drapeau neutre",
    "Arena:Neutral Flag CTF on Fortress": "Arène : Drapeau neutre",
    "Arena:Neutral Flag CTF on Isolation": "Arène : Drapeau neutre",
    "Arena:Neutral Flag CTF on The Pit": "Arène : Drapeau neutre",

    "Arena:Slayer on Aquarius - Forge": "Arène : Assassin",
    "Arena:Slayer on Argyle": "Arène : Assassin",
    "Arena:Slayer on Bazaar - Forge": "Arène : Assassin",
    "Arena:Slayer on Behemoth - Forge": "Arène : Assassin",
    "Arena:Slayer on Chasm - Forge": "Arène : Assassin",
    "Arena:Slayer on Cliffhanger - Forge": "Arène : Assassin",
    "Arena:Slayer on Detachment": "Arène : Assassin",
    "Arena:Slayer on Dredge": "Arène : Assassin",
    "Arena:Slayer on Ecotone": "Arène : Assassin",
    "Arena:Slayer on Empyrean": "Arène : Assassin",
    "Arena:Slayer on Forbidden - Forge": "Arène : Assassin",
    "Arena:Slayer on Forest - Forge": "Arène : Assassin",
    "Arena:Slayer on Illusion - Forge": "Arène : Assassin",
    "Arena:Slayer on Live Fire - Forge": "Arène : Assassin",
    "Arena:Slayer on Nemesis": "Arène : Assassin",
    "Arena:Slayer on Origin": "Arène : Assassin",
    "Arena:Slayer on Prism - Forge": "Arène : Assassin",
    "Arena:Slayer on Recharge - Forge": "Arène : Assassin",
    "Arena:Slayer on Solitude": "Arène : Assassin",
    "Arena:Slayer on Streets - Forge": "Arène : Assassin",

    "Arena:Strongholds on Cliffside": "Arène : Bases",
    "Arena:Strongholds on Curfew": "Arène : Bases",
    "Arena:Strongholds on Domicile": "Arène : Bases",
    "Arena:Strongholds on Forest": "Arène : Bases",
    "Arena:Strongholds on Fortress": "Arène : Bases",
    "Arena:Strongholds on Houseki": "Arène : Bases",
    "Arena:Strongholds on Illusion": "Arène : Bases",
    "Arena:Strongholds on Opulence": "Arène : Bases",
    "Arena:Strongholds on Origin": "Arène : Bases",
    "Arena:Strongholds on Perilous": "Arène : Bases",
    "Arena:Strongholds on Recharge": "Arène : Bases",
    "Arena:Strongholds on Snowbound": "Arène : Bases",
    "Arena:Strongholds on Solution": "Arène : Bases",
    "Arena:Strongholds on Streets": "Arène : Bases",
    "Arena:Strongholds on Vagabond": "Arène : Bases",

    "Arena:Team Slayer on Aquarius - Forge": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Argyle": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Behemoth - Forge": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Catalyst - Forge": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Cliffhanger - Forge": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Detachment": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Dredge": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Elevation": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Empyrean": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Forbidden - Forge": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Illusion - Forge": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Live Fire - Forge": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Nemesis": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Origin": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Prism - Forge": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Recharge - Forge": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Solitude": "Arène : Assassin en équipe",
    "Arena:Team Slayer on Streets - Forge": "Arène : Assassin en équipe",

    "Arena:Team Snipers on High Ground": "Arène : Snipers en équipe",
    "Arena:Team Snipers on Isolation": "Arène : Snipers en équipe",
    "Arena:Team Snipers on Takamanohara": "Arène : Snipers en équipe",

    "Arena:VIP on Catalyst": "Arène : VIP",

    "Assault:Neutral Bomb on Absolution": "Arène : Bombe neutre",
    "Assault:Neutral Bomb on Origin": "Arène : Bombe neutre",
    "Assault:One Bomb on Curfew": "Arène : Bombe neutre",

    "Community:Fiesta Slayer on High Ground": "Fiesta",
    "Community:Fiesta Slayer on Snowbound": "Fiesta",
    "Fiesta:Slayer on Behemoth - Forge": "Fiesta",
    "Fiesta:Slayer on Catalyst - Forge": "Fiesta",

    "Community:Slayer on Absolution": "Assassin",
    "Community:Slayer on Cliffside": "Assassin",
    "Community:Slayer on Critical Dewpoint": "Assassin",
    "Community:Slayer on Curfew": "Assassin",
    "Community:Slayer on Domicile": "Assassin",
    "Community:Slayer on Dynasty": "Assassin",
    "Community:Slayer on Fortress": "Assassin",
    "Community:Slayer on Goliath": "Assassin",
    "Community:Slayer on Isolation": "Assassin",
    "Community:Slayer on Kaiketsu": "Assassin",
    "Community:Slayer on Salvation": "Assassin",
    "Community:Slayer on Shiro": "Assassin",
    "Community:Slayer on Smallhalla": "Assassin",
    "Community:Slayer on Snowbound": "Assassin",
    "Community:Slayer on Sylvanus": "Assassin",
    "Community:Slayer on Takamanohara": "Assassin",
    "Community:Slayer on The Pit": "Assassin",
    "Community:Slayer on Vagabond": "Assassin",

    "Community:Team Slayer on Absolution": "Assassin en équipe",
    "Community:Team Slayer on Banished Narrows": "Assassin en équipe",
    "Community:Team Slayer on Cliffside": "Assassin en équipe",
    "Community:Team Slayer on Curfew": "Assassin en équipe",
    "Community:Team Slayer on Domicile": "Assassin en équipe",
    "Community:Team Slayer on Dynasty": "Assassin en équipe",
    "Community:Team Slayer on Fortress": "Assassin en équipe",
    "Community:Team Slayer on Goliath": "Assassin en équipe",
    "Community:Team Slayer on High Ground": "Assassin en équipe",
    "Community:Team Slayer on Houseki": "Assassin en équipe",
    "Community:Team Slayer on Kaiketsu": "Assassin en équipe",
    "Community:Team Slayer on Kiken'na": "Assassin en équipe",
    "Community:Team Slayer on Opulence": "Assassin en équipe",
    "Community:Team Slayer on Perilous": "Assassin en équipe",
    "Community:Team Slayer on Shiro": "Assassin en équipe",
    "Community:Team Slayer on Shogun": "Assassin en équipe",
    "Community:Team Slayer on Snowbound": "Assassin en équipe",
    "Community:Team Slayer on Solution": "Assassin en équipe",
    "Community:Team Slayer on Sylvanus": "Assassin en équipe",
    "Community:Team Slayer on Takamanohara": "Assassin en équipe",
    "Community:Team Slayer on The Pit": "Assassin en équipe",
    "Community:Team Slayer on Vagabond": "Assassin en équipe",

    "Husky Raid:Assault on Urban Raid": "Husky Raid",
    "Husky Raid:CTF on Corpo": "Husky Raid CDD",
    "Husky Raid:CTF on Pharaoh": "Husky Raid CDD",

    "Ranked:Oddball on Lattice - Ranked": "Oddball classé",
    "Ranked:Oddball on Recharge": "Oddball classé",
    "Ranked:Slayer on Origin - Ranked": "Assassin classé",
    "Ranked:Slayer on Solitude - Ranked": "Assassin classé",
    "Ranked:Slayer on Streets - Ranked": "Assassin classé",

    "Super Fiesta:Slayer on Behemoth - Forge": "Super fiesta",
    "Super Fiesta:Slayer on Catalyst - Forge": "Super fiesta",
    "Super Fiesta:Slayer on Chasm - Forge": "Super fiesta",
    "Super Fiesta:Slayer on Cliffhanger - Forge": "Super fiesta",
    "Super Fiesta:Slayer on Forbidden - Forge": "Super fiesta",
    "Super Fiesta:Slayer on Forest - Forge": "Super fiesta",
    "Super Fiesta:Slayer on Illusion - Forge": "Super fiesta",
    "Super Fiesta:Slayer on Live Fire - Forge": "Super fiesta",
    "Super Fiesta:Slayer on Prism - Forge": "Super fiesta",
    "Super Fiesta:Slayer on Recharge - Forge": "Super fiesta",
    "Super Fiesta:Slayer on Streets - Forge": "Super fiesta",

    "Super Husky Raid:CTF on Corpo": "Super husky Raid CDD",
    "Super Husky Raid:CTF on Disciple": "Super husky Raid CDD",
    "Super Husky Raid:CTF on Pharaoh": "Super husky Raid CDD",
    "Super Husky Raid:CTF on Ronin": "Super husky Raid CDD",

    "Tactical:Slayer on Aquarius - Forge": "Assassin tactique",
    "Tactical:Slayer on Bazaar - Forge": "Assassin tactique",
    "Tactical:Slayer on Cliffhanger - Forge": "Assassin tactique",
    "Tactical:Slayer on Cliffside": "Assassin tactique",
    "Tactical:Slayer on Detachment": "Assassin tactique",
    "Tactical:Slayer on Dredge": "Assassin tactique",
    "Tactical:Slayer on Illusion - Forge": "Assassin tactique",
    "Tactical:Slayer on Salvation": "Assassin tactique",
    "Tactical:Slayer on Solitude": "Assassin tactique",
    "Tactical:Slayer on Starboard": "Assassin tactique",
    "Tactical:Slayer on Takamanohara": "Assassin tactique",
    "Tactical:Slayer on The Pit": "Assassin tactique",
}


def translate_playlist_name(name: str | None) -> str | None:
    if name is None:
        return None
    s = str(name).strip()
    return PLAYLIST_FR.get(s, s)


def translate_pair_name(name: str | None) -> str | None:
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None

    # 1) Match exact
    if s in PAIR_FR:
        return PAIR_FR[s]

    # 2) Normalisation douce (casse) pour supporter des valeurs du type "arena:Team Slayer".
    candidate = s
    if ":" in s:
        prefix, rest = s.split(":", 1)
        prefix = prefix.strip()
        rest = rest.strip()
        if prefix:
            prefix = prefix[:1].upper() + prefix[1:].lower()
        # Si la partie mode est totalement en minuscules, on la TitleCase ("oddball" -> "Oddball").
        if rest and rest == rest.lower():
            rest = " ".join(w[:1].upper() + w[1:] for w in rest.split())
        candidate = f"{prefix}:{rest}" if prefix else rest

    if candidate in PAIR_FR:
        return PAIR_FR[candidate]

    # 3) Fallback: si on n'a pas la carte (pas de " on "), on cherche un pair connu avec ce préfixe.
    # Exemple: "Arena:Team Slayer" -> match "Arena:Team Slayer on <map>".
    base = candidate
    if " on " not in base and base:
        prefix_key = base + " on "
        for k, v in PAIR_FR.items():
            if k.startswith(prefix_key):
                return v

        # 4) Fallback générique pour éviter des labels techniques (ex: "arena:oddball").
        if base.startswith("Arena:"):
            rest = base.split(":", 1)[1].strip()
            arena_mode_fr = {
                "Slayer": "Assassin",
                "Team Slayer": "Assassin en équipe",
                "Oddball": "Oddball",
                "CTF": "Capture du drapeau",
                "Neutral Flag CTF": "Drapeau neutre",
                "King of the Hill": "Roi de la colline",
                "Strongholds": "Bases",
            }
            return f"Arène : {arena_mode_fr.get(rest, rest)}"

    return s
