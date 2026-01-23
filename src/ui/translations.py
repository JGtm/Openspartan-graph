"""Traductions UI (FR).

Ici on centralise les mappings de libellés (playlist, mode/pair) afin de:
- réduire la liste de valeurs distinctes dans l'UI
- afficher des labels en français quand on les connaît

Ce fichier est généré automatiquement depuis Playlist_modes_translations.json
puis peut être ajusté manuellement si besoin.
"""

from __future__ import annotations


PLAYLIST_FR: dict[str, str] = {
    "Big Team Battle": "Grande bataille en équipe",
    "Big Team Battle: Refresh": "Grande bataille en équipe : Refresh",
    "Big Team Social": "Grande bataille sociale",
    "Firefight": "Baptême du feu",
    "Firefight: Heroic King of the Hill": "Baptême du feu : Roi de la colline héroïque",
    "Firefight: Legendary King of the Hill": "Baptême du feu : Roi de la colline légendaire",
    "Quick Play": "Partie rapide",
    "Ranked Arena": "Arène classée",
    "Ranked Slayer": "Assassin classé",
    "Rumble Pit": "Mêlée générale",
    "SURVIVE THE UNDEAD": "Survivre aux morts-vivants",
    "Squad Battle": "Combat d'escouade",
    "Super Fiesta": "Super Fiesta",
    "Team Snipers": "Snipers en équipe",
    # IDs de playlists "Partie rapide" (fallback si nom non résolu)
    "a446725e-b281-414c-a21e": "Partie rapide",
    "bdceefb3-1c52-4848-a6b7": "Partie rapide",
}


# NOTE: ce mapping cible MatchStats.PlaylistMapModePairs (pair_name)
# Exemple: "Arena:CTF on Aquarius" -> "Arène : Capture du drapeau"
PAIR_FR: dict[str, str] = {
    # -------------------------------------------------------------------------
    # Variantes sans carte (fallback génériques)
    # -------------------------------------------------------------------------
    "Arena:CTF": "Arène : Capture du drapeau",
    "Arena:King of the Hill": "Arène : Roi de la colline",
    "Arena:Neutral Flag CTF": "Arène : Drapeau neutre",
    "Arena:Oddball": "Arène : Oddball",
    "Arena:Slayer": "Arène : Assassin",
    "Arena:Team Slayer": "Arène : Assassin en équipe",
    "Arena:Strongholds": "Arène : Bases",
    "Arena:Attrition": "Arène : Attrition",
    "Arena:One Flag CTF": "Arène : Drapeau neutre",
    "Arena:Escalation Slayer": "Arène : Escalade",
    "Arena:FFA Slayer": "Arène : Assassin FFA",
    "Arena:Shotty Snipes Slayer": "Arène : Shotty Snipers",
    "Arena:Team Snipers": "Arène : Snipers en équipe",
    "BTB:CTF": "BTB : Capture du drapeau",
    "BTB:Slayer": "BTB : Assassin",
    "BTB:Total Control": "BTB : Contrôle total",
    "BTB:Stockpile": "BTB : Stockage",
    "BTB:Fiesta CTF": "BTB : Fiesta CDD",
    "BTB:Fiesta Slayer": "BTB : Fiesta Assassin",
    "BTB:Fiesta Total Control": "BTB : Fiesta Contrôle total",
    "BTB:One Flag CTF": "BTB : Drapeau neutre",
    "BTB:Extraction": "BTB : Extraction",
    "BTB:Escalation Slayer": "BTB : Escalade",
    "BTB:Sentry Defense": "BTB : Défense sentinelle",
    "BTB:Team Snipers": "BTB : Snipers en équipe",
    "BTB Heavies:CTF": "BTB Heavies : Capture du drapeau",
    "BTB Heavies:Slayer": "BTB Heavies : Assassin",
    "BTB Heavies:Total Control": "BTB Heavies : Contrôle total",
    "Ranked:CTF": "Classé : Capture du drapeau",
    "Ranked:Slayer": "Classé : Assassin",
    "Ranked:Oddball": "Classé : Oddball",
    "Ranked:Strongholds": "Classé : Bases",
    "Ranked:King of the Hill": "Classé : Roi de la colline",
    "Tactical:Slayer": "Tactique : Assassin",
    "Community:Slayer": "Communauté : Assassin",
    "Community:Team Slayer": "Communauté : Assassin en équipe",
    "Event:Escalation Slayer": "Événement : Escalade",
    "Super Fiesta:Slayer": "Super Fiesta : Assassin",
    "Fiesta:FFA Slayer": "Fiesta : Assassin FFA",
    "Firefight:Heroic King of the Hill": "Baptême du feu : Roi de la colline héroïque",
    "Firefight:Legendary King of the Hill": "Baptême du feu : Roi de la colline légendaire",
    "Gruntpocalypse:Heroic KOTH": "Gruntpocalypse : Roi de la colline héroïque",
    "Husky Raid:CTF": "Husky Raid : CDD",
    "Super Husky Raid:CTF": "Super Husky Raid : CDD",
    "Assault:Neutral Bomb": "Arène : Bombe neutre",
    "Assault:Neutral Bomb Squad": "Arène : Escouade bombe neutre",

    # -------------------------------------------------------------------------
    # Arena
    # -------------------------------------------------------------------------
    "Arena:Attrition on Catalyst": "Arène : Attrition",
    "Arena:Attrition on Empyrean": "Arène : Attrition",
    "Arena:CTF on Aquarius": "Arène : Capture du drapeau",
    "Arena:CTF on Bazaar": "Arène : Capture du drapeau",
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
    "Arena:CTF on Shiro": "Arène : Capture du drapeau",
    "Arena:CTF on Shirov": "Arène : Capture du drapeau",
    "Arena:CTF on Snowbound": "Arène : Capture du drapeau",
    "Arena:CTF on Starboard": "Arène : Capture du drapeau",
    "Arena:CTF on Takamanohara": "Arène : Capture du drapeau",
    "Arena:CTF on Absolution": "Arène : Capture du drapeau",
    "Arena:CTF on Banished Narrows": "Arène : Capture du drapeau",
    "Arena:CTF on Catalyst": "Arène : Capture du drapeau",
    "Arena:Escalation Slayer on Argyle": "Arène : Escalade",
    "Arena:Escalation Slayer on Chasm": "Arène : Escalade",
    "Arena:Escalation Slayer on Cliffhanger": "Arène : Escalade",
    "Arena:Escalation Slayer on Detachment": "Arène : Escalade",
    "Arena:Escalation Slayer on Empyrean": "Arène : Escalade",
    "Arena:FFA Slayer on Forest - Forge": "Arène : Assassin FFA",
    "Arena:King of the Hill on Absolution": "Arène : Roi de la colline",
    "Arena:King of the Hill on Banished Narrows": "Arène : Roi de la colline",
    "Arena:King of the Hill on Behemoth": "Arène : Roi de la colline",
    "Arena:King of the Hill on Catalyst": "Arène : Roi de la colline",
    "Arena:King of the Hill on Chasm": "Arène : Roi de la colline",
    "Arena:King of the Hill on Cliffhanger": "Arène : Roi de la colline",
    "Arena:King of the Hill on Curfew": "Arène : Roi de la colline",
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
    "Arena:Neutral Flag CTF on Recharge": "Arène : Drapeau neutre",
    "Arena:Neutral Flag CTF on The Pit": "Arène : Drapeau neutre",
    "Arena:Oddball on Empyrean": "Arène : Oddball",
    "Arena:Oddball on Live Fire": "Arène : Oddball",
    "Arena:Oddball on Recharge": "Arène : Oddball",
    "Arena:Oddball on Starboard": "Arène : Oddball",
    "Arena:Oddball on Streets": "Arène : Oddball",
    "Arena:One Flag CTF on Cliffhanger": "Arène : Drapeau neutre",
    "Arena:One Flag CTF on Salvation": "Arène : Drapeau neutre",
    "Arena:Shotty Snipes Slayer on Detachment": "Arène : Shotty Snipers",
    "Arena:Slayer on Aquarius": "Arène : Assassin",
    "Arena:Slayer on Aquarius - Forge": "Arène : Assassin",
    "Arena:Slayer on Argyle": "Arène : Assassin",
    "Arena:Slayer on Bazaar - Forge": "Arène : Assassin",
    "Arena:Slayer on Behemoth": "Arène : Assassin",
    "Arena:Slayer on Behemoth - Forge": "Arène : Assassin",
    "Arena:Slayer on Chasm": "Arène : Assassin",
    "Arena:Slayer on Chasm - Forge": "Arène : Assassin",
    "Arena:Slayer on Cliffhanger": "Arène : Assassin",
    "Arena:Slayer on Cliffhanger - Forge": "Arène : Assassin",
    "Arena:Slayer on Detachment": "Arène : Assassin",
    "Arena:Slayer on Dredge": "Arène : Assassin",
    "Arena:Slayer on Ecotone": "Arène : Assassin",
    "Arena:Slayer on Empyrean": "Arène : Assassin",
    "Arena:Slayer on Forbidden - Forge": "Arène : Assassin",
    "Arena:Slayer on Forest - Forge": "Arène : Assassin",
    "Arena:Slayer on Illusion - Forge": "Arène : Assassin",
    "Arena:Slayer on Live Fire": "Arène : Assassin",
    "Arena:Slayer on Live Fire - Forge": "Arène : Assassin",
    "Arena:Slayer on Nemesis": "Arène : Assassin",
    "Arena:Slayer on Origin": "Arène : Assassin",
    "Arena:Slayer on Prism - Forge": "Arène : Assassin",
    "Arena:Slayer on Recharge": "Arène : Assassin",
    "Arena:Slayer on Recharge - Forge": "Arène : Assassin",
    "Arena:Slayer on Solitude": "Arène : Assassin",
    "Arena:Slayer on Streets - Forge": "Arène : Assassin",
    "Arena:Strongholds on Chasm": "Arène : Bases",
    "Arena:Strongholds on Cliffhanger": "Arène : Bases",
    "Arena:Strongholds on Cliffside": "Arène : Bases",
    "Arena:Strongholds on Curfew": "Arène : Bases",
    "Arena:Strongholds on Detachment": "Arène : Bases",
    "Arena:Strongholds on Domicile": "Arène : Bases",
    "Arena:Strongholds on Forest": "Arène : Bases",
    "Arena:Strongholds on Fortress": "Arène : Bases",
    "Arena:Strongholds on Houseki": "Arène : Bases",
    "Arena:Strongholds on Illusion": "Arène : Bases",
    "Arena:Strongholds on Live Fire": "Arène : Bases",
    "Arena:Strongholds on Opulence": "Arène : Bases",
    "Arena:Strongholds on Origin": "Arène : Bases",
    "Arena:Strongholds on Perilous": "Arène : Bases",
    "Arena:Strongholds on Prism": "Arène : Bases",
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
    "Arena:Team Snipers on Argyle": "Arène : Snipers en équipe",
    "Arena:Team Snipers on Bazaar": "Arène : Snipers en équipe",
    "Arena:Team Snipers on Catalyst": "Arène : Snipers en équipe",
    "Arena:Team Snipers on Chasm": "Arène : Snipers en équipe",
    "Arena:Team Snipers on Empyrean": "Arène : Snipers en équipe",
    "Arena:Team Snipers on High Ground": "Arène : Snipers en équipe",
    "Arena:Team Snipers on Isolation": "Arène : Snipers en équipe",
    "Arena:Team Snipers on Takamanohara": "Arène : Snipers en équipe",
    "Arena:VIP on Catalyst": "Arène : VIP",

    # -------------------------------------------------------------------------
    # Assault
    # -------------------------------------------------------------------------
    "Assault:Neutral Bomb Squad on Rat's Nest": "Arène : Escouade bombe neutre",
    "Assault:Neutral Bomb on Absolution": "Arène : Bombe neutre",
    "Assault:Neutral Bomb on Origin": "Arène : Bombe neutre",
    "Assault:One Bomb on Curfew": "Arène : Bombe neutre",

    # -------------------------------------------------------------------------
    # BTB (Big Team Battle)
    # -------------------------------------------------------------------------
    "BTB:CTF on Breaker": "BTB : Capture du drapeau",
    "BTB:CTF on Command": "BTB : Capture du drapeau",
    "BTB:CTF on Credence": "BTB : Capture du drapeau",
    "BTB:CTF on Deadlock": "BTB : Capture du drapeau",
    "BTB:CTF on Flood Gulch": "BTB : Capture du drapeau",
    "BTB:CTF on Fortitude": "BTB : Capture du drapeau",
    "BTB:CTF on Fragmentation": "BTB : Capture du drapeau",
    "BTB:CTF on Highpower": "BTB : Capture du drapeau",
    "BTB:CTF on Insolence": "BTB : Capture du drapeau",
    "BTB:CTF on Oasis": "BTB : Capture du drapeau",
    "BTB:CTF on Obituary": "BTB : Capture du drapeau",
    "BTB:CTF on Scarr": "BTB : Capture du drapeau",
    "BTB:CTF on Threshold": "BTB : Capture du drapeau",
    "BTB:Escalation Slayer on Insolence": "BTB : Escalade",
    "BTB:Extraction on FG": "BTB : Extraction",
    "BTB:Extraction on Refuge": "BTB : Extraction",
    "BTB:Fiesta CTF on Breaker": "BTB : Fiesta CDD",
    "BTB:Fiesta CTF on Command": "BTB : Fiesta CDD",
    "BTB:Fiesta CTF on Flood Gulch": "BTB : Fiesta CDD",
    "BTB:Fiesta CTF on Fragmentation": "BTB : Fiesta CDD",
    "BTB:Fiesta CTF on Highpower": "BTB : Fiesta CDD",
    "BTB:Fiesta CTF on Obituary": "BTB : Fiesta CDD",
    "BTB:Fiesta CTF on Threshold": "BTB : Fiesta CDD",
    "BTB:Fiesta Slayer on Breaker": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Command": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Dawnbreaker": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Deadlock": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Flood Gulch": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Fortitude": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Fragmentation": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Insolence": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Obituary": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Refuge": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Scarr": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Threshold": "BTB : Fiesta Assassin",
    "BTB:Fiesta Slayer on Thunderhead": "BTB : Fiesta Assassin",
    "BTB:Fiesta Total Control on Command": "BTB : Fiesta Contrôle total",
    "BTB:Fiesta Total Control on Fortitude": "BTB : Fiesta Contrôle total",
    "BTB:Fiesta Total Control on Refuge": "BTB : Fiesta Contrôle total",
    "BTB:One Flag CTF on Refuge": "BTB : Drapeau neutre",
    "BTB:Sentry Defense on Highpower Sentry Defense": "BTB : Défense sentinelle",
    "BTB:Sentry Defense on Oasis Sentry Defense": "BTB : Défense sentinelle",
    "BTB:Slayer on Breaker": "BTB : Assassin",
    "BTB:Slayer on Command": "BTB : Assassin",
    "BTB:Slayer on Credence": "BTB : Assassin",
    "BTB:Slayer on Deadlock": "BTB : Assassin",
    "BTB:Slayer on Fragmentation": "BTB : Assassin",
    "BTB:Slayer on Highpower": "BTB : Assassin",
    "BTB:Slayer on Oasis": "BTB : Assassin",
    "BTB:Slayer on Obituary": "BTB : Assassin",
    "BTB:Slayer on Refuge": "BTB : Assassin",
    "BTB:Slayer on Scarr": "BTB : Assassin",
    "BTB:Slayer on Threshold": "BTB : Assassin",
    "BTB:Slayer on Thunderhead": "BTB : Assassin",
    "BTB:Stockpile on Deadlock": "BTB : Stockage",
    "BTB:Stockpile on Fragmentation": "BTB : Stockage",
    "BTB:Stockpile on Highpower": "BTB : Stockage",
    "BTB:Team Snipers on Obituary": "BTB : Snipers en équipe",
    "BTB:Team Snipers on Refuge": "BTB : Snipers en équipe",
    "BTB:Total Control on Breaker": "BTB : Contrôle total",
    "BTB:Total Control on Command": "BTB : Contrôle total",
    "BTB:Total Control on Deadlock": "BTB : Contrôle total",
    "BTB:Total Control on Fragmentation": "BTB : Contrôle total",
    "BTB:Total Control on Highpower": "BTB : Contrôle total",
    "BTB:Total Control on Oasis": "BTB : Contrôle total",
    "BTB:Total Control on Obituary": "BTB : Contrôle total",
    "BTB:Total Control on Scarr": "BTB : Contrôle total",
    "BTB:Total Control on Thunderhead": "BTB : Contrôle total",

    # -------------------------------------------------------------------------
    # BTB Heavies
    # -------------------------------------------------------------------------
    "BTB Heavies:CTF on Breaker Heavies": "BTB Heavies : Capture du drapeau",
    "BTB Heavies:CTF on Fortitude Heavies": "BTB Heavies : Capture du drapeau",
    "BTB Heavies:CTF on Fragmentation Heavies": "BTB Heavies : Capture du drapeau",
    "BTB Heavies:CTF on Highpower Heavies": "BTB Heavies : Capture du drapeau",
    "BTB Heavies:CTF on Insolence Heavies": "BTB Heavies : Capture du drapeau",
    "BTB Heavies:CTF on Oasis Heavies": "BTB Heavies : Capture du drapeau",
    "BTB Heavies:CTF on Obituary Heavies": "BTB Heavies : Capture du drapeau",
    "BTB Heavies:CTF on Thunderhead Heavies": "BTB Heavies : Capture du drapeau",
    "BTB Heavies:Slayer on Breaker Heavies": "BTB Heavies : Assassin",
    "BTB Heavies:Slayer on Deadlock Heavies": "BTB Heavies : Assassin",
    "BTB Heavies:Slayer on Fortitude Heavies": "BTB Heavies : Assassin",
    "BTB Heavies:Slayer on Fragmentation Heavies": "BTB Heavies : Assassin",
    "BTB Heavies:Slayer on Oasis Heavies": "BTB Heavies : Assassin",
    "BTB Heavies:Slayer on Obituary Heavies": "BTB Heavies : Assassin",
    "BTB Heavies:Slayer on Refuge Heavies": "BTB Heavies : Assassin",
    "BTB Heavies:Slayer on Thunderhead Heavies": "BTB Heavies : Assassin",
    "BTB Heavies:Total Control on Breaker Heavies": "BTB Heavies : Contrôle total",
    "BTB Heavies:Total Control on Fragmentation Heavies": "BTB Heavies : Contrôle total",
    "BTB Heavies:Total Control on Highpower Heavies": "BTB Heavies : Contrôle total",
    "BTB Heavies:Total Control on Oasis Heavies": "BTB Heavies : Contrôle total",

    # -------------------------------------------------------------------------
    # Community
    # -------------------------------------------------------------------------
    "Community:Fiesta Slayer on High Ground": "Fiesta",
    "Community:Fiesta Slayer on Snowbound": "Fiesta",
    "Community:Shotty Snipe Slayer FFA on Dynasty": "Communauté : Shotty Snipers FFA",
    "Community:Slayer on Absolution": "Communauté : Assassin",
    "Community:Slayer on Cliffside": "Communauté : Assassin",
    "Community:Slayer on Critical Dewpoint": "Communauté : Assassin",
    "Community:Slayer on Curfew": "Communauté : Assassin",
    "Community:Slayer on Domicile": "Communauté : Assassin",
    "Community:Slayer on Dynasty": "Communauté : Assassin",
    "Community:Slayer on Fortress": "Communauté : Assassin",
    "Community:Slayer on Goliath": "Communauté : Assassin",
    "Community:Slayer on Isolation": "Communauté : Assassin",
    "Community:Slayer on Kaiketsu": "Communauté : Assassin",
    "Community:Slayer on Salvation": "Communauté : Assassin",
    "Community:Slayer on Shiro": "Communauté : Assassin",
    "Community:Slayer on Smallhalla": "Communauté : Assassin",
    "Community:Slayer on Snowbound": "Communauté : Assassin",
    "Community:Slayer on Sylvanus": "Communauté : Assassin",
    "Community:Slayer on Takamanohara": "Communauté : Assassin",
    "Community:Slayer on The Pit": "Communauté : Assassin",
    "Community:Slayer on Vagabond": "Communauté : Assassin",
    "Community:Team Slayer on Absolution": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Banished Narrows": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Cliffside": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Curfew": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Domicile": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Dynasty": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Fortress": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Goliath": "Communauté : Assassin en équipe",
    "Community:Team Slayer on High Ground": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Houseki": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Kaiketsu": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Kiken'na": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Opulence": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Perilous": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Shiro": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Shogun": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Snowbound": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Solution": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Starboard": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Sylvanus": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Takamanohara": "Communauté : Assassin en équipe",
    "Community:Team Slayer on The Pit": "Communauté : Assassin en équipe",
    "Community:Team Slayer on Vagabond": "Communauté : Assassin en équipe",

    # -------------------------------------------------------------------------
    # Event
    # -------------------------------------------------------------------------
    "Event:Escalation Slayer on Chasm": "Événement : Escalade",
    "Event:Escalation Slayer on Cliffhanger": "Événement : Escalade",
    "Event:Escalation Slayer on Streets": "Événement : Escalade",

    # -------------------------------------------------------------------------
    # Fiesta
    # -------------------------------------------------------------------------
    "Fiesta:FFA Slayer on Forest": "Fiesta : Assassin FFA",
    "Fiesta:Slayer on Behemoth - Forge": "Fiesta",
    "Fiesta:Slayer on Catalyst - Forge": "Fiesta",

    # -------------------------------------------------------------------------
    # Firefight / Gruntpocalypse
    # -------------------------------------------------------------------------
    "Firefight:Heroic King of the Hill on Oasis": "Baptême du feu : Roi de la colline héroïque",
    "Firefight:Legendary King of the Hill on Oasis": "Baptême du feu : Roi de la colline légendaire",
    "Gruntpocalypse:Heroic KOTH on Vallaheim Firefight": "Gruntpocalypse : Roi de la colline héroïque",
    "ght:Heroic King of the Hill on Vallaheim Firefight": "Baptême du feu : Roi de la colline héroïque",

    # -------------------------------------------------------------------------
    # Husky Raid / Super Husky Raid
    # -------------------------------------------------------------------------
    "Husky Raid:Assault on Urban Raid": "Husky Raid",
    "Husky Raid:CTF on Corpo": "Husky Raid : CDD",
    "Husky Raid:CTF on Merchant's Square": "Husky Raid : CDD",
    "Husky Raid:CTF on Pharaoh": "Husky Raid : CDD",
    "Super Husky Raid:CTF on Corpo": "Super Husky Raid : CDD",
    "Super Husky Raid:CTF on Disciple": "Super Husky Raid : CDD",
    "Super Husky Raid:CTF on Merchant's Square": "Super Husky Raid : CDD",
    "Super Husky Raid:CTF on Outlook": "Super Husky Raid : CDD",
    "Super Husky Raid:CTF on Pharaoh": "Super Husky Raid : CDD",
    "Super Husky Raid:CTF on Ronin": "Super Husky Raid : CDD",
    "Super Husky Raid:CTF on Warehouse": "Super Husky Raid : CDD",

    # -------------------------------------------------------------------------
    # Ranked
    # -------------------------------------------------------------------------
    "Ranked:CTF 3 Captures on Argyle": "Classé : CDD 3 captures",
    "Ranked:CTF on Aquarius": "Classé : Capture du drapeau",
    "Ranked:CTF on Empyrean": "Classé : Capture du drapeau",
    "Ranked:King of the Hill on Live Fire": "Classé : Roi de la colline",
    "Ranked:King of the Hill on Recharge": "Classé : Roi de la colline",
    "Ranked:King of the Hill on Solitude": "Classé : Roi de la colline",
    "Ranked:Oddball on Lattice - Ranked": "Classé : Oddball",
    "Ranked:Oddball on Live Fire": "Classé : Oddball",
    "Ranked:Oddball on Recharge": "Classé : Oddball",
    "Ranked:Oddball on Streets": "Classé : Oddball",
    "Ranked:Slayer on Aquarius": "Classé : Assassin",
    "Ranked:Slayer on Empyrean": "Classé : Assassin",
    "Ranked:Slayer on Forest - Ranked": "Classé : Assassin",
    "Ranked:Slayer on Live Fire": "Classé : Assassin",
    "Ranked:Slayer on Origin - Ranked": "Classé : Assassin",
    "Ranked:Slayer on Solitude - Ranked": "Classé : Assassin",
    "Ranked:Slayer on Streets - Ranked": "Classé : Assassin",
    "Ranked:Strongholds on Live Fire": "Classé : Bases",
    "Ranked:Strongholds on Streets": "Classé : Bases",

    # -------------------------------------------------------------------------
    # Super Fiesta
    # -------------------------------------------------------------------------
    "Super Fiesta:Slayer on Argyle": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Behemoth - Forge": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Catalyst - Forge": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Chasm - Forge": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Cliffhanger - Forge": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Dynasty": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Empyrean": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Forbidden - Forge": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Forest": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Forest - Forge": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Houseki": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Illusion - Forge": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Live Fire - Forge": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Opulence": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Prism - Forge": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Recharge - Forge": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Shiro": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Shogun": "Super Fiesta : Assassin",
    "Super Fiesta:Slayer on Streets - Forge": "Super Fiesta : Assassin",

    # -------------------------------------------------------------------------
    # Tactical
    # -------------------------------------------------------------------------
    "Tactical:Slayer on Aquarius - Forge": "Tactique : Assassin",
    "Tactical:Slayer on Bazaar - Forge": "Tactique : Assassin",
    "Tactical:Slayer on Cliffhanger - Forge": "Tactique : Assassin",
    "Tactical:Slayer on Cliffside": "Tactique : Assassin",
    "Tactical:Slayer on Detachment": "Tactique : Assassin",
    "Tactical:Slayer on Dredge": "Tactique : Assassin",
    "Tactical:Slayer on Illusion - Forge": "Tactique : Assassin",
    "Tactical:Slayer on Recharge": "Tactique : Assassin",
    "Tactical:Slayer on Salvation": "Tactique : Assassin",
    "Tactical:Slayer on Solitude": "Tactique : Assassin",
    "Tactical:Slayer on Starboard": "Tactique : Assassin",
    "Tactical:Slayer on Takamanohara": "Tactique : Assassin",
    "Tactical:Slayer on The Pit": "Tactique : Assassin",

    # -------------------------------------------------------------------------
    # Autres / Events spéciaux
    # -------------------------------------------------------------------------
    "urvive The Undead 3.0 on TFF | Night Of The Undead": "Survivre aux morts-vivants 3.0",
}


def translate_playlist_name(name: str | None) -> str | None:
    """Traduit un nom de playlist en français."""
    if name is None:
        return None
    s = str(name).strip()
    return PLAYLIST_FR.get(s, s)


def translate_pair_name(name: str | None) -> str | None:
    """Traduit un nom de mode/pair en français.

    Stratégie de fallback:
    1. Match exact dans PAIR_FR
    2. Normalisation de la casse et retry
    3. Match par préfixe (mode sans carte)
    4. Fallback générique pour modes Arena
    """
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
            # Préfixes multi-mots à préserver
            prefix_lower = prefix.lower()
            if prefix_lower == "btb heavies":
                prefix = "BTB Heavies"
            elif prefix_lower == "btb":
                prefix = "BTB"
            elif prefix_lower == "super fiesta":
                prefix = "Super Fiesta"
            elif prefix_lower == "super husky raid":
                prefix = "Super Husky Raid"
            elif prefix_lower == "husky raid":
                prefix = "Husky Raid"
            else:
                prefix = prefix[:1].upper() + prefix[1:].lower()
        # Si la partie mode est totalement en minuscules, on la TitleCase ("oddball" -> "Oddball").
        if rest and rest == rest.lower():
            rest = " ".join(w[:1].upper() + w[1:] for w in rest.split())
        candidate = f"{prefix}:{rest}" if prefix else rest

    if candidate in PAIR_FR:
        return PAIR_FR[candidate]

    # 3) Fallback: extraire le mode sans carte et chercher le fallback générique
    base = candidate
    mode_without_map = base
    if " on " in base:
        mode_without_map = base.split(" on ", 1)[0].strip()
    
    # Chercher le mode sans carte dans les fallbacks génériques
    if mode_without_map in PAIR_FR:
        return PAIR_FR[mode_without_map]

    # 4) Fallback générique pour tous les préfixes connus
    generic_mode_translations = {
        "Slayer": "Assassin",
        "Team Slayer": "Assassin en équipe",
        "FFA Slayer": "Assassin FFA",
        "Fiesta Slayer": "Fiesta Assassin",
        "Oddball": "Oddball",
        "CTF": "Capture du drapeau",
        "Neutral Flag CTF": "Drapeau neutre",
        "One Flag CTF": "Drapeau neutre",
        "King of the Hill": "Roi de la colline",
        "Strongholds": "Bases",
        "Attrition": "Attrition",
        "Escalation Slayer": "Escalade",
        "Team Snipers": "Snipers en équipe",
        "Shotty Snipes Slayer": "Shotty Snipers",
        "Total Control": "Contrôle total",
        "Stockpile": "Stockage",
        "Extraction": "Extraction",
        "Land Grab": "Land Grab",
        "VIP": "VIP",
    }
    
    prefix_translations = {
        "Arena": "Arène",
        "BTB": "BTB",
        "BTB Heavies": "BTB Heavies",
        "Ranked": "Classé",
        "Tactical": "Tactique",
        "Community": "Communauté",
        "Event": "Événement",
        "Fiesta": "Fiesta",
        "Super Fiesta": "Super Fiesta",
        "Firefight": "Baptême du feu",
        "Gruntpocalypse": "Gruntpocalypse",
        "Husky Raid": "Husky Raid",
        "Super Husky Raid": "Super Husky Raid",
        "Assault": "Assaut",
    }
    
    if ":" in mode_without_map:
        prefix, mode_part = mode_without_map.split(":", 1)
        prefix = prefix.strip()
        mode_part = mode_part.strip()
        
        prefix_fr = prefix_translations.get(prefix, prefix)
        mode_fr = generic_mode_translations.get(mode_part, mode_part)
        
        return f"{prefix_fr} : {mode_fr}"

    return s
