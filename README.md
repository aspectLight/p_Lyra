# INF8175 — Projet Lyra · Hex MCTS

**Dépôt GitHub :** [`p_Lyra`](https://github.com/aspectLight/p_Lyra)

Agent Hex **Lyra** pour la compétition **[Abyss](https://abyss-a24.corail-lab.ca/)** — projet INF8175 (A2025), CORAIL Lab, Polytechnique Montréal.

**Authors:** Mathis Ors (2206596), Ahmed Sami Benabbou (2122457)

## Dépôts INF8175 (même convention de nommage)

| Travail | Dépôt |
|---------|--------|
| Devoir 1 — Pacman search | `inf8175-devoir1-pacman` |
| Devoir 2 — Contraintes & recherche locale | `inf8175-devoir2-a25` |
| Projet Hex — Lyra | [`p_Lyra`](https://github.com/aspectLight/p_Lyra) |

## Approach

Monte Carlo Tree Search with RAVE, pattern-based priors and playouts, inferior cell elimination (ICE), and virtual connection search.

## Setup

```bash
pip install -r requirements.txt
```

## Submit

1. Agent entry point : `my_player.py` (`MyPlayer` class).
2. Upload on [Abyss](https://abyss-a24.corail-lab.ca/).
3. Board size : **14×14**

Spécification : [Projet_Hex_A2025.pdf](Projet_Hex_A2025.pdf)

## Layout

```
my_player.py              # Seahorse interface → Abyss
src_2206596_2122457/      # MCTS, ICE, VC, patterns
requirements.txt
```
