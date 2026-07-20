# Crash du visualiseur — rapport de mission

Branche : `crash-visualiser` · Plateforme : macOS (Darwin 25.4.0, Apple M5 Pro) · winit 0.30.12, wgpu 28

## Résumé

Le visualiseur ne pouvait **jamais** fonctionner sur macOS : winit exige que son event
loop vive sur le thread principal du processus, or il était créé dans un thread
secondaire. Le `panic!` correspondant tuait le thread manager et déversait un backtrace
complet par-dessus l'écran alterné de ratatui — c'est le « crash » observé.

Correction appliquée : inversion du threading. L'event loop winit possède désormais le
thread principal, le TUI et l'entraînement tournent sur un thread worker.

## 1. Reproduction

Le TUI étant interactif, un exemple minimal isole le chemin fautif sans passer par lui :

```bash
RUST_BACKTRACE=full cargo run -p main --example repro_visualiser
```

Avant le correctif (backtrace complet dans `crash.log`) :

```
[repro] gpu ready: Apple M5 Pro
[repro] spawning visualiser window...

thread '<unnamed>' panicked at winit-0.30.12/src/platform_impl/macos/event_loop.rs:221:14:
on macOS, `EventLoop` must be created on the main thread!
...
  19: winit::platform_impl::macos::event_loop::EventLoop<T>::new
         at winit-0.30.12/src/platform_impl/macos/event_loop.rs:221:14
  20: winit::event_loop::EventLoopBuilder<T>::build
  21: winit::event_loop::EventLoop<()>::new
  22: bat_building::visualiser::open_window_manager
         at bat_building/src/visualiser/mod.rs:705:28
  23: bat_building::visualiser::visualiser_manager_tx::{{closure}}::{{closure}}
         at bat_building/src/visualiser/mod.rs:169:13
  24: std::sys::backtrace::__rust_begin_short_backtrace
  25: std::thread::lifecycle::spawn_unchecked::{{closure}}::{{closure}}
```

## 2. Cause racine

**Fichier : `bat_building/src/visualiser/mod.rs`, lignes 165-173 (avant correctif).**

```rust
fn visualiser_manager_tx() -> &'static Sender<ManagerCommand> {
    VISUALISER_CMD_TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<ManagerCommand>();
        std::thread::spawn(move || {
            open_window_manager(rx);   // ← EventLoop::new() dans un thread secondaire
        });
        tx
    })
}
```

`open_window_manager` appelait `EventLoop::new()` (ligne 705). Sur macOS, winit y fait
`MainThreadMarker::new().expect("on macOS, `EventLoop` must be created on the main thread!")` :
c'est une contrainte AppKit dure (`NSApplication` est réservé au thread principal),
et **il n'existe aucun `with_any_thread` pour macOS** — contrairement à X11/Wayland,
que le code utilisait déjà pour contourner le problème sous Linux.

### Mécanisme du « crash » visible

1. `tui/mod.rs:176` appelle `warmup_visualiser()` au démarrage de chaque run.
2. Le thread manager est spawné → `EventLoop::new()` → panic immédiat.
3. Le panic n'est pas `abort` (pas de `panic = "abort"` dans les profils) : seul le
   thread manager meurt, le processus survit avec un code de sortie 0.
4. Mais le hook de panic écrit message + backtrace sur **stderr**, pendant que ratatui
   tient le terminal en raw mode + écran alterné → affichage détruit, TUI illisible.
   Pour l'utilisateur, c'est indistinguable d'un crash.
5. Effet de bord : `closed_flag` n'étant jamais posé, `is_closed()` renvoie `false` à
   vie — le contrôleur croit qu'une fenêtre existe et `[v]` ne fait plus rien du tout.

Sous Linux le bug était masqué par les appels `with_any_thread(true)`.

## 3. Correctif appliqué

macOS n'offrant pas d'échappatoire, la seule correction possible est structurelle :
**inverser le threading**.

### `bat_building/src/visualiser/mod.rs`

- Nouvelle API publique `run_on_main_thread(worker)` : installe le `Sender` du manager,
  lance `worker` sur un thread secondaire, puis fait tourner `event_loop.run_app(...)`
  sur le thread appelant (= le thread principal).
- `open_window_manager(rx)` devient `build_event_loop() -> Option<EventLoop<()>>` ;
  quand aucun backend fenêtré n'est disponible, l'application tourne simplement sans
  visualiseur au lieu d'échouer.
- Nouvelle commande `ManagerCommand::Shutdown`, envoyée depuis un **drop guard**
  (`ShutdownOnDrop`) : même si le worker panique, l'event loop est libéré et le
  processus ne se bloque pas.
- `visualiser_manager_tx()` ne spawne plus de thread ; il renvoie `Option<&Sender>`, et
  `spawn_window_with_visibility` marque proprement le handle fermé si aucun manager ne
  tourne (au lieu de laisser un handle fantôme, cf. point 5 ci-dessus).
- Les `with_any_thread(true)` Linux sont retirés : ils n'ont plus lieu d'être maintenant
  que la boucle est sur le thread principal, ce que tous les backends attendent.
- macOS : `with_activation_policy(ActivationPolicy::Accessory)`. Le TUI reste
  l'interface principale ; la fenêtre s'affiche sans ajouter d'icône au Dock ni voler
  le focus du terminal au démarrage.

### `main/src/main.rs`

```rust
fn main() {
    bat_building::visualiser::run_on_main_thread(|| {
        let config = match tui::run() { Ok(c) => c, Err(_) => return };
        run_execution_loop(config);
    });
}
```

`warmup_visualiser()` est conservé (compatibilité d'API) mais devient un simple
diagnostic : il signale si le manager n'a pas été démarré via `run_on_main_thread`.

## 4. Vérification

**Exemple `repro_visualiser`** — plus de panic ; fenêtre + surface + pipeline créés,
**310 frames présentées** en ~5 s, sortie code 0.

**Application réelle** (`./target/debug/main`, pilotée via `tmux send-keys`,
Greyscale_Diffusion, poids aléatoires, entraînement 50 000 pas) :

| Vérification | Résultat |
|---|---|
| TUI s'affiche et répond aux touches | OK (contrôle : identique au binaire pré-correctif) |
| Enregistrement source au démarrage du run | OK — fenêtre pré-créée masquée (`desired_visible=false`) |
| `[v]` → visualiseur visible | OK — frames présentées en continu (~480 en 10 s) |
| Entraînement pendant l'affichage | OK — pas 830 → 1085 → 1502, `Status: Running` |
| Bascule `[v]` off puis on | OK, pas de régression |
| **stderr** pendant tout le run | **vide** (avant : backtrace complet) |
| Sortie via `[q]` | `EXITED=0`, pas de blocage (event loop bien libéré) |

## 5. Notes

- Les sondes temporaires (`[probe] …`) utilisées pour prouver le rendu ont été retirées ;
  seule la correction subsiste dans l'arbre.
- Le correctif est également valable pour Windows et Linux, où le thread principal est
  la disposition attendue par winit — le chemin Linux X11/Wayland est conservé.
- `crash.log` conserve le backtrace d'origine à titre documentaire.
