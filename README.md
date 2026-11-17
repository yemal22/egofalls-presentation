# EgoFalls presentation

Interactive 14-slide research draft, inspired by the Human–Machine Perception teaching decks shared by the presenter. Original self-contained HTML/CSS/JavaScript implementation, with no build dependencies.

## Present

Open `index.html` locally or the repository's GitHub Pages deployment. Arrow keys or Space advance; N toggles speaker notes; O toggles overview; F toggles fullscreen. Browser Print exports all slides.

## Edit

Each `<section>` in `index.html` is one slide. Its `data-note` contains presenter notes. Sampling and fusion illustrations are computed in the embedded script. All demonstrations are explicitly illustrative; no model performance is claimed.

## Before presenting final results

- Confirm participant assignments and one shared split protocol for both branches.
- Replace pending metrics with saved held-out predictions; include class counts and participant-level errors.
- Add measured visual examples from the exploration notebook and the actual audio architecture.
- Confirm team names and presentation date.

Current video baseline: frozen ResNet18 mean-pooled appearance features (512), eight motion summaries, and logistic regression. Fusion uses aligned video/audio probabilities, with validation-selected weight and threshold.

Design reference: https://delamarifer.github.io/human-machine-perception/day05-planb/deck.html
