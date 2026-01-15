# Orbitals Viz (Schrödinger + Snelson-style)

## Что внутри
- **Schrödinger (водородоподобные)**: точная ψ(n,l,m) для 1 электрона (H, He+, Li2+ ...). Режимы:
  - `cloud` (Monte-Carlo по |ψ|²),
  - `density` (изоповерхность |ψ|²),
  - `signed` (две изоповерхности для Re(ψ) со знаком).

- **Snelson-style (иллюстративно)**: геометрическая модель “волновых колец” на сфере.
  Это **не** стандартная квантовая теория; используйте как альтернативную визуальную модель.

## Быстрый старт (CLI → HTML)
```bash
pip install -r requirements.txt
python run.py --model schrodinger --n 3 --l 2 --m 0 --Z 1 --mode cloud --out orbital.html
python run.py --model schrodinger --n 2 --l 1 --m 0 --Z 2 --mode signed --out heplus.html
python run.py --model snelson --electrons 8 --shell_radius 8 --ring_radius 2 --wave_k 8 --out snelson.html
```

## Веб-приложение (Dash)
```bash
pip install -r requirements.txt
python run_dash.py
```
Откроется локально на `http://127.0.0.1:8050`.

## Размещение на сайте
- **Самый простой путь**: задеплоить Dash (Render / Railway / любой VPS). Это будет “живой” сайт.
- **Если нужен статический сайт**: генерируй готовые HTML (CLI) для набора орбиталей и вставляй в страницу через `<iframe>`.
