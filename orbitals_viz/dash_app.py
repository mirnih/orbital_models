from __future__ import annotations

import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

from .grid import make_cartesian_grid, suggest_extent
from .hydrogen import psi_hydrogen_cartesian
from .viz import make_figure, make_cloud_figure
from .sampler import metropolis_sample_positions
from .snelson import make_snelson_figure


def create_app() -> Dash:
    # suppress_callback_exceptions=True помогает при сложных layout’ах,
    # но здесь мы ещё и держим оба набора контролов в layout всегда (просто скрываем).
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Orbitals Viz — Schrödinger & Snelson (illustrative)"

    sch_controls = html.Div(
        id="sch-controls",
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(6, 1fr)",
            "gap": "10px",
            "marginTop": "10px",
        },
        children=[
            html.Div([html.Label("n"), dcc.Slider(1, 6, 1, value=3, id="n")]),
            html.Div([html.Label("l"), dcc.Slider(0, 5, 1, value=1, id="l")]),
            html.Div([html.Label("m"), dcc.Slider(-5, 5, 1, value=0, id="m")]),
            html.Div([html.Label("Z (H=1, He+=2…)"), dcc.Slider(1, 6, 1, value=1, id="Z")]),
            html.Div([html.Label("Режим"), dcc.Dropdown(["cloud", "signed", "density"], value="cloud", id="mode", clearable=False)]),
            html.Div([html.Label("База"), dcc.Dropdown(
                [{"label": "реальные орбитали", "value": "real"}, {"label": "комплексные", "value": "complex"}],
                value="real", id="basis", clearable=False
            )]),

            html.Div([html.Label("Точек (cloud)"), dcc.Slider(5000, 160000, 5000, value=60000, id="points")]),
            html.Div([html.Label("Шаг Metropolis"), dcc.Slider(0.2, 3.0, 0.1, value=1.2, id="step")]),
            html.Div([html.Label("Burn-in"), dcc.Slider(0.0, 0.5, 0.02, value=0.15, id="burnin")]),

            html.Div([html.Label("Grid (iso)"), dcc.Slider(40, 120, 5, value=85, id="grid")]),
            html.Div([html.Label("Iso-quantile"), dcc.Slider(0.90, 0.999, 0.001, value=0.985, id="isoq")]),

            html.Div([html.Label("Slice plane"), dcc.Dropdown(["xy", "xz", "yz"], value="xy", id="slice_plane", clearable=False)]),
            html.Div([html.Label("Slice pos"), dcc.Slider(-10, 10, 0.2, value=0.0, id="slice_pos")]),

            html.Div([html.Label("Цвет cloud"), dcc.Dropdown(["none", "sign", "re"], value="sign", id="cloud_color", clearable=False)]),
            html.Div([html.Label("Показать ядро"), dcc.Checklist(
                options=[{"label": "да", "value": "yes"}], value=["yes"], id="sch-nuc"
            )]),
        ],
    )

    sn_controls = html.Div(
        id="sn-controls",
        style={
            "display": "none",
            "gridTemplateColumns": "repeat(6, 1fr)",
            "gap": "10px",
            "marginTop": "10px",
        },
        children=[
            html.Div([html.Label("Кол-во «электронов»-колец"), dcc.Slider(1, 40, 1, value=8, id="sn-electrons")]),
            html.Div([html.Label("Радиус оболочки"), dcc.Slider(2, 20, 0.5, value=8, id="sn-shell")]),
            html.Div([html.Label("Радиус кольца"), dcc.Slider(0.5, 6, 0.1, value=2.0, id="sn-ring")]),
            html.Div([html.Label("Амплитуда волны"), dcc.Slider(0.0, 2.0, 0.05, value=0.35, id="sn-amp")]),
            html.Div([html.Label("Частота волны k"), dcc.Slider(1, 18, 1, value=8, id="sn-k")]),
            html.Div([html.Label("Показать ядро"), dcc.Checklist(
                options=[{"label": "да", "value": "yes"}], value=["yes"], id="sn-nuc"
            )]),
        ],
    )

    app.layout = html.Div(
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "12px"},
        children=[
            html.H2("Визуализация пространственных орбиталей — две модели"),
            html.P([
                "1) ",
                html.B("Шрёдингер (водородоподобные)"),
                " — точная модель для 1 электрона (H, He+, Li2+ ...). ",
                "2) ",
                html.B("Снельсон (иллюстративно)"),
                " — геометрическая идея с «волновыми кольцами» (не квант.-мех. ψ).",
            ]),
            dcc.Tabs(
                id="tabs",
                value="tab-sch",
                children=[
                    dcc.Tab(label="Шрёдингер", value="tab-sch"),
                    dcc.Tab(label="Снельсон (иллюстративно)", value="tab-sn"),
                ],
            ),
            sch_controls,
            sn_controls,
            dcc.Loading(dcc.Graph(id="graph", style={"height": "72vh"}), type="default"),
        ],
    )

    @app.callback(
        Output("sch-controls", "style"),
        Output("sn-controls", "style"),
        Input("tabs", "value"),
    )
    def toggle_controls(tab):
        base = {"display": "grid", "gridTemplateColumns": "repeat(6, 1fr)", "gap": "10px", "marginTop": "10px"}
        hidden = {"display": "none"}
        if tab == "tab-sn":
            return hidden, base
        return base, hidden

    @app.callback(
        Output("graph", "figure"),
        Input("tabs", "value"),

        # Schrödinger
        Input("n", "value"), Input("l", "value"), Input("m", "value"), Input("Z", "value"),
        Input("mode", "value"), Input("basis", "value"),
        Input("points", "value"), Input("step", "value"), Input("burnin", "value"),
        Input("grid", "value"), Input("isoq", "value"),
        Input("slice_plane", "value"), Input("slice_pos", "value"),
        Input("cloud_color", "value"), Input("sch-nuc", "value"),

        # Snelson
        Input("sn-electrons", "value"), Input("sn-shell", "value"), Input("sn-ring", "value"),
        Input("sn-amp", "value"), Input("sn-k", "value"), Input("sn-nuc", "value"),
    )
    def update_figure(
        tab,
        n, l, m, Z, mode, basis,
        points, step, burnin,
        gridN, isoq,
        slice_plane, slice_pos,
        cloud_color, sch_nuc,
        sn_e, sn_shell, sn_ring, sn_amp, sn_k, sn_nuc
    ):
        try:
            if tab == "tab-sn":
                show_nucleus = (sn_nuc is not None and "yes" in sn_nuc)
                return make_snelson_figure(
                    electrons=int(sn_e),
                    shell_radius=float(sn_shell),
                    ring_radius=float(sn_ring),
                    wave_amplitude=float(sn_amp),
                    wave_number=int(sn_k),
                    title=f"Snelson-style (иллюстративно): N={int(sn_e)}",
                    show_nucleus=show_nucleus,
                )

            # Schrödinger
            n = int(n)
            l = int(l)
            m = int(m)

            # безопасная “подрезка” квантовых чисел
            l = max(0, min(l, n - 1))
            m = max(-l, min(m, l))

            real_orbital = (basis == "real")
            Zf = float(Z)
            extent = suggest_extent(n, Zf)
            show_nucleus = (sch_nuc is not None and "yes" in sch_nuc)

            title = f"Schrödinger: n={n}, l={l}, m={m}, Z={Zf:g}, mode={mode}"

            if mode == "cloud":
                eps = 1e-300

                def logp(r3):
                    x, y, z = r3
                    psi = psi_hydrogen_cartesian(
                        n, l, m, x, y, z,
                        nuclear_charge=Zf,
                        real_orbital=real_orbital
                    )
                    p = (abs(psi) ** 2) + eps
                    return float(np.log(p))

                samples, acc = metropolis_sample_positions(
                    logp,
                    n_samples=int(points),
                    burnin=float(burnin),
                    step_size=float(step),
                    init_box=0.6,
                    thin=1,
                )

                color_value = None
                if cloud_color in ("sign", "re"):
                    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
                    psi_vals = psi_hydrogen_cartesian(
                        n, l, m, x, y, z,
                        nuclear_charge=Zf,
                        real_orbital=real_orbital
                    )
                    re_vals = np.real(psi_vals)
                    color_value = np.sign(re_vals) if cloud_color == "sign" else re_vals

                fig = make_cloud_figure(
                    samples,
                    color_value=color_value,
                    title=title + f" | acc≈{acc:.1f}%",
                    point_size=2.0,
                    opacity=0.55,
                    show_nucleus=show_nucleus,
                )
                fig.update_layout(scene=dict(
                    xaxis_range=[-extent, extent],
                    yaxis_range=[-extent, extent],
                    zaxis_range=[-extent, extent],
                ))
                return fig

            X, Y, Zgrid, coords = make_cartesian_grid(int(gridN), extent, dtype=np.float32)
            psi = psi_hydrogen_cartesian(
                n, l, m, X, Y, Zgrid,
                nuclear_charge=Zf,
                real_orbital=real_orbital
            )

            return make_figure(
                psi=psi,
                coords=coords,
                mode=str(mode),
                iso_quantile=float(isoq),
                slice_plane=str(slice_plane),
                slice_pos=float(slice_pos),
                title=title,
                show_nucleus=show_nucleus,
            )

        except Exception as e:
            fig = go.Figure()
            fig.update_layout(title=f"Ошибка: {e}")
            return fig

    return app


def run(host: str = "127.0.0.1", port: int = 8050, debug: bool = True):
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()
