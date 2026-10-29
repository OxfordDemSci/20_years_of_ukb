"""D55: one figure style for every analysis.

Every figure is drawn at one width on a fixed height ladder, with one set of font sizes,
one colormap, black edges and a fixed set of greys. The style itself lives in
`universal_settings.yml` `style.base`; these tests stop a plotting module or notebook cell
from drifting away from it again with a literal.
"""
import ast
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import shared_style as S  # noqa: E402

#: Unimported legacy module with no live caller (STATE question 5).
EXEMPT = {"data_analysis_04_non_academic_patents_ipc_dendrogram.py"}
MODULES = sorted(p for p in (ROOT / "src" / "utils").glob("data_analysis_*.py")
                 if p.name not in EXEMPT) + [ROOT / "src" / "utils" / "shared_patent_utils.py"]
NOTEBOOKS = sorted((ROOT / "src" / "data_analysis").glob("0[0-5]_*.ipynb"))

FONT_KW = {"fontsize", "labelsize", "title_fontsize"}
EDGE_KW = {"edgecolor", "edgecolors", "markeredgecolor", "mec"}
EDGE_VALUES = {"black", "white", "none", "face"}
CMAP_BUILDERS = {"get_cmap", "from_list", "light_palette", "dark_palette", "diverging_palette",
                 "blend_palette", "ListedColormap"}
ALLOWED_HEX = {c.upper() for c in [*S.PALETTE, *S.NON_ACADEMIC_PALETTE, *S.NEUTRALS.values()]}
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _is_number(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
        and not isinstance(node.value, bool)


def _reads_font_size(node):
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith("_fs")
               for n in ast.walk(node))


def _func(call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def violations(source):
    """Every literal in `source` that bypasses the shared style, as (line, message)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = _func(node)
            if name in CMAP_BUILDERS:
                found.append((node.lineno, f"builds its own colormap via {name}()"))
            for kw in node.keywords:
                value = kw.value
                if kw.arg in FONT_KW and (_is_number(value) or (
                        isinstance(value, ast.BinOp) and _reads_font_size(value))):
                    found.append((node.lineno, f"{kw.arg}={ast.unparse(value)}"))
                elif kw.arg == "figsize" and isinstance(value, ast.Tuple) \
                        and all(_is_number(e) for e in value.elts):
                    found.append((node.lineno, f"figsize={ast.unparse(value)}"))
                elif kw.arg in EDGE_KW and isinstance(value, ast.Constant) \
                        and isinstance(value.value, str) and value.value.lower() not in EDGE_VALUES:
                    found.append((node.lineno, f"{kw.arg}={value.value!r}"))
                elif kw.arg in EDGE_KW and isinstance(value, ast.Call) and _func(value) == "palette":
                    found.append((node.lineno, f"{kw.arg}={ast.unparse(value)}: edges are black"))
                elif kw.arg == "cmap" and isinstance(value, ast.Constant) and isinstance(value.value, str):
                    found.append((node.lineno, f"named colormap {value.value!r}"))
                elif kw.arg == "dpi" and name in {"figure", "subplots"}:
                    found.append((node.lineno, "figure-level dpi"))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) \
                        and key.value.endswith("_fs") and _is_number(value):
                    found.append((node.lineno, f"style override {key.value!r}: {value.value}"))
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr.startswith("_ukb_") and t.attr.endswith("_fs")
                for t in node.targets) and _is_number(node.value):
            found.append((node.lineno, f"{ast.unparse(node.targets[0])} = {ast.unparse(node.value)}"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and HEX.match(node.value) \
                and node.value.upper() not in ALLOWED_HEX:
            found.append((node.lineno, f"off-palette colour {node.value}"))
    return found


def notebook_code(path):
    cells = json.loads(path.read_text())["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            # IPython magics and shell escapes are not Python.
            source = "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                               for line in source.splitlines())
            yield index, source


class LiteralStyleTests(unittest.TestCase):
    def test_plotting_modules_take_every_size_and_colour_from_the_style(self):
        for path in MODULES:
            with self.subTest(module=path.name):
                self.assertEqual(violations(path.read_text()), [])

    def test_notebook_cells_take_every_size_and_colour_from_the_style(self):
        for path in NOTEBOOKS:
            for index, source in notebook_code(path):
                with self.subTest(notebook=path.name, cell=index):
                    self.assertEqual(violations(source), [])

    def test_the_guard_catches_each_kind_of_literal(self):
        caught = violations(
            'ax.set_xlabel("x", fontsize=12)\n'
            'ax.text(0, 0, "y", fontsize=style["annot_fs"] - 2)\n'
            'plt.subplots(figsize=(10, 6), dpi=300)\n'
            'ax.bar(x, y, edgecolor="grey")\n'
            'ax.imshow(m, cmap="Blues")\n'
            'LinearSegmentedColormap.from_list("m", ["w", "k"])\n'
            'ax._ukb_label_fs = 17\n'
            'c = "#6B7280"\n'
            'cstyle = {**STYLE, "label_fs": 17}\n'
            'ax.bar(x, y, edgecolor=palette("navy"))\n')
        self.assertEqual(len(caught), 11)


class StyleSectionTests(unittest.TestCase):
    SHARED = ("fig_width", "fig_aspects", "panel_label_fs", "title_fs", "label_fs", "tick_fs",
              "legend_fs", "annot_fs", "small_fs", "body_fs", "colorbar_label_fs", "font_family",
              "edgecolor", "separator_color", "edge_linewidth", "line_width", "ref_linewidth",
              "axes_linewidth", "tick_length", "tick_width", "neutral_colors")

    def test_every_notebook_section_has_identical_typography_strokes_and_sizes(self):
        sections = S.load_settings()["style"]["notebooks"]
        loaded = {name: S.load_style(name, activate=False) for name in sections}
        reference = {k: loaded["01_growth"][k] for k in self.SHARED}
        for name, style in loaded.items():
            with self.subTest(section=name):
                self.assertEqual({k: style[k] for k in self.SHARED}, reference)

    def test_four_font_sizes_in_all(self):
        style = S.load_style("02_content", activate=False)
        sizes = {S.font_size(role, style) for role in S.FONT_ROLES}
        self.assertEqual(sizes, {22, 16, 14, 13})

    def test_a_section_that_overrides_typography_is_a_crash(self):
        yaml = ("style:\n  base: {label_fs: 16, colors: ['#E66859']}\n"
                "  notebooks:\n    x: {label_fs: 18}\n")
        with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as fh:
            fh.write(yaml)
        with self.assertRaisesRegex(KeyError, "label_fs"):
            S.load_style("x", path=fh.name, activate=False)

    def test_the_yaml_and_the_module_constants_agree(self):
        base = S.load_settings()["style"]["base"]
        self.assertEqual(base["fig_width"], S.FIG_WIDTH)
        self.assertEqual(base["fig_aspects"], S.FIG_ASPECTS)
        self.assertEqual(base["neutral_colors"], S.NEUTRALS)
        self.assertEqual(base["edgecolor"], S.EDGE_COLOR)
        self.assertEqual(base["separator_color"], S.SEPARATOR_COLOR)


class FigureGeometryTests(unittest.TestCase):
    def test_figsize_is_the_project_width_on_a_ladder_rung(self):
        for name, ratio in S.FIG_ASPECTS.items():
            self.assertEqual(S.figsize(name), (S.FIG_WIDTH, S.FIG_WIDTH * ratio))
        self.assertEqual(S.figsize(0.66), S.figsize("landscape"))
        with self.assertRaises(KeyError):
            S.figsize("poster")

    def test_every_export_is_exactly_the_project_width(self):
        style = {**S.load_style("02_content", activate=False), "save": True, "formats": ["pdf"]}
        with tempfile.TemporaryDirectory() as tmp:
            style["savedir"] = tmp
            for labels in ("short", "a much longer category label " * 3):
                fig, ax = S.panel_grid(1, 1, style, aspect="landscape", layout="constrained")
                ax.barh([labels, "b"], [1, 2])
                (path,) = S.savefig(fig, "probe", style=style)
                plt.close(fig)
                self.assertAlmostEqual(self._pdf_width(path), S.FIG_WIDTH, places=2)
            # Content spilling past the canvas is a layout defect: exported uncropped, loudly.
            fig, ax = S.panel_grid(1, 1, style, aspect="landscape")
            ax.barh(["a much longer category label " * 6, "b"], [1, 2])
            with self.assertWarnsRegex(UserWarning, "wider than the 18in project width"):
                (path,) = S.savefig(fig, "overflow", style=style)
            plt.close(fig)
            self.assertGreater(self._pdf_width(path), S.FIG_WIDTH)

    @staticmethod
    def _pdf_width(path):
        box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+[\d.]+\s+([\d.]+)", path.read_bytes())
        return (float(box[2]) - float(box[1])) / 72

    def test_there_is_one_colormap_and_it_paints_missing_cells_white(self):
        cmap = S.figure_colormap()
        self.assertIs(S.blue_cream_red_colormap, S.figure_colormap)
        self.assertEqual(cmap.name, "blue_cream_red")
        self.assertEqual(cmap.get_bad().tolist(), [1.0, 1.0, 1.0, 1.0])
        for retired in ("warm_cold_colormap", "academic_impact_colormap", "blue_colormap",
                        "author_geography_colormap", "sequential_colormap"):
            self.assertFalse(hasattr(S, retired), retired)


if __name__ == "__main__":
    unittest.main()
