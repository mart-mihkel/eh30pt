#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["python-docx==1.2.0", "reportlab==5.1.0"]
# ///
"""Export the IOS LaTeX paper to an editable document using the publisher's styles."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from docx import Document  # ty: ignore[unresolved-import]
from docx.enum.style import WD_STYLE_TYPE  # ty: ignore[unresolved-import]
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT  # ty: ignore[unresolved-import]
from docx.enum.text import WD_ALIGN_PARAGRAPH  # ty: ignore[unresolved-import]
from docx.oxml import OxmlElement  # ty: ignore[unresolved-import]
from docx.oxml.ns import qn  # ty: ignore[unresolved-import]
from docx.shared import Cm, Emu, Length, Pt  # ty: ignore[unresolved-import]
from docx.text.paragraph import Paragraph  # ty: ignore[unresolved-import]
from reportlab.pdfbase.pdfmetrics import stringWidth  # ty: ignore[unresolved-import]

if TYPE_CHECKING:
    from docx.document import Document as WordDocument  # ty: ignore[unresolved-import]
    from docx.oxml.table import CT_TrPr  # ty: ignore[unresolved-import]
    from docx.oxml.text.paragraph import CT_P  # ty: ignore[unresolved-import]
    from docx.section import Section  # ty: ignore[unresolved-import]
    from docx.styles.style import CharacterStyle, ParagraphStyle  # ty: ignore[unresolved-import]
    from docx.table import Table  # ty: ignore[unresolved-import]
    from lxml.etree import ElementBase, _Element  # ty: ignore[unresolved-import]

ROOT: Path = Path(__file__).resolve().parent.parent
TEMPLATE: Path = ROOT / "sty/IOSPressBookArticleWordTemplate.dotx"


@dataclass(frozen=True)
class Macro:
    """A parsed LaTeX macro and its source range."""

    start: int
    end: int
    argument: str
    option: str | None


class BibEntry(TypedDict):
    """A bibliography entry returned by Pandoc."""

    id: str
    DOI: NotRequired[str]


def macros(text: str, name: str) -> list[Macro]:
    """Read braced arguments without truncating nested LaTeX commands."""
    result: list[Macro] = []
    pattern: re.Pattern[str] = re.compile(r"\\" + name + r"\s*(?:\[([^]]*)\])?\s*\{")

    for match in pattern.finditer(text):
        depth: int = 1

        for token in re.finditer(r"\\.|[{}]", text[match.end() :]):
            if token.group() == "{":
                depth += 1
            elif token.group() == "}":
                depth -= 1

            if depth == 0:
                end: int = match.end() + token.start()
                result.append(
                    Macro(match.start(), end + 1, text[match.end() : end], match[1])
                )
                break
        else:
            msg = f"Unclosed \\{name} argument"
            raise ValueError(msg)

    return result


def strip_comments(text: str) -> str:
    """Remove LaTeX comments while preserving escaped percent signs."""
    return re.sub(r"(?<!\\)((?:\\\\)*)%[^\n]*", r"\1", text)


def flatten(path: Path, ancestors: tuple[Path, ...] = ()) -> str:
    """Expand nested LaTeX input files."""
    path = path.resolve()
    if path in ancestors:
        msg = f"Recursive LaTeX input: {path}"
        raise ValueError(msg)

    text: str = strip_comments(path.read_text())
    for command in reversed(macros(text, "input")):
        child: Path = path.parent / command.argument
        if not child.suffix:
            child = child.with_suffix(".tex")

        text = (
            text[: command.start]
            + flatten(child, (*ancestors, path))
            + text[command.end :]
        )

    return text


def environment(text: str, name: str) -> str:
    """Extract the contents of a named LaTeX environment."""
    return (
        text.split(r"\begin{" + name + "}", 1)[1]
        .split(r"\end{" + name + "}", 1)[0]
        .strip()
    )


def styled(style: str, text: str) -> str:
    """Wrap text in an IOS style environment."""
    return f"\\begin{{ios{style}}}\n{text}\n\\end{{ios{style}}}\n\n"


def _pandoc_path() -> str:
    path = shutil.which("pandoc")
    if path is None:
        msg = "Pandoc is not installed or not on PATH"
        raise FileNotFoundError(msg)

    return path


def bibliography(outdir: Path) -> tuple[dict[str, int], str]:
    """Build numbered references from the compiled bibliography."""
    result: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
        [_pandoc_path(), "ref.bib", "-f", "biblatex", "-t", "csljson"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    entries: list[BibEntry] = cast("list[BibEntry]", json.loads(result.stdout))
    dois: dict[str, str] = {
        entry["id"]: entry["DOI"] for entry in entries if "DOI" in entry
    }
    bbl: str = (outdir / "main.bbl").read_text()
    items: list[Macro] = macros(bbl, "bibitem")
    numbers: dict[str, int] = {}
    paragraphs: list[str] = [r"\section*{References}"]

    for index, item in enumerate(items, start=1):
        numbers[item.argument] = index
        end: int = (
            items[index].start
            if index < len(items)
            else bbl.index(r"\end{thebibliography}")
        )
        text: str = bbl[item.end : end].replace(r"\newblock", " ").strip()
        doi: str | None = dois.get(item.argument)
        if doi:
            doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
            text += rf" doi:\href{{https://doi.org/{doi}}}{{{doi}}}"

        paragraphs.append(styled("References", rf"\textnormal{{[{index}]}} {text}"))

    if not items:
        msg = "No bibliography entries; run just compile first"
        raise ValueError(msg)

    return numbers, "\n".join(paragraphs)


def _authors(front: str) -> list[str]:
    authors: list[str] = []

    for author in macros(front, "author"):
        name: str = author.argument.strip()
        for surname in reversed(macros(name, "snm")):
            name = (
                name[: surname.start] + surname.argument.upper() + name[surname.end :]
            )
        for given in reversed(macros(name, "fnms")):
            name = name[: given.start] + given.argument + name[given.end :]

        notes: list[Macro] = macros(name, "thanks")
        for note in reversed(notes):
            name = name[: note.start] + name[note.end :]

        name = name.strip() + r"\textsuperscript{" + (author.option or "").lower() + "}"
        name += "".join(r"\footnote{" + note.argument + "}" for note in notes)
        authors.append(name)

    if not authors:
        msg = "No authors found in IOS frontmatter"
        raise ValueError(msg)

    return authors


def prepare_latex(source: str, numbers: dict[str, int], references: str) -> str:
    """Convert the paper into LaTeX suitable for Pandoc."""
    for todo in reversed(macros(source, "todo")):
        source = source[: todo.start] + source[todo.end :]

    front: str = environment(source, "frontmatter")
    title: str = macros(front, "title")[0].argument
    authors = _authors(front)
    affiliations: list[str] = [
        styled(
            "Affiliation",
            r"\textsuperscript{"
            + (address.option or "").lower()
            + "} "
            + address.argument,
        )
        for address in macros(front, "address")
    ]
    abstract: str = environment(front, "abstract")
    keywords: str = environment(front, "keyword").replace(r"\sep", ", ")
    body: str = source.split(r"\end{frontmatter}", 1)[1].split(
        r"\bibliographystyle", 1
    )[0]

    labels: dict[str, int] = {}
    for kind in ("table", "figure"):
        floats: list[str] = re.findall(
            r"\\begin\{" + kind + r"\}(.*?)\\end\{" + kind + r"\}", body, re.DOTALL
        )
        for index, content in enumerate(floats, start=1):
            for label in macros(content, "label"):
                labels[label.argument] = index

    for reference in reversed(macros(body, "ref")):
        body = (
            body[: reference.start]
            + str(labels[reference.argument])
            + body[reference.end :]
        )

    result: str = "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage{graphicx,booktabs,amsmath,amssymb}",
            r"\begin{document}",
            styled("Title", title),
            styled("Author", ", ".join(authors)),
            *affiliations,
            styled("Abstract", r"\textbf{Abstract.} " + abstract),
            styled("Keywords", r"\textbf{Keywords.} " + keywords),
            body,
            references,
            r"\end{document}",
        ]
    )

    for citation in reversed(macros(result, "cite")):
        values: str = ",".join(
            str(numbers[key.strip()]) for key in citation.argument.split(",")
        )
        result = result[: citation.start] + f"[{values}]" + result[citation.end :]

    return result


def disable_numbering(style: ParagraphStyle) -> None:
    """Disable inherited numbering for a paragraph style."""
    # Pandoc does not preserve template numbering, so labels are explicit.
    properties: _Element | None = style.element.find(qn("w:pPr"))
    if properties is not None:
        for numbering in properties.findall(qn("w:numPr")):
            properties.remove(numbering)


def table_column_widths(
    table: Table,
    available_width: Length,
) -> list[Length]:
    """Size columns from Times font metrics rather than their position in this paper."""
    minimum_widths: list[float] = []
    for column_index in range(len(table.columns)):
        content_width: float = 0
        for row_index, row in enumerate(table.rows):
            for paragraph in row.cells[column_index].paragraphs:
                bold: bool = row_index == 0 or any(run.bold for run in paragraph.runs)
                italic: bool = any(run.italic for run in paragraph.runs)
                font: str = (
                    "Times-BoldItalic"
                    if bold and italic
                    else "Times-Bold"
                    if bold
                    else "Times-Italic"
                    if italic
                    else "Times-Roman"
                )
                for line in paragraph.text.splitlines():
                    content_width = max(content_width, stringWidth(line, font, 8))

        # Two points on either side are the minimum; spare width is shared evenly.
        minimum_widths.append(content_width + 4)

    total: float = sum(minimum_widths)
    spare: float = max(0, available_width.pt - total) / len(minimum_widths)
    scale: float = min(1, available_width.pt / total)
    widths: list[Length] = [Pt((width + spare) * scale) for width in minimum_widths]
    widths[-1] = Emu(available_width - sum(widths[:-1]))
    return widths


def format_table(
    table: Table,
    available_width: Length,
) -> None:
    """Apply the IOS example's compact three-rule layout to an editable table."""
    table.style = "Normal Table"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    properties: _Element = table._tbl.tblPr  # noqa: SLF001
    preferred_width: _Element | None = properties.find(qn("w:tblW"))
    if preferred_width is None:
        msg = "Table is missing its preferred width"
        raise ValueError(msg)

    preferred_width.set(qn("w:type"), "dxa")
    preferred_width.set(qn("w:w"), str(available_width.twips))

    for tag in ("w:tblCellMar", "w:tblBorders"):
        for previous in properties.findall(qn(tag)):
            properties.remove(previous)

    margins: _Element = OxmlElement("w:tblCellMar")
    for side in ("top", "left", "bottom", "right"):
        margin: _Element = OxmlElement(f"w:{side}")
        margin.set(qn("w:w"), "40" if side in ("left", "right") else "0")
        margin.set(qn("w:type"), "dxa")
        margins.append(margin)

    table._tbl.tblPr.insert_element_before(cast("ElementBase", margins), "w:tblLook")  # noqa: SLF001
    borders: _Element = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border: _Element = OxmlElement(f"w:{side}")
        border.set(qn("w:val"), "single" if side in ("top", "bottom") else "nil")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "auto")
        borders.append(border)

    table._tbl.tblPr.insert_element_before(  # noqa: SLF001
        cast("ElementBase", borders),
        "w:shd",
        "w:tblLayout",
        "w:tblCellMar",
        "w:tblLook",
    )

    _format_table_rows(table, available_width)


def _format_table_rows(table: Table, available_width: Length) -> None:
    # Explicit content-based widths render consistently in Word and LibreOffice;
    # their automatic table layout algorithms do not agree on minimum widths.
    widths: list[Length] = table_column_widths(table, available_width)
    for column, width in zip(table.columns, widths, strict=True):
        column.width = width

    for row_index, row in enumerate(table.rows):
        row_properties: CT_TrPr = row._tr.get_or_add_trPr()  # noqa: SLF001
        if row_properties.find(qn("w:cantSplit")) is None:
            row_properties.append(OxmlElement("w:cantSplit"))

        if row_index == 0 and row_properties.find(qn("w:tblHeader")) is None:
            row_properties.append(OxmlElement("w:tblHeader"))

        _format_row_cells(table, row_index, widths)


def _format_row_cells(table: Table, row_index: int, widths: list[Length]) -> None:
    row = table.rows[row_index]
    for column_index, cell in enumerate(row.cells):
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.width = widths[column_index]

        for previous in cell._tc.get_or_add_tcPr().findall(qn("w:tcBorders")):  # noqa: SLF001
            cell._tc.get_or_add_tcPr().remove(previous)  # noqa: SLF001

        if row_index == 0:
            header_borders: _Element = OxmlElement("w:tcBorders")
            rule: _Element = OxmlElement("w:bottom")
            rule.set(qn("w:val"), "single")
            rule.set(qn("w:sz"), "4")
            rule.set(qn("w:color"), "auto")
            header_borders.append(rule)
            cell._tc.get_or_add_tcPr().insert_element_before(  # noqa: SLF001
                cast("ElementBase", header_borders),
                "w:shd",
                "w:noWrap",
                "w:tcMar",
                "w:vAlign",
            )

        for paragraph in cell.paragraphs:
            paragraph.style = "Table"
            paragraph.paragraph_format.first_line_indent = Cm(0)
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1
            paragraph.paragraph_format.keep_with_next = row_index < len(table.rows) - 1
            paragraph.paragraph_format.keep_together = True
            if row_index == 0:
                paragraph.alignment = (
                    WD_ALIGN_PARAGRAPH.LEFT
                    if column_index == 0
                    else WD_ALIGN_PARAGRAPH.CENTER
                )

            for run in paragraph.runs:
                run.font.size = Pt(8)
                if row_index == 0:
                    run.bold = True


def format_document(path: Path) -> None:
    """Apply publisher formatting to the exported document."""
    document: WordDocument = Document(str(path))
    normal: ParagraphStyle = cast("ParagraphStyle", document.styles["Normal"])
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    normal.paragraph_format.line_spacing = 1
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.widow_control = True

    for candidate in document.styles:
        if candidate.type != WD_STYLE_TYPE.PARAGRAPH:
            continue

        style: ParagraphStyle = cast("ParagraphStyle", candidate)
        run_properties: _Element | None = style.element.find(qn("w:rPr"))
        if run_properties is None:
            run_properties = OxmlElement("w:rPr")
            style.element.append(run_properties)

        language: _Element | None = run_properties.find(qn("w:lang"))
        if language is None:
            language = OxmlElement("w:lang")
            run_properties.append(language)

        language.set(qn("w:val"), "en-GB")
        if (style.name or "").lower().startswith(
            "heading"
        ) or style.name == "References":
            disable_numbering(style)

        if (style.name or "").lower().startswith("heading"):
            style.paragraph_format.left_indent = Cm(0)
            style.paragraph_format.first_line_indent = Cm(0)

    references: ParagraphStyle = cast("ParagraphStyle", document.styles["References"])
    references.paragraph_format.left_indent = Cm(0.63)
    references.paragraph_format.first_line_indent = Cm(-0.63)
    references.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # The blank publisher template lacks the styles used by Pandoc for footnotes.
    footnotes: ParagraphStyle = cast(
        "ParagraphStyle",
        document.styles.add_style("Footnote Text", WD_STYLE_TYPE.PARAGRAPH),
    )
    footnotes.base_style = document.styles["Footnote"]
    footnotes.font.size = Pt(8)
    footnotes.paragraph_format.first_line_indent = Cm(0)
    marker: CharacterStyle = cast(
        "CharacterStyle",
        document.styles.add_style("Footnote Reference", WD_STYLE_TYPE.CHARACTER),
    )
    marker.font.superscript = True

    paragraphs = document.paragraphs
    _format_paragraphs(paragraphs)

    _finish_document(document, paragraphs, path)


def _format_paragraphs(paragraphs: list[Paragraph]) -> None:
    after_heading: bool = False
    for paragraph in paragraphs:
        style_name: str = (paragraph.style.name or "") if paragraph.style else ""
        if style_name in ("Normal", "Body Text", "First Paragraph"):
            paragraph.style = "NoindentNormal" if after_heading else "Normal"

        after_heading = style_name.lower().startswith("heading")
        if paragraph._p.findall(".//" + qn("w:drawing")):  # noqa: SLF001
            paragraph.paragraph_format.first_line_indent = Cm(0)
            paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(12)
            paragraph.paragraph_format.keep_with_next = True

        if style_name == "CaptionLong":
            is_table: bool = paragraph.text.startswith("Table ")
            paragraph.paragraph_format.keep_with_next = is_table
            paragraph.paragraph_format.keep_together = True
            if is_table:
                paragraph.paragraph_format.space_before = Pt(12)
            else:
                paragraph.paragraph_format.space_after = Pt(12)


def _finish_document(
    document: WordDocument, paragraphs: list[Paragraph], path: Path
) -> None:
    for image in document.inline_shapes:
        image.height = Emu(int(image.height * Cm(12.4) / image.width))
        image.width = Cm(12.4)

    section: Section = document.sections[0]
    if (
        section.page_width is None
        or section.left_margin is None
        or section.right_margin is None
    ):
        msg = "Document section is missing page width or margins"
        raise ValueError(msg)

    available_width: Length = Emu(
        section.page_width - section.left_margin - section.right_margin
    )

    for table in document.tables:
        format_table(table, available_width)

        following: _Element | None = table._tbl.getnext()  # noqa: SLF001
        while following is not None and following.tag in (
            qn("w:bookmarkStart"),
            qn("w:bookmarkEnd"),
        ):
            following = following.getnext()

        if following is not None and following.tag == qn("w:p"):
            Paragraph(
                cast("CT_P", following), document
            ).paragraph_format.space_before = Pt(12)

    document.core_properties.language = "en-GB"
    document.core_properties.title = paragraphs[0].text
    document.core_properties.author = next(
        p.text for p in paragraphs if p.style and p.style.name == "Author"
    )
    document.save(str(path))


def main() -> None:
    """Export the compiled paper to DOCX."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "build")
    args: argparse.Namespace = parser.parse_args()
    outdir: Path = cast("Path", args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    numbers: dict[str, int]
    references: str
    numbers, references = bibliography(outdir)
    source: str = flatten(ROOT / "main.tex")
    prepared: Path = outdir / "main-word.tex"
    prepared.write_text(prepare_latex(source, numbers, references))
    output: Path = outdir / "main.docx"
    subprocess.run(  # noqa: S603
        [
            _pandoc_path(),
            str(prepared),
            "--from=latex",
            "--to=docx",
            f"--reference-doc={TEMPLATE}",
            f"--lua-filter={ROOT / 'scripts/ios-docx.lua'}",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    format_document(output)


if __name__ == "__main__":
    main()
