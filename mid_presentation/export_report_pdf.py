#!/usr/bin/python3
"""Export Markdown to A4 PDF: /usr/bin/python3 export_report_pdf.py.

Uses installed ReportLab/Pillow, reads Markdown and figures without changing
sources, and atomically replaces only the report PDF. No model requests.
"""
import argparse
import html
import json
from pathlib import Path
import re
import tempfile

from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfgen.textobject import PDFTextObject
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent
INK, TEAL = colors.HexColor('#243448'), colors.HexColor('#006C66')
MUTED, RULE = colors.HexColor('#66778A'), colors.HexColor('#DCE6ED')
PALE = colors.HexColor('#EDF5F5')
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 18 * mm
TEXT_WIDTH = PAGE_WIDTH - 2 * MARGIN
GLYPHS = {}


def register_fonts():
    # Droid has Chinese but lacks Latin, digits and arrows. Select embedded
    # fonts per glyph instead of trusting viewer-specific font fallback.
    fonts = {
        'CJK': '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        'CJK-Bold': '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        'Latin': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        'Latin-Bold': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        'Mono': '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    }
    for name, path in fonts.items():
        if not Path(path).is_file():
            raise RuntimeError('Missing document font: ' + Path(path).name)
        font = TTFont(name, path)
        pdfmetrics.registerFont(font)
        GLYPHS[name] = font.face.charToGlyph
    pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJK-Bold', italic='CJK', boldItalic='CJK-Bold')
    pdfmetrics.registerFontFamily('Latin', normal='Latin', bold='Latin-Bold', italic='Latin', boldItalic='Latin-Bold')


class EmphasisText(PDFTextObject):
    def _formatText(self, text):
        rendered = super()._formatText(text)
        if self._fontname != 'CJK-Bold':
            return rendered
        # Available CJK font has one weight; a thin same-colour outline gives
        # emphasis while preserving embedded, selectable Chinese characters.
        colour = getattr(self, '_fillColorObj', INK)
        rgb = tuple(colour[:3]) if isinstance(colour, (tuple, list)) else (colour.red, colour.green, colour.blue)
        return ('%.5f %.5f %.5f RG 0.18 w 2 Tr ' % rgb) + rendered + ' 0 Tr'


class ReportCanvas(Canvas):
    def beginText(self, x=0, y=0, direction=None):
        return EmphasisText(self, x, y, direction=direction)


def font_text(text, bold=False, code=False):
    output, run, current = [], [], None
    for char in text:
        latin = 'Mono' if code else ('Latin-Bold' if bold else 'Latin')
        cjk = 'CJK-Bold' if bold else 'CJK'
        if ord(char) in GLYPHS[latin] or char.isspace():
            name = latin
        elif ord(char) in GLYPHS[cjk]:
            name = cjk
        else:
            raise ValueError('Unsupported report glyph U+%04X' % ord(char))
        if current is not None and name != current:
            output.append('<font name="%s">%s</font>' % (current, html.escape(''.join(run))))
            run = []
        current = name
        run.append(char)
    if run:
        output.append('<font name="%s">%s</font>' % (current, html.escape(''.join(run))))
    return ''.join(output)


INLINE = re.compile(r'\*\*(.+?)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)')


def inline(text, bold=False):
    output, position = [], 0
    for match in INLINE.finditer(text):
        output.append(font_text(text[position:match.start()], bold=bold))
        strong, code, label, destination = match.groups()
        if strong is not None:
            output.append('<font color="#005C56">' + inline(strong, bold=True) + '</font>')
        elif code is not None:
            output.append('<font backColor="#EEF3F7">' + font_text(code, bold=bold, code=True) + '</font>')
        else:
            content = inline(label, bold=bold)
            # Portable relative links remain relative; never resolve a host path.
            absolute = destination.startswith(('/', 'file:')) or bool(re.match(r'^[A-Za-z]:', destination))
            output.append(content if absolute else '<link href="%s" color="#2262A5">%s</link>' %
                          (html.escape(destination, quote=True), content))
        position = match.end()
    output.append(font_text(text[position:], bold=bold))
    return ''.join(output)


def styles():
    body = ParagraphStyle('Body', fontName='CJK', fontSize=10.5, leading=16, textColor=INK,
                          wordWrap='CJK', spaceAfter=7, allowWidows=0, allowOrphans=0, splitLongWords=1)
    return {
        'body': body,
        'title': ParagraphStyle('Title', parent=body, fontSize=19, leading=29, textColor=TEAL,
                                spaceAfter=10, keepWithNext=True),
        'h2': ParagraphStyle('Section', parent=body, fontSize=13.5, leading=21, textColor=TEAL,
                             spaceBefore=13, spaceAfter=8, keepWithNext=True),
        'h3': ParagraphStyle('Subsection', parent=body, fontSize=11.5, leading=18, textColor=TEAL,
                             spaceBefore=9, spaceAfter=5, keepWithNext=True),
        'date': ParagraphStyle('Date', parent=body, fontSize=9, leading=14, textColor=MUTED,
                               spaceAfter=12, keepWithNext=True),
        'caption': ParagraphStyle('Caption', parent=body, fontName='Latin', fontSize=8, leading=12,
                                  alignment=TA_CENTER, textColor=MUTED, spaceBefore=4, spaceAfter=11),
        'cell': ParagraphStyle('Cell', parent=body, fontSize=8.8, leading=13, spaceAfter=0,
                               allowWidows=1, allowOrphans=1),
        'headcell': ParagraphStyle('HeadCell', parent=body, fontSize=8.8, leading=13,
                                   textColor=TEAL, spaceAfter=0),
        'code': ParagraphStyle('Code', parent=body, fontName='Mono', fontSize=9, leading=14,
                               backColor=PALE, borderPadding=7, leftIndent=7, rightIndent=7, spaceAfter=9),
    }


def table_cells(line):
    return [cell.strip().replace(r'\|', '|') for cell in re.split(r'(?<!\\)\|', line.strip().strip('|'))]


def table_widths(headers):
    n = len(headers)
    if n == 2:
        weights = [0.34, 0.66] if headers[0] == '内容' else [0.79, 0.21]
    elif n == 3:
        weights = [0.21, 0.37, 0.42]
    elif n == 4:
        weights = [0.24, 0.15, 0.31, 0.30] if any('95%' in h for h in headers) else [0.40, 0.20, 0.20, 0.20]
    elif n == 5:
        weights = [0.36, 0.16, 0.16, 0.16, 0.16]
    else:
        weights = [1.0 / n] * n
    return [TEXT_WIDTH * w for w in weights]


def make_table(lines, style):
    rows = [table_cells(line) for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r':?-{3,}:?', c) for c in row)]
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('Markdown table column count mismatch')
    cells = [[Paragraph(inline(cell, bold=(i == 0)), style['headcell' if i == 0 else 'cell'])
              for cell in row] for i, row in enumerate(rows)]
    table = Table(cells, colWidths=table_widths(rows[0]), repeatRows=1, hAlign='LEFT', splitByRow=1)
    table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('BACKGROUND', (0, 0), (-1, 0), PALE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FAFCFD')]),
        ('LINEBELOW', (0, 0), (-1, 0), 0.8, TEAL), ('LINEBELOW', (0, 1), (-1, -1), 0.4, RULE),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    table.spaceBefore, table.spaceAfter = 5, 11
    return table


def make_figure(line, source_dir, style):
    caption, filename = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line).groups()
    path = source_dir / filename
    if not path.is_file():
        raise ValueError('Missing report figure: ' + Path(filename).name)
    with PillowImage.open(path) as image:
        width, height = image.size
    scale = min(TEXT_WIDTH / width, 92 * mm / height)
    image = Image(str(path), width=width * scale, height=height * scale)
    image.hAlign = 'CENTER'
    return KeepTogether([Spacer(1, 5), image, Paragraph(inline(caption), style['caption'])])


def parse_markdown(source, style):
    lines = source.read_text(encoding='utf-8').splitlines()
    story, pending, title, figures, tables = [], [], None, 0, 0

    def flush():
        if pending:
            text = ' '.join(pending)
            story.append(Paragraph(inline(text), style['date' if text.startswith('日期：') else 'body']))
            pending.clear()

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush()
        elif line.startswith('```'):
            flush()
            code = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith('```'):
                code.append(font_text(lines[index], code=True))
                index += 1
            story.append(Paragraph('<br/>'.join(code), style['code']))
        elif re.fullmatch(r'!\[[^\]]*\]\([^)]+\)', line):
            flush()
            story.append(make_figure(line, source.parent, style))
            figures += 1
        elif line.startswith('|'):
            flush()
            block = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                block.append(lines[index].strip())
                index += 1
            story.append(make_table(block, style))
            tables += 1
            continue
        elif re.match(r'#{1,3}\s', line):
            flush()
            marker, text = line.split(' ', 1)
            key = 'title' if marker == '#' else ('h2' if marker == '##' else 'h3')
            if marker == '#':
                if title is not None:
                    raise ValueError('Multiple document titles')
                title = text
            story.append(Paragraph(inline(text, bold=True), style[key]))
        elif re.match(r'[-*]\s', line):
            flush()
            story.append(Paragraph(font_text('• ') + inline(line[2:]), style['body']))
        else:
            pending.append(line)
        index += 1
    flush()
    if title is None:
        raise ValueError('Markdown report has no title')
    return story, title, figures, tables


def decorate_page(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_HEIGHT - 12 * mm, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 12 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont('Latin', 7)
    canvas.drawString(MARGIN, PAGE_HEIGHT - 9.5 * mm, 'CONFAGENT / MIDTERM RESEARCH REPORT')
    canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 9.5 * mm, '2026-10-20')
    canvas.line(MARGIN, 12 * mm, PAGE_WIDTH - MARGIN, 12 * mm)
    canvas.drawString(MARGIN, 8 * mm, 'ConfAgent')
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 8 * mm, str(document.page))
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=OUT / '中期研究报告.md')
    parser.add_argument('--output', type=Path, default=OUT / '中期研究报告.pdf')
    args = parser.parse_args()
    register_fonts()
    story, title, figure_count, table_count = parse_markdown(args.source, styles())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='confagent_pdf_') as temporary:
        stage = Path(temporary) / 'report.pdf'
        document = SimpleDocTemplate(str(stage), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                                     topMargin=MARGIN, bottomMargin=MARGIN, title=title, author='ConfAgent',
                                     subject='Midterm research results, evidence and limitations',
                                     allowSplitting=1, pageCompression=1)
        document.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page, canvasmaker=ReportCanvas)
        if not stage.read_bytes().startswith(b'%PDF'):
            raise RuntimeError('Invalid PDF output')
        destination = args.output.with_suffix(args.output.suffix + '.tmp')
        destination.write_bytes(stage.read_bytes())
        destination.replace(args.output)
    print(json.dumps({'pdf': args.output.name, 'converter': 'ReportLab', 'pages': document.page,
                      'figures': figure_count, 'tables': table_count, 'gpu_used': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
