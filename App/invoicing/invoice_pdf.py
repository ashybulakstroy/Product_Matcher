"""Local payment invoice; all financial values come from the screen model."""
from decimal import Decimal
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether

from App.invoicing.invoice_presentation import print_fields, suggestion_fill


def number(value):
    return f"{Decimal(str(value)):,.2f}".replace(",", " ").replace(".", ",")


@lru_cache(maxsize=1)
def register_fonts():
    candidates = [(Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
                  (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("Invoice", str(regular)))
            pdfmetrics.registerFont(TTFont("InvoiceBold", str(bold)))
            pdfmetrics.registerFontFamily("Invoice", normal="Invoice", bold="InvoiceBold")
            return
    raise RuntimeError("Не найден шрифт с поддержкой кириллицы для PDF")


def render_invoice_pdf(document, *, allow_draft=False):
    if not document["ready"] and not allow_draft:
        raise ValueError("; ".join(document["errors"]))
    register_fonts()
    output = BytesIO()
    width = 178*mm
    pdf = SimpleDocTemplate(output, pagesize=A4, leftMargin=10*mm, rightMargin=22*mm,
                            topMargin=11*mm, bottomMargin=16*mm,
                            title=f"Счет на оплату {document['number']}", author=document["issuer"].get("name") or "")
    normal = ParagraphStyle("body", fontName="Invoice", fontSize=9, leading=11)
    bold = ParagraphStyle("bold", parent=normal, fontName="InvoiceBold")
    heading = ParagraphStyle("title", parent=bold, fontSize=14, leading=17, spaceAfter=0)
    right = ParagraphStyle("number", parent=normal, alignment=TA_RIGHT)
    right_bold = ParagraphStyle("number-bold", parent=bold, alignment=TA_RIGHT)
    center = ParagraphStyle("center", parent=bold, alignment=TA_CENTER)
    small = ParagraphStyle("small", parent=normal, fontSize=8, leading=9)
    notice = ParagraphStyle("notice", parent=small, alignment=TA_CENTER, leftIndent=30*mm)
    validity = ParagraphStyle("validity", parent=center, textColor=colors.red)
    tax = ParagraphStyle("tax", parent=center, fontSize=11, leading=13)
    def p(value, style=normal):
        return Paragraph(escape(str(value or "")), style)
    issuer = document["issuer"]
    form = print_fields(document)
    story = []
    story += [p(form["terms"], notice), Spacer(1, 5*mm), p(form["validity"], validity),
             Spacer(1, 5*mm), p("Образец платежного поручения", bold)]
    def bank_paragraph(lines, centered=False):
        return Paragraph("<br/>".join(f"<b>{escape(str(text or ''))}</b>" if strong else escape(str(text or ""))
                                    for text, strong in lines), center if centered else normal)
    bank = Table([
        [bank_paragraph([("Бенефициар:", True), (issuer.get("name"), True),
                         (f"ИИН: {issuer.get('tax_id') or ''}", False)]),
         bank_paragraph([("ИИК", True), ("", False), (issuer.get("iban"), True)], True), "",
         bank_paragraph([("Кбе", True), ("", False), (issuer.get("kbe"), True)], True)],
        [bank_paragraph([("Банк бенефициара:", False), (issuer.get("bank"), False)]),
         bank_paragraph([("БИК", True), (issuer.get("bic"), True)], True),
         bank_paragraph([("Код назначения платежа", True), (issuer.get("payment_code"), True)], True), ""],
    ], colWidths=[width*.56, width*.19, width*.06, width*.19])
    bank.setStyle(TableStyle([("GRID", (0,0), (-1,-1), .5, colors.black),
                               ("VALIGN", (0,0), (-1,-1), "TOP"), ("SPAN", (1,0), (2,0)), ("SPAN", (2,1), (3,1)),
                               ("LEFTPADDING", (0,0), (-1,-1), 2), ("RIGHTPADDING", (0,0), (-1,-1), 2),
                               ("TOPPADDING", (0,0), (-1,-1), 1), ("BOTTOMPADDING", (0,0), (-1,-1), 1)]))
    parties = Table([[p(label), p(form[key], bold)] for label, key in
                     (("Поставщик:", "supplier"), ("Покупатель:", "buyer"), ("Договор:", "contract"))],
                    colWidths=[23*mm, width-23*mm], minRowHeights=[0, 7*mm, 7*mm])
    parties.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"),
                                ("LEFTPADDING", (0,0), (-1,-1), 2), ("RIGHTPADDING", (0,0), (-1,-1), 2),
                                ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
    story += [bank, Spacer(1, 8*mm), p(f"Счет на оплату № {document.get('number') or ''} от {form['date']}", heading),
              Spacer(1, 4*mm), HRFlowable(width="100%", thickness=1.2, color=colors.black), Spacer(1, 1*mm), parties,
              Spacer(1, 1*mm)]
    rows = [[p(label, center) for label in ("№", "Наименование", "Кол-во", "Ед.", "Цена", "Сумма")]]
    for index, row in enumerate(document["items"], 1):
        printed = form["items"][index-1]
        rows.append([p(index, small), p(row.get("name"), small), p(printed["quantity"], right),
                     p(row.get("unit")), p(printed["price"], right), p(printed["amount"], right)])
    table = Table(rows, colWidths=[11*mm, 81*mm, 17*mm, 13*mm, 26*mm, 30*mm], repeatRows=1)
    for index, row in enumerate(document['items'], 1):
        if not row.get('name'):
            table.setStyle(TableStyle([('BACKGROUND', (1,index), (1,index), colors.HexColor('#ffd6d6'))]))
        elif row.get('match_status') == 'suggested':
            table.setStyle(TableStyle([('BACKGROUND', (1,index), (1,index), colors.HexColor(suggestion_fill(row.get('suggestion_score'))))]))
    table.setStyle(TableStyle([("GRID", (0,0), (-1,-1), .5, colors.black), ("VALIGN", (0,0), (-1,-1), "TOP"),
                               ("BOX", (0,0), (-1,-1), 1, colors.black),
                               ("LEFTPADDING", (0,0), (-1,-1), 2), ("RIGHTPADDING", (0,0), (-1,-1), 2),
                               ("TOPPADDING", (0,0), (-1,-1), 1), ("BOTTOMPADDING", (0,0), (-1,-1), 1)]))
    totals = Table([[p("Итого:", right_bold), p(form["total"], right_bold)],
                    [p(form.get('vat_label','В том числе НДС:'), right_bold), p(form["vat"], right_bold)]], colWidths=[width-30*mm, 30*mm])
    totals.setStyle(TableStyle([("RIGHTPADDING", (0,0), (-1,-1), 2), ("TOPPADDING", (0,0), (-1,-1), 1),
                                ("BOTTOMPADDING", (0,0), (-1,-1), 1)]))
    signature = Table([[p("Исполнитель", bold), "", p(f"/{issuer['executor']}/" if issuer.get("executor") else "", small)]],
                      colWidths=[27*mm, 80*mm, width-107*mm])
    signature.setStyle(TableStyle([("LINEBELOW", (1,0), (1,0), .5, colors.black), ("VALIGN", (0,0), (-1,-1), "BOTTOM"),
                                   ("LEFTPADDING", (0,0), (-1,-1), 2)]))
    ending = [totals, Spacer(1, 5*mm),
              p(f"Всего наименований {len(document['items'])}, на сумму {form['total']} теңге"),
              p(f"Всего к оплате: {form['amount_words']}", bold), Spacer(1, 2*mm),
              HRFlowable(width="100%", thickness=1.2, color=colors.black), Spacer(1, 2*mm), signature,
              Spacer(1, 17*mm), p(form["tax_notice"], tax)]
    story += [table, Spacer(1, 2*mm), KeepTogether(ending)]
    def footer(canvas, doc):
        canvas.setFont("Invoice", 8)
        canvas.drawRightString(195*mm, 9*mm, str(doc.page))
    pdf.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
