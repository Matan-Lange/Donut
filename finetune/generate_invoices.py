import os
import random
import json
import argparse
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

# Initialize Faker with Hebrew locale
fake = Faker('he_IL')

import re


def reverse_numbers_in_string(text):
    def reverse_match(match):
        return match.group(0)[::-1]

    return re.sub(r'\d+', reverse_match, text)


def reverse_hebrew_text(text):
    return reverse_numbers_in_string(text[::-1])


def load_fonts():
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        bold_font = ImageFont.truetype("arialbd.ttf", 20)
        italic_font = ImageFont.truetype("ariali.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        bold_font = font
        italic_font = font
    return font, bold_font, italic_font


class Invoice:
    def __init__(self, invoice_number, date, client_name, client_address, supplier_name, supplier_address,
                 payment_terms, vat_percentage, additional_notes, items):
        self.invoice_number = invoice_number
        self.date = date
        self.client_name = client_name
        self.client_address = client_address
        self.supplier_name = supplier_name
        self.supplier_address = supplier_address
        self.payment_terms = payment_terms
        self.vat_percentage = vat_percentage
        self.additional_notes = additional_notes
        self.items = items
        self.font, self.bold_font, self.italic_font = load_fonts()
        self.total_with_vat = 0

    def draw_header(self, draw, width, current_height, padding):
        raise NotImplementedError("Subclasses should implement this method")

    def draw_title(self, draw, width, current_height, padding):
        raise NotImplementedError("Subclasses should implement this method")

    def draw_footer(self, draw, width, current_height, padding):
        raise NotImplementedError("Subclasses should implement this method")

    def draw_invoice_details(self, draw, width, current_height, padding):
        invoice_number_text = reverse_hebrew_text(f"מספר חשבונית: {self.invoice_number}")
        date_text = reverse_hebrew_text(f"תאריך: {self.date}")
        draw.text((width - padding, current_height), invoice_number_text, fill="black", font=self.font, anchor="ra")
        current_height += self.font.getbbox("hg")[3]
        draw.text((width - padding, current_height), date_text, fill="black", font=self.font, anchor="ra")
        current_height += self.font.getbbox("hg")[3] + padding
        return current_height

    def draw_client_supplier_info(self, draw, width, current_height, padding):
        client_info = f"לקוח: {self.client_name}\nכתובת: {self.client_address}"
        supplier_info = f"ספק: {self.supplier_name}\nכתובת ספק: {self.supplier_address}"
        for line in client_info.split('\n'):
            reversed_line = reverse_hebrew_text(line)
            draw.text((width - padding, current_height), reversed_line, fill="black", font=self.font, anchor="ra")
            current_height += self.font.getbbox("hg")[3] + 10

        current_height += self.font.getbbox("hg")[3] + 10  # Add spacing between client and supplier info

        for line in supplier_info.split('\n'):
            reversed_line = reverse_hebrew_text(line)
            draw.text((width - padding, current_height), reversed_line, fill="black", font=self.font, anchor="ra")
            current_height += self.font.getbbox("hg")[3] + 10
        return current_height

    def draw_payment_terms(self, draw, width, current_height, padding):
        payment_terms_text = reverse_hebrew_text(f"תנאי תשלום: {self.payment_terms}")
        draw.text((width - padding, current_height), payment_terms_text, fill="black", font=self.font, anchor="ra")
        current_height += self.font.getbbox("hg")[3] + padding + 10
        return current_height

    def draw_table_headers(self, draw, width, current_height, padding):
        headers = ["פריט", "כמות", "מחיר יחידה", "סה\"כ"]
        header_positions = [width - padding - 150 * i for i in range(len(headers))]
        draw.rectangle([(padding, current_height), (width - padding, current_height + self.font.getbbox("hg")[3] + 10)],
                       outline="black")
        for i, header in enumerate(headers):
            reversed_header = reverse_hebrew_text(header)
            draw.text((header_positions[i], current_height + 5), reversed_header, fill="black", font=self.bold_font,
                      anchor="ra")
        current_height += self.font.getbbox("hg")[3] + 20
        return current_height, header_positions

    def draw_items(self, draw, width, current_height, padding, header_positions):
        total_amount = 0
        items_data = []
        for item in self.items:
            item_name = reverse_hebrew_text(item[0])
            total_amount += item[3]
            fields = [item_name, item[1], f"{item[2]:.2f}", f"{item[3]:.2f}"]
            item_data = {"name": item[0], "quantity": item[1], "price": item[2], "total": item[3]}
            items_data.append(item_data)
            draw.rectangle(
                [(padding, current_height), (width - padding, current_height + self.font.getbbox("hg")[3] + 10)],
                outline="black")
            for i, field in enumerate(fields):
                draw.text((header_positions[i], current_height + 5), str(field), fill="black", font=self.font,
                          anchor="ra")
            current_height += self.font.getbbox("hg")[3] + 20

        current_height += padding // 2
        draw.line([(padding, current_height), (width - padding, current_height)], fill="black")
        current_height += padding // 2
        return current_height, total_amount, items_data

    def draw_totals(self, draw, width, current_height, padding, total_amount):
        vat_amount = round(total_amount * self.vat_percentage, 2)
        self.total_with_vat = total_amount + vat_amount
        vat_text = reverse_hebrew_text(f"מע\"מ ({self.vat_percentage * 100:.0f}%): {vat_amount:.2f} ש\"ח")
        total_text = reverse_hebrew_text(f"סה\"כ לתשלום כולל מע\"מ: {self.total_with_vat:.2f} ש\"ח")
        draw.text((width - padding, current_height), vat_text, fill="black", font=self.bold_font, anchor="ra")
        current_height += self.font.getbbox("hg")[3] + 10
        draw.text((width - padding, current_height), total_text, fill="black", font=self.bold_font, anchor="ra")
        current_height += self.font.getbbox("hg")[3] + padding + 10
        return current_height

    def draw_additional_notes(self, draw, width, current_height, padding):
        draw.line([(padding, current_height), (width - padding, current_height)], fill="black")
        current_height += padding // 2
        notes_title = reverse_hebrew_text("הערות נוספות")
        draw.text((width - padding, current_height), notes_title, fill="black", font=self.bold_font, anchor="ra")
        current_height += self.font.getbbox("hg")[3] + 10
        additional_notes_reversed = reverse_hebrew_text(self.additional_notes)
        draw.text((width - padding, current_height), additional_notes_reversed, fill="black", font=self.italic_font,
                  anchor="ra")
        current_height += self.font.getbbox("hg")[3] + padding
        return current_height

    def save_image(self, directory, file_name):
        # Define the image size and background color
        width, height = 600, 1400
        background_color = "white"
        image = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(image)

        padding = 10
        current_height = self.draw_header(draw, width, 0, padding)
        current_height = self.draw_title(draw, width, current_height, padding)
        current_height = self.draw_invoice_details(draw, width, current_height, padding)
        current_height = self.draw_client_supplier_info(draw, width, current_height, padding)
        current_height = self.draw_payment_terms(draw, width, current_height, padding)
        current_height, header_positions = self.draw_table_headers(draw, width, current_height, padding)
        current_height, total_amount, items_data = self.draw_items(draw, width, current_height, padding,
                                                                   header_positions)
        current_height = self.draw_totals(draw, width, current_height, padding, total_amount)
        current_height = self.draw_additional_notes(draw, width, current_height, padding)
        self.draw_footer(draw, width, current_height, padding)

        # Save the image
        file_path = os.path.join(directory, file_name)
        image.save(file_path)
        return file_path


class Layout1(Invoice):
    def draw_header(self, draw, width, current_height, padding):
        header_text = reverse_hebrew_text("חברת חשבוניות לדוגמה בע\"מ")
        header_bbox = draw.textbbox((0, 0), header_text, font=self.bold_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_height = header_bbox[3] - header_bbox[1]
        draw.rectangle([(0, 0), (width, header_height + padding * 2)], fill="lightblue")
        draw.text(((width - header_width) / 2, padding), header_text, fill="darkblue", font=self.bold_font)
        current_height += header_height + padding * 2
        return current_height

    def draw_title(self, draw, width, current_height, padding):
        title = reverse_hebrew_text("חשבונית מס")
        title_bbox = draw.textbbox((0, 0), title, font=self.bold_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_width) / 2, current_height), title, fill="darkblue", font=self.bold_font)
        current_height += title_height + padding
        return current_height

    def draw_footer(self, draw, width, current_height, padding):
        footer_text = reverse_hebrew_text("תודה על קנייתכם!")
        footer_bbox = draw.textbbox((0, 0), footer_text, font=self.italic_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        draw.rectangle([(0, current_height), (width, current_height + padding * 2)], fill="lightblue")
        draw.text(((width - footer_width) / 2, current_height + padding), footer_text, fill="darkblue",
                  font=self.italic_font)
        return current_height + padding * 2


class Layout2(Invoice):
    def draw_header(self, draw, width, current_height, padding):
        header_text = reverse_hebrew_text("חברת חשבוניות לדוגמה בע\"מ")
        header_bbox = draw.textbbox((0, 0), header_text, font=self.bold_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_height = header_bbox[3] - header_bbox[1]
        draw.rectangle([(0, 0), (width, header_height + padding * 2)], fill="lightgreen")
        draw.text(((width - header_width) / 2, padding), header_text, fill="darkgreen", font=self.bold_font)
        current_height += header_height + padding * 2
        return current_height

    def draw_title(self, draw, width, current_height, padding):
        title = reverse_hebrew_text("חשבונית מס")
        title_bbox = draw.textbbox((0, 0), title, font=self.bold_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_width) / 2, current_height), title, fill="darkgreen", font=self.bold_font)
        current_height += title_height + padding
        return current_height

    def draw_footer(self, draw, width, current_height, padding):
        footer_text = reverse_hebrew_text("תודה על קנייתכם!")
        footer_bbox = draw.textbbox((0, 0), footer_text, font=self.italic_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        draw.rectangle([(0, current_height), (width, current_height + padding * 2)], fill="lightgreen")
        draw.text(((width - footer_width) / 2, current_height + padding), footer_text, fill="darkgreen",
                  font=self.italic_font)
        return current_height + padding * 2


class Layout3(Invoice):
    def draw_header(self, draw, width, current_height, padding):
        header_text = reverse_hebrew_text("חברת חשבוניות לדוגמה בע\"מ")
        header_bbox = draw.textbbox((0, 0), header_text, font=self.bold_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_height = header_bbox[3] - header_bbox[1]
        draw.rectangle([(0, 0), (width, header_height + padding * 2)], fill="lightcoral")
        draw.text(((width - header_width) / 2, padding), header_text, fill="darkred", font=self.bold_font)
        current_height += header_height + padding * 2
        return current_height

    def draw_title(self, draw, width, current_height, padding):
        title = reverse_hebrew_text("חשבונית מס")
        title_bbox = draw.textbbox((0, 0), title, font=self.bold_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_width) / 2, current_height), title, fill="darkred", font=self.bold_font)
        current_height += title_height + padding
        return current_height

    def draw_footer(self, draw, width, current_height, padding):
        footer_text = reverse_hebrew_text("תודה על קנייתכם!")
        footer_bbox = draw.textbbox((0, 0), footer_text, font=self.italic_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        draw.rectangle([(0, current_height), (width, current_height + padding * 2)], fill="lightcoral")
        draw.text(((width - footer_width) / 2, current_height + padding), footer_text, fill="darkred",
                  font=self.italic_font)
        return current_height + padding * 2


class Layout4(Invoice):
    def draw_header(self, draw, width, current_height, padding):
        header_text = reverse_hebrew_text("חברת חשבוניות לדוגמה בע\"מ")
        header_bbox = draw.textbbox((0, 0), header_text, font=self.bold_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_height = header_bbox[3] - header_bbox[1]
        draw.rectangle([(0, 0), (width, header_height + padding * 2)], fill="lightyellow")
        draw.text(((width - header_width) / 2, padding), header_text, fill="darkgoldenrod", font=self.bold_font)
        current_height += header_height + padding * 2
        return current_height

    def draw_title(self, draw, width, current_height, padding):
        title = reverse_hebrew_text("חשבונית מס")
        title_bbox = draw.textbbox((0, 0), title, font=self.bold_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_width) / 2, current_height), title, fill="darkgoldenrod", font=self.bold_font)
        current_height += title_height + padding
        return current_height

    def draw_footer(self, draw, width, current_height, padding):
        footer_text = reverse_hebrew_text("תודה על קנייתכם!")
        footer_bbox = draw.textbbox((0, 0), footer_text, font=self.italic_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        draw.rectangle([(0, current_height), (width, current_height + padding * 2)], fill="lightyellow")
        draw.text(((width - footer_width) / 2, current_height + padding), footer_text, fill="darkgoldenrod",
                  font=self.italic_font)
        return current_height + padding * 2


class Layout5(Invoice):
    def draw_header(self, draw, width, current_height, padding):
        header_text = reverse_hebrew_text("חברת חשבוניות לדוגמה בע\"מ")
        header_bbox = draw.textbbox((0, 0), header_text, font=self.bold_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_height = header_bbox[3] - header_bbox[1]
        draw.rectangle([(0, 0), (width, header_height + padding * 2)], fill="lightpink")
        draw.text(((width - header_width) / 2, padding), header_text, fill="deeppink", font=self.bold_font)
        current_height += header_height + padding * 2
        return current_height

    def draw_title(self, draw, width, current_height, padding):
        title = reverse_hebrew_text("חשבונית מס")
        title_bbox = draw.textbbox((0, 0), title, font=self.bold_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_width) / 2, current_height), title, fill="deeppink", font=self.bold_font)
        current_height += title_height + padding
        return current_height

    def draw_footer(self, draw, width, current_height, padding):
        footer_text = reverse_hebrew_text("תודה על קנייתכם!")
        footer_bbox = draw.textbbox((0, 0), footer_text, font=self.italic_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        draw.rectangle([(0, current_height), (width, current_height + padding * 2)], fill="lightpink")
        draw.text(((width - footer_width) / 2, current_height + padding), footer_text, fill="deeppink",
                  font=self.italic_font)
        return current_height + padding * 2


class Layout6(Invoice):
    def draw_header(self, draw, width, current_height, padding):
        header_text = reverse_hebrew_text("חברת חשבוניות לדוגמה בע\"מ")
        header_bbox = draw.textbbox((0, 0), header_text, font=self.bold_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_height = header_bbox[3] - header_bbox[1]
        draw.rectangle([(0, 0), (width, header_height + padding * 2)], fill="lightcyan")
        draw.text(((width - header_width) / 2, padding), header_text, fill="darkcyan", font=self.bold_font)
        current_height += header_height + padding * 2
        return current_height

    def draw_title(self, draw, width, current_height, padding):
        title = reverse_hebrew_text("חשבונית מס")
        title_bbox = draw.textbbox((0, 0), title, font=self.bold_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_width) / 2, current_height), title, fill="darkcyan", font=self.bold_font)
        current_height += title_height + padding
        return current_height

    def draw_footer(self, draw, width, current_height, padding):
        footer_text = reverse_hebrew_text("תודה על קנייתכם!")
        footer_bbox = draw.textbbox((0, 0), footer_text, font=self.italic_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        draw.rectangle([(0, current_height), (width, current_height + padding * 2)], fill="lightcyan")
        draw.text(((width - footer_width) / 2, current_height + padding), footer_text, fill="darkcyan",
                  font=self.italic_font)
        return current_height + padding * 2


class Layout7(Invoice):
    def draw_header(self, draw, width, current_height, padding):
        header_text = reverse_hebrew_text("חברת חשבוניות לדוגמה בע\"מ")
        header_bbox = draw.textbbox((0, 0), header_text, font=self.bold_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_height = header_bbox[3] - header_bbox[1]
        draw.rectangle([(0, 0), (width, header_height + padding * 2)], fill="lightsteelblue")
        draw.text(((width - header_width) / 2, padding), header_text, fill="darkslateblue", font=self.bold_font)
        current_height += header_height + padding * 2
        return current_height

    def draw_title(self, draw, width, current_height, padding):
        title = reverse_hebrew_text("חשבונית מס")
        title_bbox = draw.textbbox((0, 0), title, font=self.bold_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_width) / 2, current_height), title, fill="darkslateblue", font=self.bold_font)
        current_height += title_height + padding
        return current_height

    def draw_footer(self, draw, width, current_height, padding):
        footer_text = reverse_hebrew_text("תודה על קנייתכם!")
        footer_bbox = draw.textbbox((0, 0), footer_text, font=self.italic_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        draw.rectangle([(0, current_height), (width, current_height + padding * 2)], fill="lightsteelblue")
        draw.text(((width - footer_width) / 2, current_height + padding), footer_text, fill="darkslateblue",
                  font=self.italic_font)
        return current_height + padding * 2


class Layout8(Invoice):
    def draw_header(self, draw, width, current_height, padding):
        header_text = reverse_hebrew_text("חברת חשבוניות לדוגמה בע\"מ")
        header_bbox = draw.textbbox((0, 0), header_text, font=self.bold_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_height = header_bbox[3] - header_bbox[1]
        draw.rectangle([(0, 0), (width, header_height + padding * 2)], fill="lightseagreen")
        draw.text(((width - header_width) / 2, padding), header_text, fill="darkseagreen", font=self.bold_font)
        current_height += header_height + padding * 2
        return current_height

    def draw_title(self, draw, width, current_height, padding):
        title = reverse_hebrew_text("חשבונית מס")
        title_bbox = draw.textbbox((0, 0), title, font=self.bold_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_width) / 2, current_height), title, fill="darkseagreen", font=self.bold_font)
        current_height += title_height + padding
        return current_height

    def draw_footer(self, draw, width, current_height, padding):
        footer_text = reverse_hebrew_text("תודה על קנייתכם!")
        footer_bbox = draw.textbbox((0, 0), footer_text, font=self.italic_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        draw.rectangle([(0, current_height), (width, current_height + padding * 2)], fill="lightseagreen")
        draw.text(((width - footer_width) / 2, current_height + padding), footer_text, fill="darkseagreen",
                  font=self.italic_font)
        return current_height + padding * 2


class Layout9(Invoice):
    def draw_header(self, draw, width, current_height, padding):
        header_text = reverse_hebrew_text("חברת חשבוניות לדוגמה בע\"מ")
        header_bbox = draw.textbbox((0, 0), header_text, font=self.bold_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_height = header_bbox[3] - header_bbox[1]
        draw.rectangle([(0, 0), (width, header_height + padding * 2)], fill="lightgoldenrodyellow")
        draw.text(((width - header_width) / 2, padding), header_text, fill="darkgoldenrod", font=self.bold_font)
        current_height += header_height + padding * 2
        return current_height

    def draw_title(self, draw, width, current_height, padding):
        title = reverse_hebrew_text("חשבונית מס")
        title_bbox = draw.textbbox((0, 0), title, font=self.bold_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_width) / 2, current_height), title, fill="darkgoldenrod", font=self.bold_font)
        current_height += title_height + padding
        return current_height

    def draw_footer(self, draw, width, current_height, padding):
        footer_text = reverse_hebrew_text("תודה על קנייתכם!")
        footer_bbox = draw.textbbox((0, 0), footer_text, font=self.italic_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        draw.rectangle([(0, current_height), (width, current_height + padding * 2)], fill="lightgoldenrodyellow")
        draw.text(((width - footer_width) / 2, current_height + padding), footer_text, fill="darkgoldenrod",
                  font=self.italic_font)
        return current_height + padding * 2


class Layout10(Invoice):
    def draw_header(self, draw, width, current_height, padding):
        header_text = reverse_hebrew_text("חברת חשבוניות לדוגמה בע\"מ")
        header_bbox = draw.textbbox((0, 0), header_text, font=self.bold_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_height = header_bbox[3] - header_bbox[1]
        draw.rectangle([(0, 0), (width, header_height + padding * 2)], fill="lightgrey")
        draw.text(((width - header_width) / 2, padding), header_text, fill="black", font=self.bold_font)
        current_height += header_height + padding * 2
        return current_height

    def draw_title(self, draw, width, current_height, padding):
        title = reverse_hebrew_text("חשבונית מס")
        title_bbox = draw.textbbox((0, 0), title, font=self.bold_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_width) / 2, current_height), title, fill="black", font=self.bold_font)
        current_height += title_height + padding
        return current_height

    def draw_footer(self, draw, width, current_height, padding):
        footer_text = reverse_hebrew_text("תודה על קנייתכם!")
        footer_bbox = draw.textbbox((0, 0), footer_text, font=self.italic_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        draw.rectangle([(0, current_height), (width, current_height + padding * 2)], fill="lightgrey")
        draw.text(((width - footer_width) / 2, current_height + padding), footer_text, fill="black",
                  font=self.italic_font)
        return current_height + padding * 2


class InvoiceGenerator:
    def __init__(self, total_invoices, output_directory="invoices"):
        self.total_invoices = total_invoices
        self.output_directory = output_directory
        self.used_invoice_numbers = set()
        os.makedirs(self.output_directory, exist_ok=True)
        self.create_subdirectories()

    def create_subdirectories(self):
        for sub_dir in ['train', 'validation', 'test']:
            os.makedirs(os.path.join(self.output_directory, sub_dir), exist_ok=True)

    def generate_unique_invoice_number(self):
        while True:
            invoice_number = random.randint(1000, 9999)
            if invoice_number not in self.used_invoice_numbers:
                self.used_invoice_numbers.add(invoice_number)
                return invoice_number

    def generate_fake_data(self):
        client_name = fake.name()
        client_address = fake.address()
        supplier_name = fake.name()
        supplier_address = fake.address()
        payment_terms = random.choice(["30 ימים", "60 ימים", "90 ימים"])
        vat_percentage = random.choice([0.17, 0.18])
        additional_notes = fake.sentence(nb_words=10)
        items = []
        for _ in range(random.randint(3, 7)):
            item_name = fake.word()
            item_quantity = random.randint(1, 10)
            item_price = round(random.uniform(20, 200), 2)
            item_total = round(item_quantity * item_price, 2)
            items.append((item_name, item_quantity, item_price, item_total))
        return client_name, client_address, supplier_name, supplier_address, payment_terms, vat_percentage, additional_notes, items

    def generate_invoices(self):
        train_split = int(self.total_invoices * 0.6)
        validation_split = int(self.total_invoices * 0.2)
        test_split = self.total_invoices - train_split - validation_split

        splits = {
            "train": train_split,
            "validation": validation_split,
            "test": test_split
        }

        layouts = [Layout1, Layout2, Layout3, Layout4, Layout5, Layout6, Layout7, Layout8, Layout9, Layout10]

        for split_name, split_count in splits.items():
            metadata_file = os.path.join(self.output_directory, split_name, "metadata.jsonl")
            with open(metadata_file, "w", encoding='utf8') as meta_file:
                for _ in range(split_count):
                    client_name, client_address, supplier_name, supplier_address, payment_terms, vat_percentage, additional_notes, items = self.generate_fake_data()
                    invoice_number = self.generate_unique_invoice_number()
                    date = datetime.now().strftime('%d/%m/%Y')
                    layout_class = random.choice(layouts)
                    invoice = layout_class(invoice_number, date, client_name, client_address, supplier_name,
                                           supplier_address, payment_terms, vat_percentage, additional_notes, items)
                    file_name = f"fake_hebrew_invoice_{invoice_number}.png"
                    image_path = invoice.save_image(os.path.join(self.output_directory, split_name), file_name)
                    parsing_data = {
                        "image_name": image_path,
                        "receipt_number": invoice_number,
                        "date": date,
                        "client": {
                            "name": client_name,
                            "address": client_address
                        },
                        "supplier": {
                            "name": supplier_name,
                            "address": supplier_address
                        },
                        "items": [{"name": item[0], "quantity": item[1], "price": item[2], "total": item[3]} for item in
                                  items],
                        "total_sum": invoice.total_with_vat
                    }
                    meta_file.write(json.dumps(parsing_data, ensure_ascii=False) + "\n")
            print(f"Metadata saved in {metadata_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hebrew invoices data")
    parser.add_argument("--total_invoices", type=int, required=True, help="Total number of invoices to generate")
    parser.add_argument("--output_directory", type=str, default="invoices",
                        help="Output directory to save invoices and metadata")

    args = parser.parse_args()
    generator = InvoiceGenerator(args.total_invoices, args.output_directory)
    generator.generate_invoices()
