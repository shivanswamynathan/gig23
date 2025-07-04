# Generated by Django 5.2.3 on 2025-06-30 04:54

import django.core.validators
import django.db.models.deletion
from decimal import Decimal
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('document_processing', '0002_itemwisegrn'),
    ]

    operations = [
        migrations.CreateModel(
            name='InvoiceData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('attachment_number', models.CharField(choices=[('1', 'Attachment 1'), ('2', 'Attachment 2'), ('3', 'Attachment 3'), ('4', 'Attachment 4'), ('5', 'Attachment 5')], max_length=2, verbose_name='Attachment Number')),
                ('attachment_url', models.URLField(max_length=1000, verbose_name='Original Attachment URL')),
                ('file_type', models.CharField(choices=[('pdf_text', 'PDF - Text Based'), ('pdf_image', 'PDF - Image Based'), ('image', 'Image File'), ('unknown', 'Unknown/Failed')], max_length=20, verbose_name='File Processing Type')),
                ('original_file_extension', models.CharField(blank=True, help_text='Original file extension (.pdf, .jpg, .png, etc.)', max_length=10, null=True, verbose_name='Original File Extension')),
                ('vendor_name', models.CharField(blank=True, max_length=255, null=True)),
                ('vendor_pan', models.CharField(blank=True, max_length=10, null=True)),
                ('vendor_gst', models.CharField(blank=True, max_length=15, null=True)),
                ('invoice_date', models.DateField(blank=True, null=True)),
                ('invoice_number', models.CharField(blank=True, max_length=100, null=True)),
                ('po_number', models.CharField(blank=True, db_index=True, max_length=200, null=True)),
                ('invoice_value_without_gst', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))])),
                ('cgst_rate', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('cgst_amount', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))])),
                ('sgst_rate', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('sgst_amount', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))])),
                ('igst_rate', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('igst_amount', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))])),
                ('total_gst_amount', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))])),
                ('invoice_total_post_gst', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))])),
                ('items_data', models.JSONField(blank=True, null=True)),
                ('processing_status', models.CharField(choices=[('pending', 'Pending'), ('processing', 'Processing'), ('completed', 'Completed'), ('failed', 'Failed')], default='pending', max_length=20)),
                ('error_message', models.TextField(blank=True, null=True)),
                ('extracted_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Invoice Data',
                'verbose_name_plural': 'Invoice Data Records',
                'db_table': 'invoice_data',
                'ordering': ['-created_at'],
                'indexes': [models.Index(fields=['po_number'], name='invoice_dat_po_numb_1e4a02_idx'), models.Index(fields=['invoice_number'], name='invoice_dat_invoice_630bef_idx'), models.Index(fields=['vendor_gst'], name='invoice_dat_vendor__0e8ced_idx'), models.Index(fields=['processing_status'], name='invoice_dat_process_402248_idx'), models.Index(fields=['file_type'], name='invoice_dat_file_ty_217381_idx'), models.Index(fields=['attachment_url'], name='invoice_dat_attachm_a17ef4_idx')],
            },
        ),
        migrations.CreateModel(
            name='InvoiceItemData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('item_description', models.CharField(max_length=1000, verbose_name='Item Description')),
                ('hsn_code', models.CharField(blank=True, max_length=20, null=True, verbose_name='HSN Code')),
                ('quantity', models.DecimalField(blank=True, decimal_places=4, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.0000'))], verbose_name='Quantity')),
                ('unit_of_measurement', models.CharField(blank=True, max_length=20, null=True, verbose_name='Unit of Measurement')),
                ('unit_price', models.DecimalField(blank=True, decimal_places=4, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.0000'))], verbose_name='Unit Price')),
                ('invoice_value_item_wise', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))], verbose_name='Item-wise Invoice Value')),
                ('cgst_rate', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True, verbose_name='CGST Rate')),
                ('cgst_amount', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))], verbose_name='CGST Amount')),
                ('sgst_rate', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True, verbose_name='SGST Rate')),
                ('sgst_amount', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))], verbose_name='SGST Amount')),
                ('igst_rate', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True, verbose_name='IGST Rate')),
                ('igst_amount', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))], verbose_name='IGST Amount')),
                ('total_tax_amount', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))], verbose_name='Total Tax Amount')),
                ('item_total_amount', models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True, validators=[django.core.validators.MinValueValidator(Decimal('0.00'))], verbose_name='Item Total Amount')),
                ('po_number', models.CharField(blank=True, db_index=True, max_length=200, null=True, verbose_name='PO Number')),
                ('invoice_number', models.CharField(blank=True, db_index=True, max_length=100, null=True, verbose_name='Invoice Number')),
                ('vendor_name', models.CharField(blank=True, max_length=255, null=True, verbose_name='Vendor Name')),
                ('item_sequence', models.PositiveIntegerField(default=1, verbose_name='Item Sequence')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='Created At')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='Updated At')),
                ('invoice_data', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='invoice_items', to='document_processing.invoicedata', verbose_name='Invoice Data')),
            ],
            options={
                'verbose_name': 'Invoice Item Data',
                'verbose_name_plural': 'Invoice Items Data',
                'db_table': 'invoice_item_data',
                'ordering': ['invoice_data', 'item_sequence'],
                'indexes': [models.Index(fields=['po_number'], name='invoice_ite_po_numb_2991a8_idx'), models.Index(fields=['invoice_number'], name='invoice_ite_invoice_d345b8_idx'), models.Index(fields=['hsn_code'], name='invoice_ite_hsn_cod_408402_idx'), models.Index(fields=['vendor_name'], name='invoice_ite_vendor__047635_idx'), models.Index(fields=['invoice_data', 'item_sequence'], name='invoice_ite_invoice_29e912_idx')],
            },
        ),
    ]
