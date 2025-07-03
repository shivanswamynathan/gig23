# reconciliation/utils/simple_async_reconciliation.py

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional
from django.db import transaction
from django.db.models import Sum, Count
from asgiref.sync import sync_to_async
from langchain_google_genai import GoogleGenerativeAI
from django.conf import settings
import os

from document_processing.models import InvoiceData, InvoiceItemData, ItemWiseGrn,InvoiceGrnReconciliation, ReconciliationBatch


logger = logging.getLogger(__name__)


class ReconciliationProcessor:
    """
    Simple async reconciliation processor for 10,000+ records
    Uses delay system like invoice processors and concurrent processing
    """
    
    def __init__(self, delay_seconds: float = 1.0, max_concurrent: int = 10):
        self.delay_seconds = delay_seconds  # Delay between LLM calls
        self.max_concurrent = max_concurrent
        self.setup_llm()
        
        self.stats = {
            'total_processed': 0,
            'perfect_matches': 0,
            'partial_matches': 0,
            'llm_matches': 0,
            'no_matches': 0,
            'errors': 0
        }
    
    def setup_llm(self):
        """Setup LLM with proper configuration"""
        try:
            api_key = getattr(settings, 'GOOGLE_API_KEY', None) or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.warning("No GOOGLE_API_KEY found, LLM features disabled")
                self.llm = None
                return
            
            model_name = getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash')
            self.llm = GoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.1
            )
            logger.info(f"LLM initialized: {model_name}")
            
        except Exception as e:
            logger.error(f"LLM setup failed: {str(e)}")
            self.llm = None
    
    async def _invoke_llm_with_retry_and_delay(self, prompt: str) -> str:
        """LLM call with retry logic and delay (like invoice processors)"""
        max_retries = 3
        attempt = 0
        
        while attempt < max_retries:
            try:
                # Apply delay before each call
                if self.delay_seconds > 0:
                    await asyncio.sleep(self.delay_seconds)
                
                # Run LLM call in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.llm.invoke, prompt)
                return result
                
            except Exception as e:
                err_msg = str(e)
                if '429' in err_msg and 'retry_delay' in err_msg:
                    import re
                    match = re.search(r'retry_delay[":\s]*([0-9.]+)', err_msg)
                    if match:
                        retry_delay = float(match.group(1))
                        logger.warning(f"Rate limit hit, waiting {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        attempt += 1
                        continue
                raise
        
        raise Exception(f"Max retries reached for LLM call")
    
    async def process_batch_async(self, invoice_ids: List[int] = None, batch_size: int = 100) -> Dict[str, Any]:
        """
        Process invoices in batches asynchronously
        
        Args:
            invoice_ids: Optional list of invoice IDs
            batch_size: Number of invoices to process in each batch
        """
        try:
            logger.info(f"Starting async reconciliation with delay={self.delay_seconds}s, concurrent={self.max_concurrent}")
            
            # Get invoices
            if invoice_ids:
                invoices = await sync_to_async(list)(
                    InvoiceData.objects.filter(id__in=invoice_ids, processing_status='completed')
                )
            else:
                invoices = await sync_to_async(list)(
                    InvoiceData.objects.filter(processing_status='completed')
                )
            
            total_invoices = len(invoices)
            logger.info(f"Processing {total_invoices} invoices...")
            
            # Process in batches with semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            # Split into batches
            results = []
            for i in range(0, total_invoices, batch_size):
                batch = invoices[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} invoices")
                
                # Create tasks for this batch
                tasks = [
                    self._process_single_invoice_async(invoice, semaphore)
                    for invoice in batch
                ]
                
                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Invoice {batch[j].id} failed: {str(result)}")
                        self.stats['errors'] += 1
                    else:
                        results.append(result)
                        self.stats['total_processed'] += 1
                
                # Log progress
                logger.info(f"Completed batch {i//batch_size + 1}/{(total_invoices + batch_size - 1)//batch_size}")
                logger.info(f"Progress: {self.stats['total_processed']}/{total_invoices} ({self.stats['total_processed']/total_invoices*100:.1f}%)")
            
            logger.info("Async reconciliation completed!")
            logger.info(f"Stats: {self.stats}")
            
            return {
                'success': True,
                'total_processed': self.stats['total_processed'],
                'stats': self.stats,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }
    
    async def _process_single_invoice_async(self, invoice: InvoiceData, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process single invoice with concurrency control"""
        async with semaphore:
            try:
                logger.info(f"Processing invoice {invoice.id} - PO: {invoice.po_number}")
                
                # Step 1: Find matching GRN records (async)
                grn_items = await self._find_grn_matches_async(invoice)
                
                if not grn_items:
                    self.stats['no_matches'] += 1
                    return await self._create_no_match_record_async(invoice)
                
                # Step 2: Get invoice items (async)
                invoice_items = await sync_to_async(list)(
                    InvoiceItemData.objects.filter(invoice_data=invoice)
                )
                
                # Step 3: Basic amount comparison
                match_result = await self._basic_amount_comparison_async(invoice, grn_items)
                
                # Step 4: LLM field-by-field comparison (if LLM available)
                llm_analysis = None
                if self.llm and len(grn_items) <= 20:  # Limit LLM calls for performance
                    llm_analysis = await self._llm_field_comparison_async(invoice, invoice_items, grn_items)
                
                # Step 5: Create reconciliation record
                reconciliation = await self._create_reconciliation_record_async(
                    invoice, grn_items, match_result, llm_analysis
                )
                
                # Update stats
                if match_result['match_status'] == 'perfect_match':
                    self.stats['perfect_matches'] += 1
                elif match_result['match_status'] == 'partial_match':
                    self.stats['partial_matches'] += 1
                
                if llm_analysis:
                    self.stats['llm_matches'] += 1
                
                return {
                    'invoice_id': invoice.id,
                    'reconciliation_id': reconciliation.id,
                    'match_status': match_result['match_status'],
                    'variance_pct': match_result.get('variance_pct', 0),
                    'llm_discrepancies': len(llm_analysis.get('discrepancies', [])) if llm_analysis else 0
                }
                
            except Exception as e:
                logger.error(f"Error processing invoice {invoice.id}: {str(e)}")
                raise
    
    async def _find_grn_matches_async(self, invoice: InvoiceData) -> List[ItemWiseGrn]:
        """Find matching GRN records (async)"""
        # Strategy 1: Exact match
        if invoice.po_number and invoice.grn_number:
            grn_items = await sync_to_async(list)(
                ItemWiseGrn.objects.filter(
                    po_no=invoice.po_number,
                    grn_no=invoice.grn_number
                )
            )
            if grn_items:
                return grn_items
        
        # Strategy 2: PO only
        if invoice.po_number:
            grn_items = await sync_to_async(list)(
                ItemWiseGrn.objects.filter(po_no=invoice.po_number)
            )
            return grn_items
        
        return []
    
    async def _basic_amount_comparison_async(self, invoice: InvoiceData, grn_items: List[ItemWiseGrn]) -> Dict[str, Any]:
        """Basic amount comparison (async)"""
        # Aggregate GRN amounts
        total_grn_amount = sum(item.total or 0 for item in grn_items)
        invoice_total = invoice.invoice_total_post_gst or 0
        
        if invoice_total > 0:
            variance_pct = abs((invoice_total - total_grn_amount) / invoice_total * 100)
        else:
            variance_pct = 100
        
        # Determine match status
        if variance_pct <= 2:
            match_status = 'perfect_match'
        elif variance_pct <= 10:
            match_status = 'partial_match'
        else:
            match_status = 'amount_mismatch'
        
        return {
            'match_status': match_status,
            'variance_pct': float(variance_pct),
            'invoice_total': float(invoice_total),
            'grn_total': float(total_grn_amount),
            'grn_items_count': len(grn_items)
        }
    
    async def _llm_field_comparison_async(self, invoice: InvoiceData, invoice_items: List[InvoiceItemData], 
                                        grn_items: List[ItemWiseGrn]) -> Optional[Dict[str, Any]]:
        """Enhanced LLM field-by-field comparison similar to OpenAI example"""
        if not self.llm:
            return None
        
        try:
            # Prepare detailed invoice data (similar to OpenAI example)
            invoice_json = {
                "header_info": {
                    "po_number": invoice.po_number or "",
                    "grn_number": invoice.grn_number or "",
                    "invoice_number": invoice.invoice_number or "",
                    "invoice_date": str(invoice.invoice_date) if invoice.invoice_date else "",
                    "vendor_name": invoice.vendor_name or "",
                    "vendor_gst": invoice.vendor_gst or "",
                    "vendor_pan": invoice.vendor_pan or ""
                },
                "financial_totals": {
                    "invoice_value_without_gst": str(invoice.invoice_value_without_gst or 0),
                    "cgst_amount": str(invoice.cgst_amount or 0),
                    "sgst_amount": str(invoice.sgst_amount or 0),
                    "igst_amount": str(invoice.igst_amount or 0),
                    "total_gst_amount": str(invoice.total_gst_amount or 0),
                    "invoice_total_post_gst": str(invoice.invoice_total_post_gst or 0)
                },
                "line_items": []
            }
            
            # Add invoice line items
            for item in invoice_items[:10]:  # Limit to 10 items for token efficiency
                invoice_json["line_items"].append({
                    "item_sequence": item.item_sequence,
                    "item_description": item.item_description or "",
                    "hsn_code": item.hsn_code or "",
                    "quantity": str(item.quantity or 0),
                    "unit_of_measurement": item.unit_of_measurement or "",
                    "unit_price": str(item.unit_price or 0),
                    "invoice_value_item_wise": str(item.invoice_value_item_wise or 0),
                    "cgst_rate": str(item.cgst_rate or 0),
                    "cgst_amount": str(item.cgst_amount or 0),
                    "sgst_rate": str(item.sgst_rate or 0),
                    "sgst_amount": str(item.sgst_amount or 0),
                    "igst_rate": str(item.igst_rate or 0),
                    "igst_amount": str(item.igst_amount or 0),
                    "item_total_amount": str(item.item_total_amount or 0)
                })
            
            # Prepare detailed GRN data
            if not grn_items:
                grn_json = {"header_info": {}, "financial_totals": {}, "line_items": []}
            else:
                first_item = grn_items[0]
                grn_json = {
                    "header_info": {
                        "po_number": first_item.po_no or "",
                        "grn_number": first_item.grn_no or "",
                        "invoice_number": first_item.seller_invoice_no or "",
                        "invoice_date": str(first_item.supplier_invoice_date) if first_item.supplier_invoice_date else "",
                        "vendor_name": first_item.pickup_location or first_item.supplier or "",
                        "vendor_gst": first_item.pickup_gstin or "",
                        "grn_created_date": str(first_item.grn_created_at) if first_item.grn_created_at else ""
                    },
                    "financial_totals": {
                        "total_subtotal": str(sum(item.subtotal or 0 for item in grn_items)),
                        "total_cgst_amount": str(sum(item.cgst_tax_amount or 0 for item in grn_items)),
                        "total_sgst_amount": str(sum(item.sgst_tax_amount or 0 for item in grn_items)),
                        "total_igst_amount": str(sum(item.igst_tax_amount or 0 for item in grn_items)),
                        "total_tax_amount": str(sum(item.tax_amount or 0 for item in grn_items)),
                        "grand_total": str(sum(item.total or 0 for item in grn_items))
                    },
                    "line_items": []
                }
                
                # Add GRN line items (limit for token efficiency)
                for item in grn_items[:10]:
                    grn_json["line_items"].append({
                        "s_no": item.s_no,
                        "item_name": item.item_name or "",
                        "sku_code": item.sku_code or "",
                        "hsn_code": item.hsn_no or "",
                        "quantity": str(item.received_qty or 0),
                        "unit": item.unit or "",
                        "price": str(item.price or 0),
                        "subtotal": str(item.subtotal or 0),
                        "cgst_tax": str(item.cgst_tax or 0),
                        "cgst_tax_amount": str(item.cgst_tax_amount or 0),
                        "sgst_tax": str(item.sgst_tax or 0),
                        "sgst_tax_amount": str(item.sgst_tax_amount or 0),
                        "igst_tax": str(item.igst_tax or 0),
                        "igst_tax_amount": str(item.igst_tax_amount or 0),
                        "tax_amount": str(item.tax_amount or 0),
                        "total": str(item.total or 0)
                    })
            
            # Enhanced prompt (similar to OpenAI example)
            prompt = f"""
You are a supply-chain data auditor specializing in invoice-GRN reconciliation. Below are two JSON documents:

1. `GRN JSON`: trusted record of what was received from the supplier
2. `INVOICE JSON`: extracted data from supplier invoice

Please do the following:
- Compare both JSONs **field by field** across all sections (header_info, financial_totals, line_items)
- Identify **any missing fields, mismatched values, or extra entries**
- For numerical values, consider small rounding differences (<0.01) as acceptable
- For text fields, consider case-insensitive matching and common abbreviations
- Output a **CSV-style markdown table** with the following columns: 
  Field, GRN_Value, Invoice_Value, Discrepancy_Type, Suggestion
- Then write a brief **natural language summary** explaining the main issues and suggested actions

**COMPARISON RULES:**
1. Header Info: Match PO numbers, GRN numbers, vendor details, dates
2. Financial Totals: Compare aggregated amounts with tolerance for rounding
3. Line Items: Match quantities, rates, descriptions, tax amounts
4. Discrepancy Types: MISSING, MISMATCH, EXTRA, AMOUNT_VARIANCE, DATE_ISSUE, VENDOR_ISSUE

--- GRN JSON ---
{json.dumps(grn_json, indent=2)}

--- INVOICE JSON ---
{json.dumps(invoice_json, indent=2)}

Please provide the analysis in the format specified above.
"""
            
            # Call LLM with delay
            response = await self._invoke_llm_with_retry_and_delay(prompt)
            
            # Parse the response (similar to OpenAI example parsing)
            analysis_result = self._parse_detailed_llm_response(response)
            
            return analysis_result
        
        except Exception as e:
            logger.error(f"Enhanced LLM comparison failed: {str(e)}")
            return None
    
    def _parse_detailed_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse the detailed LLM response to extract discrepancies and summary (like OpenAI example)"""
        try:
            # Split response into table and summary parts
            lines = llm_response.strip().split('\n')
            
            # Extract markdown table
            table_lines = []
            summary_lines = []
            in_table = False
            table_ended = False
            
            for line in lines:
                if '|' in line and not table_ended:
                    # Skip header separator lines
                    if not (set(line.strip()) <= {"|", "-", " "}):
                        table_lines.append(line)
                        in_table = True
                elif in_table and '|' not in line and line.strip():
                    table_ended = True
                    summary_lines.append(line)
                elif table_ended:
                    summary_lines.append(line)
            
            # Parse discrepancies from table (like CSV parsing in OpenAI example)
            discrepancies = []
            for line in table_lines[1:]:  # Skip header row
                cols = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(cols) >= 5:
                    discrepancies.append({
                        'field': cols[0],
                        'grn_value': cols[1],
                        'invoice_value': cols[2],
                        'discrepancy_type': cols[3],
                        'suggestion': cols[4]
                    })
            
            # Extract summary
            summary = '\n'.join(summary_lines).strip()
            
            return {
                'success': True,
                'discrepancies': discrepancies,
                'summary': summary,
                'total_discrepancies': len(discrepancies),
                'raw_response': llm_response
            }
            
        except Exception as e:
            logger.error(f"Error parsing detailed LLM response: {str(e)}")
            return {
                'success': False,
                'error': f"Failed to parse LLM response: {str(e)}",
                'discrepancies': [],
                'summary': '',
                'raw_response': llm_response
            }
    
    async def _create_reconciliation_record_async(self, invoice: InvoiceData, grn_items: List[ItemWiseGrn],
                                                match_result: Dict[str, Any], llm_analysis: Optional[Dict[str, Any]]) -> InvoiceGrnReconciliation:
        """Create reconciliation record (async)"""
        reconciliation_data = {
            'invoice_data': invoice,
            'po_number': invoice.po_number or '',
            'grn_number': invoice.grn_number or '',
            'invoice_number': invoice.invoice_number or '',
            'match_status': match_result['match_status'],
            'invoice_total': Decimal(str(match_result['invoice_total'])),
            'grn_total': Decimal(str(match_result['grn_total'])),
            'total_variance_pct': Decimal(str(match_result['variance_pct'])),
            'total_grn_line_items': match_result['grn_items_count'],
            'is_auto_matched': True,
            'matching_method': 'async_llm_enhanced'
        }
        
                # Add LLM analysis notes
        if llm_analysis and llm_analysis.get('success'):
            discrepancy_count = llm_analysis.get('total_discrepancies', 0)
            summary = llm_analysis.get('summary', '')[:200]  # Limit length
            notes = f"LLM Analysis: {discrepancy_count} discrepancies found. {summary}"
            reconciliation_data['reconciliation_notes'] = notes
            
            # Store detailed analysis in a separate field if needed
            if discrepancy_count > 0:
                reconciliation_data['requires_review'] = True
        
        # Create record
        reconciliation = await sync_to_async(InvoiceGrnReconciliation.objects.create)(**reconciliation_data)
        return reconciliation
    
    async def _create_no_match_record_async(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Create no-match record (async)"""
        reconciliation_data = {
            'invoice_data': invoice,
            'po_number': invoice.po_number or '',
            'match_status': 'no_grn_found',
            'total_grn_line_items': 0,
            'is_auto_matched': True,
            'reconciliation_notes': 'No matching GRN records found'
        }
        
        reconciliation = await sync_to_async(InvoiceGrnReconciliation.objects.create)(**reconciliation_data)
        
        return {
            'invoice_id': invoice.id,
            'reconciliation_id': reconciliation.id,
            'match_status': 'no_grn_found',
            'variance_pct': 0,
            'llm_discrepancies': 0
        }


# Main function to run async reconciliation
async def run_async_reconciliation(invoice_ids: List[int] = None, delay_seconds: float = 1.0, 
                                 max_concurrent: int = 10, batch_size: int = 100) -> Dict[str, Any]:
    """
    Main function to run async reconciliation
    
    Args:
        invoice_ids: Optional list of invoice IDs
        delay_seconds: Delay between LLM calls (default 1.0)
        max_concurrent: Max concurrent processes (default 10)
        batch_size: Batch size for processing (default 100)
    """
    processor = ReconciliationProcessor(
        delay_seconds=delay_seconds,
        max_concurrent=max_concurrent
    )
    
    return await processor.process_batch_async(
        invoice_ids=invoice_ids,
        batch_size=batch_size
    )
