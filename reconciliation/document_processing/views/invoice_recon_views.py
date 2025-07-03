import asyncio
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from document_processing.utils.invoice_recon import run_async_reconciliation
from document_processing.models import InvoiceGrnReconciliation
from django.db.models import Count

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class AsyncReconciliationAPI(View):
    """
    Simple API for async invoice-GRN reconciliation
    Handles 10,000+ records efficiently with delay system
    """
    
    def post(self, request):
        """
        Start async reconciliation process
        
        POST params:
        - invoice_ids: Optional JSON array of invoice IDs
        - delay_seconds: Delay between LLM calls (default: 1.0)
        - max_concurrent: Max concurrent processes (default: 10)
        - batch_size: Batch size for processing (default: 100)
        """
        return asyncio.run(self._async_post(request))
    
    async def _async_post(self, request):
        try:
            # Parse parameters
            if request.content_type == 'application/json':
                import json
                body = json.loads(request.body.decode('utf-8'))
                invoice_ids = body.get('invoice_ids', None)
                delay_seconds = float(body.get('delay_seconds', 1.0))
                max_concurrent = int(body.get('max_concurrent', 10))
                batch_size = int(body.get('batch_size', 100))
            else:
                invoice_ids_str = request.POST.get('invoice_ids', None)
                if invoice_ids_str:
                    import json
                    invoice_ids = json.loads(invoice_ids_str)
                else:
                    invoice_ids = None
                delay_seconds = float(request.POST.get('delay_seconds', 1.0))
                max_concurrent = int(request.POST.get('max_concurrent', 10))
                batch_size = int(request.POST.get('batch_size', 100))
            
            # Validate parameters
            if delay_seconds < 0 or delay_seconds > 10:
                return JsonResponse({
                    'success': False,
                    'error': 'delay_seconds must be between 0 and 10'
                }, status=400)
            
            if max_concurrent < 1 or max_concurrent > 50:
                return JsonResponse({
                    'success': False,
                    'error': 'max_concurrent must be between 1 and 50'
                }, status=400)
            
            if batch_size < 10 or batch_size > 1000:
                return JsonResponse({
                    'success': False,
                    'error': 'batch_size must be between 10 and 1000'
                }, status=400)
            
            logger.info(f"Starting async reconciliation: delay={delay_seconds}s, concurrent={max_concurrent}, batch={batch_size}")
            
            # Run async reconciliation
            result = await run_async_reconciliation(
                invoice_ids=invoice_ids,
                delay_seconds=delay_seconds,
                max_concurrent=max_concurrent,
                batch_size=batch_size
            )
            
            if result['success']:
                return JsonResponse({
                    'success': True,
                    'message': f"Reconciliation completed successfully",
                    'data': {
                        'total_processed': result['total_processed'],
                        'stats': result['stats'],
                        'processing_params': {
                            'delay_seconds': delay_seconds,
                            'max_concurrent': max_concurrent,
                            'batch_size': batch_size
                        }
                    }
                }, status=200)
            else:
                return JsonResponse({
                    'success': False,
                    'error': result['error'],
                    'stats': result['stats']
                }, status=500)
                
        except Exception as e:
            logger.error(f"Error in async reconciliation API: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'Failed to process reconciliation: {str(e)}'
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class ReconciliationStatusAPI(View):
    """
    API to check reconciliation status and results
    """
    
    def get(self, request):
        """Get reconciliation statistics"""
        try:
            
            
            # Get overall stats
            total_reconciliations = InvoiceGrnReconciliation.objects.count()
            
            status_stats = InvoiceGrnReconciliation.objects.values('match_status').annotate(
                count=Count('id')
            ).order_by('match_status')
            
            # Recent reconciliations (last 24 hours)
            from django.utils import timezone
            from datetime import timedelta
            recent_time = timezone.now() - timedelta(hours=24)
            recent_count = InvoiceGrnReconciliation.objects.filter(
                reconciled_at__gte=recent_time
            ).count()
            
            return JsonResponse({
                'success': True,
                'data': {
                    'total_reconciliations': total_reconciliations,
                    'recent_24h': recent_count,
                    'status_breakdown': list(status_stats),
                    'last_updated': timezone.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting reconciliation status: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
