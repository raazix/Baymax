"""Optional Supermemory retrieval and approved-action memory storage."""
import os
import httpx

BASE_URL = 'https://api.supermemory.ai'
CONTAINER = 'lineguard-project'

def status():
    return {'enabled': bool(os.getenv('SUPERMEMORY_API_KEY')), 'scope': 'approved corrective actions only',
            'container': CONTAINER}

def _headers():
    token = os.getenv('SUPERMEMORY_API_KEY')
    return {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'} if token else None

async def search_precedents(inspection: dict):
    headers = _headers()
    if not headers:
        return []
    labels = ', '.join(dict.fromkeys(d.get('label', 'unknown') for d in inspection.get('defects', []))) or 'no classified defect'
    query = f"Approved corrective actions for {labels}; machine {inspection.get('machine_id')}"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8, connect=3), follow_redirects=False) as client:
            response = await client.post(BASE_URL + '/v4/search', headers=headers,
                json={'query': query, 'containerTags': [CONTAINER], 'searchMode': 'hybrid', 'limit': 3})
            response.raise_for_status()
            results = response.json().get('results', [])
            return [{'content': str(r.get('content', ''))[:900], 'score': r.get('score'),
                     'memory_id': r.get('docId') or r.get('id')} for r in results[:3] if r.get('content')]
    except (httpx.HTTPError, ValueError, TypeError):
        return []

async def remember_approved(inspection: dict):
    """Store only engineer-approved action summaries; never upload the source image."""
    headers = _headers()
    if not headers:
        return {'stored': False, 'reason': 'not_configured'}
    defects = ', '.join(dict.fromkeys(d.get('label', 'unknown') for d in inspection.get('defects', []))) or 'no detected defect'
    content = (f"LineGuard engineer-approved corrective action. Defects: {defects}. "
               f"Disposition: {inspection.get('disposition')}. "
               f"Approved action: {inspection.get('action', {}).get('text', '')}. "
               f"Outcome: pending verification; this record is not proof of defect resolution.")[:4000]
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8, connect=3), follow_redirects=False) as client:
            response = await client.post(BASE_URL + '/v4/memories', headers=headers,
                json={'containerTag': CONTAINER, 'memories': [{'content': content, 'isStatic': False,
                      'metadata': {'type': 'engineer_approved_action', 'inspection_id': inspection['id'],
                                   'verification_status': (inspection.get('verification') or {}).get('status', 'pending')}}]})
            response.raise_for_status()
            return {'stored': True}
    except (httpx.HTTPError, ValueError, TypeError):
        return {'stored': False, 'reason': 'provider_unavailable'}
