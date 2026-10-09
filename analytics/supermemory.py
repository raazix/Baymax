"""Optional Supermemory retrieval and approved-action memory storage.

Memory holds only engineer-approved actions and their verification outcomes (never images). Search results are passed
to the assistant as unverified historical context, never as evidence for the current part.
"""
import os
import httpx

BASE_URL = 'https://api.supermemory.ai'
CONTAINER = 'lineguard-project'


def status():
    return {'enabled': bool(os.getenv('SUPERMEMORY_API_KEY')),
            'scope': 'engineer-approved corrective actions and their verification outcomes only',
            'container': CONTAINER}


def _headers():
    token = os.getenv('SUPERMEMORY_API_KEY')
    return {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'} if token else None


def _labels(inspection):
    labels = []
    for d in inspection.get('defects', []):
        hint = (d.get('class_hint') or {}).get('label')
        labels.append(f"{d.get('label', 'unknown')}{f' (hint {hint})' if hint else ''}")
    return ', '.join(dict.fromkeys(labels)) or 'no detected defect'


def _context_line(inspection):
    analytics = inspection.get('analytics') or {}
    hypothesis = (analytics.get('rca') or {}).get('hypothesis', 'unknown')
    severities = ', '.join(dict.fromkeys((d.get('severity') or {}).get('level', '?') for d in inspection.get('defects', []))) or 'none'
    drift = (inspection.get('context') or {}).get('lot_history') or {}
    drift_text = ', '.join(f"{x['sensor']} {x['change']:+g}" for x in drift.get('drift', [])) or 'none flagged'
    return (f"Machine {inspection.get('machine_id')}, lot {inspection.get('lot_id')}. Defects: {_labels(inspection)}; "
            f"severity {severities}. Process hypothesis (not causal proof): {hypothesis}. Sensor drift: {drift_text}.")


def search_query(inspection):
    hypothesis = ((inspection.get('analytics') or {}).get('rca') or {}).get('hypothesis', '')
    return f"Approved corrective actions for {_labels(inspection)}; {hypothesis.replace('_', ' ')}; machine {inspection.get('machine_id')}"


async def search_precedents(inspection: dict):
    headers = _headers()
    if not headers:
        return []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8, connect=3), follow_redirects=False) as client:
            response = await client.post(BASE_URL + '/v4/search', headers=headers,
                json={'q': search_query(inspection), 'containerTags': [CONTAINER], 'searchMode': 'hybrid', 'limit': 3})
            response.raise_for_status()
            results = response.json().get('results', [])
            return [{'content': str(r.get('memory') or r.get('content') or '')[:900], 'score': r.get('similarity', r.get('score')),
                     'memory_id': r.get('id') or r.get('docId')} for r in results[:3] if r.get('memory') or r.get('content')]
    except (httpx.HTTPError, ValueError, TypeError):
        return []


async def _store(content: str, metadata: dict):
    headers = _headers()
    if not headers:
        return {'stored': False, 'reason': 'not_configured'}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10, connect=3), follow_redirects=False) as client:
            response = await client.post(BASE_URL + '/v4/memories', headers=headers,
                json={'containerTag': CONTAINER, 'memories': [{'content': content[:4000], 'isStatic': False, 'metadata': metadata}]})
            response.raise_for_status()
            return {'stored': True, 'http_status': response.status_code}
    except httpx.HTTPStatusError as error:
        return {'stored': False, 'reason': f'provider_http_{error.response.status_code}'}
    except (httpx.HTTPError, ValueError, TypeError):
        return {'stored': False, 'reason': 'provider_unavailable'}


async def remember_approved(inspection: dict):
    """Store only engineer-approved action summaries; never upload the source image."""
    action = inspection.get('action', {})
    content = (f"LineGuard engineer-approved corrective action. {_context_line(inspection)} "
               f"Approved action: {action.get('text', '')} Approved by {action.get('engineer', 'engineer')}. "
               f"Outcome: pending verification; this record is not proof of defect resolution.")
    return await _store(content, {'type': 'engineer_approved_action', 'inspection_id': inspection['id'],
                                  'machine_id': inspection.get('machine_id'), 'verification_status': 'pending'})


async def remember_verification(inspection: dict):
    """Store the verification outcome so later precedent searches see whether the approved action held."""
    verification = inspection.get('verification') or {}
    content = (f"LineGuard verification outcome. {_context_line(inspection)} "
               f"Approved action: {(inspection.get('action') or {}).get('text', '')} "
               f"Re-inspection: {verification.get('defective')} of {verification.get('parts')} follow-up parts defective "
               f"({verification.get('status', 'unknown').replace('_', ' ')}). A clean sample does not prove the fix in production.")
    return await _store(content, {'type': 'verification_outcome', 'inspection_id': inspection['id'],
                                  'machine_id': inspection.get('machine_id'), 'verification_status': verification.get('status')})
