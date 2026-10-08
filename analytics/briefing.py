"""Grounded evidence packet; useful input for a later optional LLM renderer."""
def evidence_packet(inspection: dict):
    identifier = inspection['id']
    base = f'/api/inspections/{identifier}'
    citations = [{'id': 'inspection', 'url': base, 'pointer': '/defects'},
                 {'id': 'traceability', 'url': base, 'pointer': '/telemetry'},
                 {'id': 'action', 'url': base, 'pointer': '/action'}]
    facts = {'part_id': inspection['part_id'], 'lot_id': inspection['lot_id'], 'machine_id': inspection['machine_id'],
             'source': inspection['source'], 'defects': [{k:v for k,v in d.items() if k != 'anomaly_grid'} for d in inspection['defects']], 'telemetry': inspection['telemetry'],
             'recommended_action': inspection['action']['text'], 'approval_status': inspection['action']['status']}
    facts['quality'] = inspection['quality']
    facts['calibration'] = inspection.get('calibration')
    facts['image_sha256'] = inspection['image_sha256']
    context = inspection.get('context') or {}
    facts['model_context'] = {k:v for k,v in context.items() if k in ('model', 'patchcore_model', 'telemetry_source', 'process_context')}
    if context.get('anomaly'):
        facts['anomaly'] = {k:v for k,v in context['anomaly'].items() if k in ('score', 'threshold', 'flagged')}
    if inspection.get('analytics'):
        facts['analytics'] = inspection['analytics']
        citations.append({'id': 'analytics', 'url': base, 'pointer': '/analytics'})
    return {'status': 'deterministic_evidence_packet', 'llm_used': False, 'facts': facts, 'citations': citations,
            'constraints': ['Synthetic inputs must remain labelled.', 'Do not invent measurements, confidence, causes or SOPs.',
                            'Root-cause hypotheses are not causal proof.', 'No approval or machine-control authority.'],
            'memory_eligibility': 'Unverified hypotheses and pending actions must not be stored as verified incidents.'}
