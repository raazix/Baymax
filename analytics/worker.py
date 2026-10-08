"""Durable local analytics worker. Run one worker alongside the API.

Queued jobs survive API restarts. --recover requeues interrupted work only when
the prior worker is stopped. Calculations are idempotent after a crash.
"""
import argparse
import logging
import time
from core.clock import utc_now
from analytics.service import analyze
from actions.sop_engine import recommend
from database.repository import Repository

logger = logging.getLogger(__name__)

def process_one(repo: Repository) -> bool:
    identifier = repo.claim_job()
    if identifier is None: return False
    job = repo.get_job(identifier)
    try:
        def complete(result):
            if result['action']['status'] != 'awaiting_analytics':
                return result  # Result committed before a previous worker crash.
            analytics = analyze(result['telemetry'], result['seed'], result['model_versions'])
            result['analytics'] = analytics
            result['action'] = recommend(result['defects'], analytics)
            result['audit'].append({'at': utc_now(), 'event': 'analytics_completed', 'job_id': identifier,
                                    'model_versions': analytics['model_versions'], 'seed': result['seed']})
            return result
        repo.update(job['inspection_id'], complete)
        repo.finish_job(identifier, 'completed')
    except Exception:
        logger.exception('Analytics job failed: %s', identifier)
        repo.finish_job(identifier, 'failed', 'Analytics failed; inspect worker logs before retrying.')
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    repo = Repository()
    if args.recover: repo.recover_jobs()
    try:
        if args.once:
            process_one(repo); return
        while True:
            if not process_one(repo): time.sleep(.5)
    except KeyboardInterrupt: pass
    finally: repo.engine.dispose()

if __name__ == '__main__': main()
