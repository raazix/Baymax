export type QualityFinding = { label: string; severity: { level: string } };
export function holdLevel(inspection: { quality: { passed: boolean }; defects: QualityFinding[] }): 'critical' | 'high' | null {
  if (!inspection.quality.passed) return null;
  if (inspection.defects.some(finding => finding.severity.level.toLowerCase() === 'critical')) return 'critical';
  return inspection.defects.some(finding => finding.severity.level.toLowerCase() === 'high') ? 'high' : null;
}
