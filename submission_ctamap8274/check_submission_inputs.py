"""Read-only verification of the final Zenodo package. Never starts training."""
from pathlib import Path
import argparse,csv,hashlib,json
def sha(p):
 with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--package',required=True);args=a.parse_args()
 p=Path(args.package).resolve(); errors=[]
 m=json.loads((p/'metadata/release_manifest.json').read_text(encoding='utf-8'))
 for item in m['files']:
  f=(p/item['file']).resolve()
  if not f.is_relative_to(p): raise ValueError('Manifest path leaves package')
  if not f.is_file() or f.stat().st_size!=item['bytes'] or sha(f)!=item['sha256']: errors.append(item['file'])
 label=p/'training_labels/county_training_labels_8274.csv'
 with label.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 keys={(r['year'],r['county_code']) for r in rows}
 if len(rows)!=8274 or len(keys)!=8274: errors.append('training cohort size or duplicate keys')
 if sha(label)!='073dbc850899192449f042b6d3502cce4620d7f939331844c59e8fe1abdcc482': errors.append('frozen label SHA256')
 boundary=p/'reproducibility/boundary/T2021_county.gpkg'
 if sha(boundary)!='a4e82a867d2d1d5f5b6e7620042edf2f55ae9d7d7f2897700ec628a382357cbd': errors.append('frozen boundary SHA256')
 print(json.dumps(dict(ok=not errors,manifest_entries=len(m['files']),training_records=len(rows),errors=errors,training_started=False),ensure_ascii=False,indent=2))
 raise SystemExit(1 if errors else 0)
if __name__=='__main__': main()
