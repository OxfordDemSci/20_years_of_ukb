"""
Parse IPC (International Patent Classification) XML scheme into a hierarchical CSV.
The IPC has levels: Section > Class > Subclass > Main Group > Subgroup
"""
import xml.etree.ElementTree as ET
import csv
import sys
from pathlib import Path

def extract_title(entry, ns):
    """Extract concatenated title text from ipcEntry."""
    titles = []
    text_body = entry.find('ipc:textBody', ns)
    if text_body is not None:
        title_elem = text_body.find('ipc:title', ns)
        if title_elem is not None:
            for title_part in title_elem.findall('ipc:titlePart', ns):
                text_elem = title_part.find('ipc:text', ns)
                if text_elem is not None and text_elem.text:
                    titles.append(text_elem.text.strip())
    return ' / '.join(titles) if titles else ''

def parse_ipc_entry(entry, ns, parent_code='', level=0, path=''):
    """Recursively parse ipcEntry nodes and yield flat records."""
    kind = entry.get('kind', '')
    symbol = entry.get('symbol', '')
    entry_type = entry.get('entryType', '')
    
    # Map kind codes to readable level names
    kind_map = {
        's': 'Section',
        't': 'Title',
        'c': 'Class',
        'u': 'Subclass',
        'm': 'Main Group',
        'g': 'Group'
    }
    
    level_name = kind_map.get(kind, kind)
    title = extract_title(entry, ns)
    
    # Build hierarchical path
    current_path = f"{path}/{symbol}" if path else symbol
    
    # Only yield if we have a symbol (skip title entries without codes)
    if symbol and kind != 't':
        yield {
            'symbol': symbol,
            'kind': kind,
            'level': level_name,
            'title': title,
            'parent_symbol': parent_code,
            'path': current_path,
            'entry_type': entry_type
        }
    
    # Recursively process children
    for child in entry.findall('ipc:ipcEntry', ns):
        # Use current symbol as parent for children (unless it's a title node)
        next_parent = symbol if symbol and kind != 't' else parent_code
        next_path = current_path if symbol and kind != 't' else path
        yield from parse_ipc_entry(child, ns, next_parent, level + 1, next_path)

def main():
    xml_path = '/Users/valler/Downloads/ipc_scheme_20260101/EN_ipc_scheme_20260101.xml'
    output_path = '/Users/valler/Python/RA/20_years_of_ukb/data/patent/ipc_hierarchy.csv'
    
    print(f"Parsing IPC XML from: {xml_path}")
    print(f"Output CSV will be saved to: {output_path}")
    
    # Parse XML with namespace
    ns = {'ipc': 'http://www.wipo.int/classifications/ipc/masterfiles'}
    
    print("Loading XML file (this may take a moment)...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    print("Extracting hierarchical structure...")
    
    # Collect all records
    records = []
    for entry in root.findall('ipc:ipcEntry', ns):
        for record in parse_ipc_entry(entry, ns):
            records.append(record)
    
    # Write to CSV
    print(f"Writing {len(records):,} records to CSV...")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if records:
            fieldnames = ['symbol', 'kind', 'level', 'title', 'parent_symbol', 'path', 'entry_type']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    
    print(f"✓ Successfully exported {len(records):,} IPC entries to {output_path}")
    
    # Show sample records
    print("\nSample records:")
    for i, record in enumerate(records[:5]):
        print(f"  {record['symbol']:<15} {record['level']:<15} {record['title'][:60]}")
    
    # Show statistics
    from collections import Counter
    level_counts = Counter(r['level'] for r in records)
    print("\nHierarchy breakdown:")
    for level, count in sorted(level_counts.items()):
        print(f"  {level:<15} {count:>6,} entries")

if __name__ == '__main__':
    main()
