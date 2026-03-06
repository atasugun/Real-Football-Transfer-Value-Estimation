import sys, json, unicodedata, re
sys.stdout.reconfigure(encoding='utf-8')

_CHAR_MAP = str.maketrans({
    'l': 'l', 'L': 'l',  # placeholder to start dict
})

# Full char map matching app.py
_CHAR_MAP = str.maketrans({
    '\u0142': 'l', '\u0141': 'l',
    '\u00f8': 'o', '\u00d8': 'o',
    '\u00df': 'ss',
    '\u00f0': 'd', '\u00d0': 'd',
    '\u00fe': 'th', '\u00de': 'th',
    '\u00e6': 'ae', '\u00c6': 'ae',
    '\u0153': 'oe', '\u0152': 'oe',
    '\u0111': 'd', '\u0110': 'd',
    '\u2019': '', '\u2018': '', "'": '', '-': ' ',
    '\u0131': 'i', '\u0130': 'i',  # Turkish ı/İ
})

def norm(s):
    s = s.translate(_CHAR_MAP)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()
    s = re.sub(r'[^a-z0-9 ]', '', s)
    return ' '.join(s.split())

# Test
print(norm('Yusuf Sarı'), '==', norm('yusuf sari'), '->', norm('Yusuf Sarı') == norm('yusuf sari'))
print(norm('Sarı'), '->', repr(norm('Sarı')))

# Check players_live.json
with open('players_live.json', encoding='utf-8') as f:
    players = json.load(f)

hits = [p['player_name'] for p in players if 'sari' in norm(p['player_name']) and 'yusuf' in norm(p['player_name'])]
print('Yusuf Sari hits:', hits)

# Also broader - anyone with norm containing 'sari'
sari = [p['player_name'] for p in players if norm(p['player_name']).endswith('sari') or ' sari' in norm(p['player_name'])]
print('*sari players:', sari[:10])
