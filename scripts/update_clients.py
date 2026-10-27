"""
Script to update web client (index.html) and RN client (app/src) for Step R9 parity:
- Vibe color map on first word
- Theme badge
- Key-based requests
"""
from pathlib import Path

# 1. Update backend/static/index.html
index_path = Path("backend/static/index.html")
content = index_path.read_text(encoding="utf-8")

vibe_map_code = '''
    const VIBE_COLOR_MAP = {
      slow:        { bg: 'rgba(139, 92, 246, 0.15)', border: 'rgba(139, 92, 246, 0.35)', color: '#c4b5fd' },
      heavy:       { bg: 'rgba(139, 92, 246, 0.15)', border: 'rgba(139, 92, 246, 0.35)', color: '#c4b5fd' },
      sparse:      { bg: 'rgba(45, 212, 191, 0.15)', border: 'rgba(45, 212, 191, 0.35)', color: '#2dd4bf' },
      intimate:    { bg: 'rgba(45, 212, 191, 0.15)', border: 'rgba(45, 212, 191, 0.35)', color: '#2dd4bf' },
      hushed:      { bg: 'rgba(148, 163, 184, 0.15)', border: 'rgba(148, 163, 184, 0.35)', color: '#cbd5e1' },
      laid:        { bg: 'rgba(251, 191, 36, 0.15)', border: 'rgba(251, 191, 36, 0.35)', color: '#fcd34d' },
      raw:         { bg: 'rgba(251, 146, 60, 0.15)', border: 'rgba(251, 146, 60, 0.35)', color: '#fb923c' },
      psychedelic: { bg: 'rgba(232, 121, 249, 0.15)', border: 'rgba(232, 121, 249, 0.35)', color: '#e879f9' },
      punk:        { bg: 'rgba(239, 68, 68, 0.15)', border: 'rgba(239, 68, 68, 0.35)', color: '#fca5a5' },
      lean:        { bg: 'rgba(196, 181, 253, 0.15)', border: 'rgba(196, 181, 253, 0.35)', color: '#c4b5fd' },
      dark:        { bg: 'rgba(99, 102, 241, 0.15)', border: 'rgba(99, 102, 241, 0.35)', color: '#a5b4fc' },
      bright:      { bg: 'rgba(251, 191, 36, 0.15)', border: 'rgba(251, 191, 36, 0.35)', color: '#fcd34d' },
      folk:        { bg: 'rgba(217, 119, 6, 0.15)', border: 'rgba(217, 119, 6, 0.35)', color: '#fbbf24' },
      dance:       { bg: 'rgba(168, 85, 247, 0.15)', border: 'rgba(168, 85, 247, 0.35)', color: '#d8b4fe' },
      reggaeton:   { bg: 'rgba(234, 88, 12, 0.15)', border: 'rgba(234, 88, 12, 0.35)', color: '#fb923c' },
      reggae:      { bg: 'rgba(34, 197, 94, 0.15)', border: 'rgba(34, 197, 94, 0.35)', color: '#4ade80' },
      warm:        { bg: 'rgba(251, 146, 60, 0.15)', border: 'rgba(251, 146, 60, 0.35)', color: '#fdba74' },
      driving:     { bg: 'rgba(239, 68, 68, 0.15)', border: 'rgba(239, 68, 68, 0.35)', color: '#fca5a5' },
    };

    function getVibeStyle(vibe) {
      if (!vibe) return { bg: 'rgba(255,255,255,0.08)', border: 'rgba(255,255,255,0.15)', color: '#94a3b8' };
      const firstWord = String(vibe).trim().split(/[\\s-]+/)[0].toLowerCase();
      return VIBE_COLOR_MAP[firstWord] || { bg: 'rgba(139, 92, 246, 0.15)', border: 'rgba(139, 92, 246, 0.35)', color: '#c4b5fd' };
    }
'''

if 'const VIBE_COLOR_MAP' not in content:
    target_str = 'async function reloadRecommendations()'
    idx = content.find(target_str)
    assert idx != -1, "Target reloadRecommendations not found"
    content = content[:idx] + vibe_map_code.strip() + '\n\n    ' + content[idx:]

# 2. Update reloadRecommendations to include keyParam
old_reload = 'const res = await fetch(`/api/recommendations/${currentHeroTrack.id}?diversity=${diversity}`);'
new_reload = '''const heroKey = currentHeroTrack.key || (currentHeroTrack.artist && currentHeroTrack.title ? `${currentHeroTrack.artist.toLowerCase().trim()}:${currentHeroTrack.title.toLowerCase().trim()}` : '');
        const keyParam = heroKey ? `&key=${encodeURIComponent(heroKey)}` : '';
        const res = await fetch(`/api/recommendations/${currentHeroTrack.id}?diversity=${diversity}${keyParam}`);'''
if old_reload in content:
    content = content.replace(old_reload, new_reload)

# 3. Update selectSongAndStartFlow to include key params
old_select = '''        // Fetch recommendations and enriched track stats concurrently
        const [recPromise, trackPromise] = await Promise.allSettled([
          fetch(`/api/recommendations/${song.id}?diversity=${divVal}`),
          fetch(`/api/track/${song.id}`),
        ]);'''
new_select = '''        const songKey = song.key || (song.artist && song.title ? `${song.artist.toLowerCase().trim()}:${song.title.toLowerCase().trim()}` : '');
        const keyParam = songKey ? `&key=${encodeURIComponent(songKey)}` : '';
        const trackKeyParam = songKey ? `?key=${encodeURIComponent(songKey)}` : '';
        // Fetch recommendations and enriched track stats concurrently
        const [recPromise, trackPromise] = await Promise.allSettled([
          fetch(`/api/recommendations/${song.id}?diversity=${divVal}${keyParam}`),
          fetch(`/api/track/${song.id}${trackKeyParam}`),
        ]);'''
if old_select in content:
    content = content.replace(old_select, new_select)

# 4. Update hero vibes in renderResults
old_hero_vibes = '''        const vList = (hero.vibes && hero.vibes.length) ? hero.vibes : [];
        vList.forEach(v => {
          const pill = document.createElement('span');
          pill.className = 'vibe-pill';
          pill.textContent = `🌊 ${v}`;
          vibesEl.appendChild(pill);
        });'''
new_hero_vibes = '''        const vList = (hero.vibes && hero.vibes.length) ? hero.vibes : [];
        vList.forEach(v => {
          const pill = document.createElement('span');
          pill.className = 'vibe-pill';
          const st = getVibeStyle(v);
          pill.style.background = st.bg;
          pill.style.borderColor = st.border;
          pill.style.color = st.color;
          pill.textContent = `🌊 ${v}`;
          vibesEl.appendChild(pill);
        });
        if (hero.theme) {
          const themePill = document.createElement('span');
          themePill.className = 'vibe-pill';
          themePill.style.background = 'rgba(245,158,11,0.15)';
          themePill.style.borderColor = 'rgba(245,158,11,0.35)';
          themePill.style.color = '#fbbf24';
          themePill.textContent = `📖 ${hero.theme}`;
          vibesEl.appendChild(themePill);
        }'''
if old_hero_vibes in content:
    content = content.replace(old_hero_vibes, new_hero_vibes)

# 5. Update rec cards in renderResults
old_card_vibes = '''        const vibeTags = (track.vibes || []).slice(0, 1).map(v => `<span class="rec-tag" style="background: rgba(139,92,246,0.15); border-color: rgba(139,92,246,0.3); color: #c4b5fd;">🌊 ${v}</span>`).join('');'''
new_card_vibes = '''        const vibeTags = (track.vibes || []).slice(0, 1).map(v => {
          const st = getVibeStyle(v);
          return `<span class="rec-tag" style="background: ${st.bg}; border-color: ${st.border}; color: ${st.color};">🌊 ${v}</span>`;
        }).join('');
        const themeTag = track.theme ? `<span class="rec-tag" style="background: rgba(245,158,11,0.15); border-color: rgba(245,158,11,0.35); color: #fbbf24;">📖 ${track.theme}</span>` : '';'''
if old_card_vibes in content:
    content = content.replace(old_card_vibes, new_card_vibes)

old_struct = '''            <div class="rec-structural-tags">
              ${vibeTags}
              ${langBadge}'''
new_struct = '''            <div class="rec-structural-tags">
              ${vibeTags}
              ${themeTag}
              ${langBadge}'''
if old_struct in content:
    content = content.replace(old_struct, new_struct)

index_path.write_text(content, encoding="utf-8")
print("Updated backend/static/index.html")
