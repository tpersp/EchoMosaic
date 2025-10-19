// Theme toggle (sun/moon)
  (function(){
    const root = document.documentElement;
    const btn = document.getElementById('theme-toggle');
    function apply(theme){
      const t = (theme === 'light') ? 'light' : 'dark';
      root.setAttribute('data-theme', t);
      if (btn) btn.textContent = t === 'light' ? '☀️' : '🌙';
    }
    const saved = localStorage.getItem('theme') || 'dark';
    apply(saved);
    if (btn) btn.addEventListener('click', () => {
      const cur = root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
      const next = cur === 'light' ? 'dark' : 'light';
      apply(next);
      try { localStorage.setItem('theme', next); } catch(e){}
    });
  })();

  const socket = io();
  let openAiCard = null;
  const aiSummaryUpdaters = new WeakMap();

  function openAiSettings(card) {
    if (!card) return;
    const updater = aiSummaryUpdaters.get(card);
    if (updater) updater();
    const aiSection = card.querySelector('.ai-generator');
    if (!aiSection) return;
    const backdrop = card.querySelector('.ai-modal-backdrop');
    if (openAiCard && openAiCard !== card) {
      closeAiSettings(openAiCard);
    }
    card.classList.add('ai-settings-open');
    if (backdrop) backdrop.hidden = false;
    aiSection.setAttribute('aria-hidden', 'false');
    if (typeof aiSection.focus === 'function') {
      try {
        aiSection.focus({ preventScroll: true });
      } catch (err) {
        aiSection.focus();
      }
    }
    openAiCard = card;
    document.body.classList.add('ai-modal-active');
  }

  function closeAiSettings(card) {
    if (!card) return;
    const aiSection = card.querySelector('.ai-generator');
    const backdrop = card.querySelector('.ai-modal-backdrop');
    card.classList.remove('ai-settings-open');
    if (backdrop) backdrop.hidden = true;
    if (aiSection) {
      aiSection.setAttribute('aria-hidden', 'true');
    }
    if (openAiCard === card) {
      openAiCard = null;
    }
    if (!openAiCard) {
      document.body.classList.remove('ai-modal-active');
    }
  }

  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (presetManagerModal && !presetManagerModal.hidden) {
      e.preventDefault();
      closePresetManager();
      return;
    }
    if (openAiCard) {
      e.preventDefault();
      closeAiSettings(openAiCard);
    }
  });

  const notification = document.getElementById('notification');
  const addStreamBtn = document.getElementById('add-stream');
  const openMosaicBtn = document.getElementById('open-mosaic');
  
  // Group Manager elements
  const groupTiles = document.getElementById('group-tiles');

  const dashboardGrid = document.getElementById('dashboard-grid');
  const tagFilterChips = document.getElementById('tag-filter-chips');
  const tagFilterInput = document.getElementById('tag-filter-input');
  const sortSelect = document.getElementById('sort-select');
  const tagManagerList = document.getElementById('tag-manager-list');
  const newTagInput = document.getElementById('new-tag-input');
  const createTagBtn = document.getElementById('create-tag-btn');
  const tagDatalist = document.getElementById('global-tag-options');

  // Build a set of taken stream name slugs for client-side validation
  const takenSlugs = {};
  {% for sid, conf in stream_settings.items() %}
  takenSlugs['{{ (conf.label if conf.label else sid)|slugify }}'] = '{{ sid }}';
  {% endfor %}

  const initialGlobalTags = {{ global_tags|tojson }};
  const globalTagState = {
    list: Array.isArray(initialGlobalTags) ? initialGlobalTags.slice() : [],
    map: new Map()
  };
  globalTagState.list.forEach(tag => {
    if (typeof tag === 'string') {
      globalTagState.map.set(tag.toLowerCase(), tag);
    }
  });
  const cardTagsMap = new Map();
  const activeTagFilters = [];
  let currentSortMode = 'default';

  function showNotification(msg) {
    notification.textContent = msg;
    notification.classList.add('show');
    setTimeout(() => {
      notification.classList.remove('show');
    }, 3000);
  }

  if (openMosaicBtn) {
    openMosaicBtn.addEventListener('click', () => {
      window.open('/stream', '_blank');
    });
  }

  function normalizeTagInput(value) {
    if (typeof value !== 'string') {
      return '';
    }
    const cleaned = value.replace(/\s+/g, ' ').trim();
    if (!cleaned) {
      return '';
    }
    return cleaned.length > 48 ? cleaned.slice(0, 48).trim() : cleaned;
  }

  function tagsEqual(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b)) {
      return false;
    }
    if (a.length !== b.length) {
      return false;
    }
    return a.every((tag, idx) => (tag || '').toLowerCase() === (b[idx] || '').toLowerCase());
  }

  function getCardTags(card) {
    if (!card) {
      return [];
    }
    const raw = card.dataset.tags;
    if (!raw) {
      return [];
    }
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.filter(t => typeof t === 'string') : [];
    } catch (err) {
      return [];
    }
  }

  function setCardTags(card, tags) {
    if (!card) {
      return [];
    }
    const cleaned = Array.isArray(tags) ? tags.filter(t => typeof t === 'string') : [];
    card.dataset.tags = JSON.stringify(cleaned);
    cardTagsMap.set(card, cleaned.slice());
    const chipList = card.querySelector('.tag-chip-list');
    if (chipList) {
      const input = chipList.querySelector('.tag-entry');
      chipList.querySelectorAll('.tag-chip').forEach(chip => chip.remove());
      const frag = document.createDocumentFragment();
      cleaned.forEach(tag => {
        const canonical = globalTagState.map.get(tag.toLowerCase()) || tag;
        const chip = document.createElement('span');
        chip.className = 'tag-chip';
        chip.dataset.tag = canonical;
        const label = document.createElement('span');
        label.className = 'tag-label';
        label.textContent = canonical;
        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'tag-remove';
        removeBtn.setAttribute('aria-label', `Remove ${canonical}`);
        removeBtn.innerHTML = '&times;';
        removeBtn.addEventListener('click', () => {
          const current = getCardTags(card);
          const next = current.filter(existing => existing.toLowerCase() !== canonical.toLowerCase());
          if (!tagsEqual(current, next)) {
            requestTagUpdate(card, next);
          }
        });
        chip.appendChild(label);
        chip.appendChild(removeBtn);
        frag.appendChild(chip);
      });
      if (input) {
        chipList.insertBefore(frag, input);
      } else {
        chipList.appendChild(frag);
      }
    }
    return cleaned;
  }

  function requestTagUpdate(card, nextTags) {
    if (!card) {
      return;
    }
    const streamId = card.dataset.stream;
    if (!streamId) {
      return;
    }
    const sanitized = Array.isArray(nextTags) ? nextTags.map(normalizeTagInput).filter(Boolean) : [];
    const current = getCardTags(card);
    if (tagsEqual(current, sanitized)) {
      return;
    }
    const editor = card.querySelector('.tag-editor');
    const input = editor ? editor.querySelector('.tag-entry') : null;
    const buttons = editor ? Array.from(editor.querySelectorAll('.tag-remove')) : [];
    if (input) {
      input.disabled = true;
    }
    buttons.forEach(btn => { btn.disabled = true; });
    saveSettings(streamId, { tags: sanitized }, {
      onError: () => { setCardTags(card, current); },
      onFinally: () => {
        if (input) {
          input.disabled = false;
        }
        buttons.forEach(btn => { btn.disabled = false; });
      }
    });
  }

  function attachTagEditor(card) {
    if (!card || !card.dataset.stream) {
      return;
    }
    const editor = card.querySelector('.tag-editor');
    if (!editor) {
      return;
    }
    const input = editor.querySelector('.tag-entry');
    const initial = getCardTags(card);
    setCardTags(card, initial);
    if (input) {
      const commit = () => {
        const normalized = normalizeTagInput(input.value);
        input.value = '';
        if (!normalized) {
          return;
        }
        const current = getCardTags(card);
        if (current.some(tag => tag.toLowerCase() === normalized.toLowerCase())) {
          return;
        }
        requestTagUpdate(card, current.concat([normalized]));
      };
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ',') {
          e.preventDefault();
          commit();
        }
      });
      input.addEventListener('blur', () => {
        commit();
      });
    }
  }

  function updateTagDatalist() {
    if (!tagDatalist) {
      return;
    }
    tagDatalist.innerHTML = '';
    globalTagState.list.forEach(tag => {
      if (typeof tag !== 'string') {
        return;
      }
      const opt = document.createElement('option');
      opt.value = tag;
      tagDatalist.appendChild(opt);
    });
  }

  function renderTagManager() {
    if (!tagManagerList) {
      return;
    }
    tagManagerList.innerHTML = '';
    if (!globalTagState.list.length) {
      const empty = document.createElement('p');
      empty.className = 'tag-manager-empty';
      empty.textContent = 'No tags defined yet.';
      tagManagerList.appendChild(empty);
      return;
    }
    const usage = new Map();
    cardTagsMap.forEach(tags => {
      tags.forEach(tag => {
        const key = tag.toLowerCase();
        usage.set(key, (usage.get(key) || 0) + 1);
      });
    });
    globalTagState.list.forEach(tag => {
      const key = tag.toLowerCase();
      const count = usage.get(key) || 0;
      const row = document.createElement('div');
      row.className = 'tag-manager-item';
      const chip = document.createElement('span');
      chip.className = 'tag-chip';
      const label = document.createElement('span');
      label.className = 'tag-label';
      label.textContent = tag;
      chip.appendChild(label);
      const countEl = document.createElement('span');
      countEl.className = 'tag-usage';
      countEl.textContent = `${count} stream${count === 1 ? '' : 's'}`;
      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'tag-remove-btn';
      removeBtn.textContent = 'Remove';
      if (count > 0) {
        removeBtn.disabled = true;
        removeBtn.title = 'Tag is in use';
      } else {
        removeBtn.addEventListener('click', () => removeGlobalTag(tag));
      }
      row.appendChild(chip);
      row.appendChild(countEl);
      row.appendChild(removeBtn);
      tagManagerList.appendChild(row);
    });
  }

  function removeGlobalTag(tag) {
    const encoded = encodeURIComponent(tag);
    fetch(`/tags/${encoded}`, { method: 'DELETE' })
      .then(res => res.json().then(data => ({ ok: res.ok, data })))
      .then(result => {
        if (result.ok) {
          const tags = Array.isArray(result.data.tags) ? result.data.tags : [];
          syncGlobalTags(tags);
          showNotification('Tag removed');
        } else {
          showNotification(result.data.error || 'Unable to remove tag');
        }
      })
      .catch(() => showNotification('Unable to remove tag'));
  }

  function syncGlobalTags(tags) {
    if (!Array.isArray(tags)) {
      return;
    }
    globalTagState.list = tags.slice();
    globalTagState.map.clear();
    globalTagState.list.forEach(tag => {
      if (typeof tag === 'string') {
        globalTagState.map.set(tag.toLowerCase(), tag);
      }
    });
    updateTagDatalist();
    reconcileFilterTags();
    renderTagManager();
  }

  function reconcileFilterTags() {
    if (!activeTagFilters.length) {
      return;
    }
    const next = [];
    activeTagFilters.forEach(tag => {
      const canonical = globalTagState.map.get(tag.toLowerCase());
      if (canonical) {
        if (!next.some(existing => existing.toLowerCase() === canonical.toLowerCase())) {
          next.push(canonical);
        }
      }
    });
    const changed = next.length !== activeTagFilters.length || next.some((tag, idx) => tag !== activeTagFilters[idx]);
    if (changed) {
      activeTagFilters.length = 0;
      next.forEach(tag => activeTagFilters.push(tag));
      renderFilterChips();
      applyFiltersAndSorting();
    }
  }

  function renderFilterChips() {
    if (!tagFilterChips) {
      return;
    }
    const input = tagFilterInput || null;
    tagFilterChips.querySelectorAll('.filter-tag-chip').forEach(chip => chip.remove());
    activeTagFilters.forEach(tag => {
      const chip = document.createElement('span');
      chip.className = 'tag-chip filter-tag-chip';
      chip.dataset.tag = tag;
      const label = document.createElement('span');
      label.className = 'tag-label';
      label.textContent = tag;
      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'tag-remove';
      removeBtn.setAttribute('aria-label', `Remove ${tag}`);
      removeBtn.innerHTML = '&times;';
      removeBtn.addEventListener('click', () => removeFilterTag(tag));
      chip.appendChild(label);
      chip.appendChild(removeBtn);
      if (input) {
        tagFilterChips.insertBefore(chip, input);
      } else {
        tagFilterChips.appendChild(chip);
      }
    });
  }

  function addFilterTag(tag) {
    const normalized = normalizeTagInput(tag);
    if (!normalized) {
      return;
    }
    const lower = normalized.toLowerCase();
    if (activeTagFilters.some(existing => existing.toLowerCase() === lower)) {
      return;
    }
    const canonical = globalTagState.map.get(lower) || normalized;
    activeTagFilters.push(canonical);
    renderFilterChips();
    applyFiltersAndSorting();
  }

  function removeFilterTag(tag) {
    const lower = tag.toLowerCase();
    const filtered = activeTagFilters.filter(existing => existing.toLowerCase() !== lower);
    if (filtered.length === activeTagFilters.length) {
      return;
    }
    activeTagFilters.length = 0;
    filtered.forEach(item => activeTagFilters.push(item));
    renderFilterChips();
    applyFiltersAndSorting();
  }

  let originalCardOrder = [];

  function applyFiltersAndSorting() {
    if (!dashboardGrid) {
      return;
    }
    const filterSet = new Set(activeTagFilters.map(tag => tag.toLowerCase()));
    cardTagsMap.forEach((tags, card) => {
      const matches = filterSet.size === 0 || tags.some(tag => filterSet.has(tag.toLowerCase()));
      card.dataset.matchesFilter = matches ? 'true' : 'false';
      card.style.display = matches ? '' : 'none';
    });
    while (dashboardGrid.firstChild) {
      dashboardGrid.removeChild(dashboardGrid.firstChild);
    }
    if (currentSortMode === 'group') {
      renderGroupedCards();
    } else {
      originalCardOrder.forEach(card => {
        dashboardGrid.appendChild(card);
      });
    }
  }

  function renderGroupedCards() {
    if (!dashboardGrid) {
      return;
    }
    const tagBuckets = new Map();
    const untagged = [];
    originalCardOrder.forEach(card => {
      const tags = cardTagsMap.get(card) || [];
      if (tags.length) {
        const primary = tags[0];
        const canonical = globalTagState.map.get(primary.toLowerCase()) || primary;
        if (!tagBuckets.has(canonical)) {
          tagBuckets.set(canonical, []);
        }
        tagBuckets.get(canonical).push(card);
      } else {
        untagged.push(card);
      }
    });
    const orderedKeys = [];
    globalTagState.list.forEach(tag => {
      if (tagBuckets.has(tag)) {
        orderedKeys.push(tag);
      }
    });
    tagBuckets.forEach((_, key) => {
      if (!orderedKeys.includes(key)) {
        orderedKeys.push(key);
      }
    });
    orderedKeys.forEach(key => {
      const header = document.createElement('div');
      header.className = 'tag-group-header';
      header.textContent = key;
      const cards = tagBuckets.get(key) || [];
      const anyVisible = cards.some(card => card.dataset.matchesFilter === 'true');
      header.hidden = !anyVisible;
      dashboardGrid.appendChild(header);
      cards.forEach(card => {
        dashboardGrid.appendChild(card);
      });
    });
    if (untagged.length) {
      const header = document.createElement('div');
      header.className = 'tag-group-header';
      header.textContent = 'Untagged';
      const anyVisible = untagged.some(card => card.dataset.matchesFilter === 'true');
      header.hidden = !anyVisible;
      dashboardGrid.appendChild(header);
      untagged.forEach(card => {
        dashboardGrid.appendChild(card);
      });
    }
  }

  function createGlobalTag() {
    if (!newTagInput) {
      return;
    }
    const value = normalizeTagInput(newTagInput.value);
    if (!value) {
      newTagInput.value = '';
      return;
    }
    if (createTagBtn) {
      createTagBtn.disabled = true;
    }
    fetch('/tags', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: value })
    })
      .then(res => res.json().then(data => ({ ok: res.ok, data })))
      .then(result => {
        if (result.ok) {
          const tags = Array.isArray(result.data.tags) ? result.data.tags : [];
          syncGlobalTags(tags);
          if (newTagInput) {
            newTagInput.value = '';
          }
          const created = typeof result.data.tag === 'string' ? result.data.tag : value;
          showNotification(`Tag '${created}' saved`);
        } else {
          showNotification(result.data.error || 'Unable to create tag');
        }
      })
      .catch(() => showNotification('Unable to create tag'))
      .finally(() => {
        if (createTagBtn) {
          createTagBtn.disabled = false;
        }
      });
  }

  function commitFilterInput() {
    if (!tagFilterInput) {
      return;
    }
    const normalized = normalizeTagInput(tagFilterInput.value);
    tagFilterInput.value = '';
    if (!normalized) {
      return;
    }
    addFilterTag(normalized);
  }

  updateTagDatalist();

  function refreshFoldersForCard(card, hideNsfw) {
    const folderSelect = card.querySelector('.folder-select');
    if (!folderSelect) {
      return Promise.resolve();
    }
    const streamId = card.dataset.stream;
    const url = hideNsfw ? '/folders?hide_nsfw=1' : '/folders';
    return fetch(url)
      .then(r => {
        if (!r.ok) {
          throw new Error('Failed to fetch folders');
        }
        return r.json();
      })
      .then(folders => {
        if (!Array.isArray(folders)) {
          return null;
        }
        const folderSet = new Set(folders);
        const assigned = folderSelect.dataset.currentFolder || 'all';
        const assignedLower = (assigned || '').toLowerCase();
        const shouldShowPlaceholder = Boolean(hideNsfw) && assigned && assignedLower.includes('nsfw') && !folderSet.has(assigned);
        const frag = document.createDocumentFragment();
        if (shouldShowPlaceholder) {
          const placeholder = document.createElement('option');
          placeholder.value = '__filtered__';
          placeholder.textContent = `Filtered (${assigned})`;
          placeholder.title = assigned;
          placeholder.dataset.filtered = 'true';
          placeholder.selected = true;
          frag.appendChild(placeholder);
        }
        folders.forEach(f => {
          const opt = document.createElement('option');
          opt.value = f;
          opt.title = f;
          opt.textContent = f;
          if (!shouldShowPlaceholder && assigned === f) {
            opt.selected = true;
          }
          frag.appendChild(opt);
        });
        folderSelect.replaceChildren(frag);
        let effectiveFolder;
        if (shouldShowPlaceholder) {
          folderSelect.value = '__filtered__';
          folderSelect.title = assigned;
          folderSelect.dataset.currentFolder = assigned;
          effectiveFolder = assigned || 'all';
        } else if (folderSet.has(assigned)) {
          folderSelect.value = assigned;
          folderSelect.dataset.currentFolder = assigned;
          effectiveFolder = assigned || 'all';
        } else {
          const fallback = folders[0] || 'all';
          folderSelect.value = fallback;
          folderSelect.dataset.currentFolder = fallback;
          effectiveFolder = fallback;
        }
        const selectedOption = folderSelect.options[folderSelect.selectedIndex];
        if (selectedOption) {
          folderSelect.title = selectedOption.title || selectedOption.textContent || folderSelect.value;
        } else {
          folderSelect.title = folderSelect.value || assigned;
        }
        card.dataset.hideNsfw = hideNsfw ? 'true' : 'false';
        return { effectiveFolder };
      })
      .then(result => {
        if (!result) {
          return;
        }
        const { effectiveFolder } = result;
        if (streamId) {
          loadImagesFor(streamId, effectiveFolder);
        }
      })
      .catch(err => {
        console.error('Failed to refresh folders', err);
        showNotification('Failed to refresh folder list');
      });
  }

  // Group manager logic
  function makeEl(tag, attrs = {}, children = []) {
    const e = document.createElement(tag);
    Object.entries(attrs).forEach(([k,v]) => { if (k==='class') e.className=v; else if (k==='style') e.style.cssText=v; else e.setAttribute(k,v); });
    children.forEach(c => e.appendChild(typeof c==='string' ? document.createTextNode(c) : c));
    return e;
  }

  // Simple client-side URL type detection so users know what will embed
  function detectUrlType(url) {
    if (!url) return '';
    const u = (url||'').toLowerCase();
    if (u.includes('youtube.com') || u.includes('youtu.be/')) return 'YouTube';
    if (u.includes('twitch.tv/')) return 'Twitch';
    if (u.endsWith('.m3u8') || u.endsWith('.mpd')) return 'HLS';
    if (u.startsWith('http://') || u.startsWith('https://')) return 'Website';
    return '';
  }
  // Persist expanded/collapsed tile state
  function getOpenSet() {
    try { return new Set(JSON.parse(localStorage.getItem('gmOpen') || '[]')); } catch { return new Set(); }
  }
  function setOpenSet(s) { try { localStorage.setItem('gmOpen', JSON.stringify(Array.from(s))); } catch {} }
  function markOpen(name, open) {
    if (!name) return; // only persist named groups
    const s = getOpenSet();
    if (open) s.add(name); else s.delete(name);
    setOpenSet(s);
  }
  let gmOpenSet = new Set();
  let gmCurrentGroups = {};
  function loadOpenSet() {
    try { gmOpenSet = new Set(JSON.parse(localStorage.getItem('gmOpen') || '[]')); } catch (e) { gmOpenSet = new Set(); }
  }
  function persistOpenSet() {
    try { localStorage.setItem('gmOpen', JSON.stringify(Array.from(gmOpenSet))); } catch (e) {}
  }
  function markOpen(name, open) {
    if (!name) return;
    if (open) gmOpenSet.add(name); else gmOpenSet.delete(name);
    persistOpenSet();
  }
  function isOpen(name) { return name && gmOpenSet.has(name); }

  async function loadGroupsUI() {
    try {
      loadOpenSet();
      const [streamsMeta, groupsData] = await Promise.all([
        fetch('/streams_meta').then(r=>r.json()),
        fetch('/groups').then(r=>r.json())
      ]);
      const groups = groupsData || {};
      gmCurrentGroups = groups;
      // Build tiles
      if (groupTiles) {
        groupTiles.innerHTML = '';
        // Add tile
        const addTile = makeEl('div', {class:'group-tile add'}, [
          makeEl('div', {class:'tile-header'}, ['+ New Group'])
        ]);
        addTile.addEventListener('click', () => {
          const tile = createGroupTile(streamsMeta, groups, null);
          groupTiles.prepend(tile);
          // open editor immediately
          const body = tile.querySelector('.tile-body');
          if (body) body.hidden = false;
          tile.classList.add('expanded');
          const nameInput = tile.querySelector('.tile-name');
          if (nameInput) nameInput.focus();
        });
        groupTiles.appendChild(addTile);

        // Existing group tiles
        Object.keys(groups).forEach(name => {
          groupTiles.appendChild(createGroupTile(streamsMeta, groups, name));
        });
      }
    } catch (e) { console.error('Failed to load groups UI', e); }
  }
  function selectedIds(container) {
    return Array.from(container.querySelectorAll('input[type="checkbox"]:checked')).map(c=>c.value);
  }
  function createGroupTile(streamsMeta, groups, name) {
    const gdata = name ? (groups[name]||[]) : [];
    const initial = Array.isArray(gdata) ? gdata.slice() : ((gdata?.streams || []).slice());
    const order = Array.isArray(initial) ? initial.slice() : [];
    const gLayout = Array.isArray(gdata) ? {} : (gdata?.layout || {});
    function uniqueMembers() { return Array.from(new Set(order)); }
    function hasMember(id) { return order.includes(id); }
    const isNew = !name;
    const tile = makeEl('div', {class:'group-tile'}, []);
    const header = makeEl('div', {class:'tile-header'}, []);
    const title = makeEl('span', {class:'tile-title'}, [name || 'New Group']);
    const actions = makeEl('div', {class:'tile-actions'}, []);
    const open = name ? makeEl('a', {href:`/stream/group/${encodeURIComponent(name)}`, target:'_blank'}, ['Open']) : null;
    const edit = makeEl('button', {class:'tile-edit'}, ['Edit']);
    const del = name ? makeEl('button', {class:'tile-del'}, ['Delete']) : null;
    if (open) actions.appendChild(open);
    actions.appendChild(edit);
    if (del) actions.appendChild(del);
    header.appendChild(title);
    header.appendChild(actions);
    tile.appendChild(header);
    const body = makeEl('div', {class:'tile-body', hidden: !isOpen(name)}, []);
    if (isOpen(name)) tile.classList.add('expanded');
    const nameRow = makeEl('div', {class:'tile-row'}, [
      makeEl('label', {}, ['Name: ', makeEl('input', {type:'text', class:'tile-name', value: name || ''}, [])])
    ]);
    body.appendChild(nameRow);
    // Layout selection + visual gallery
    const layoutRow = makeEl('div', {class:'tile-row'}, []);
    const layoutSelect = makeEl('select', {class:'tile-layout'}, []);
    ['grid','focus','pip'].forEach(opt => {
      const o = makeEl('option', {value:opt}, [opt.charAt(0).toUpperCase()+opt.slice(1)]);
      layoutSelect.appendChild(o);
    });
    layoutRow.appendChild(makeEl('label', {}, ['Layout: ', layoutSelect]));
    const colsInput = makeEl('input', {type:'number', min:'1', max:'8', class:'tile-cols', value: gLayout.cols || 2}, []);
    const colsWrap = makeEl('span', {class:'tile-cols-wrap'}, [' Cols: ', colsInput]);
    layoutRow.appendChild(colsWrap);
    const rowsInput = makeEl('input', {type:'number', min:'1', max:'8', class:'tile-rows', value: gLayout.rows || 2}, []);
    const rowsWrap = makeEl('span', {class:'tile-rows-wrap'}, [' Rows: ', rowsInput]);
    layoutRow.appendChild(rowsWrap);
    // Focus options
    const focusWrap = makeEl('div', {class:'tile-pip-wrap', style:'display:none;'}, []);
    const focusMode = makeEl('select', {class:'tile-focus-mode'}, []);
    ['1-2','1-3','1-5'].forEach(v => focusMode.appendChild(makeEl('option', {value:v}, [v.replace('-', ' + ')])));
    const focusPos = makeEl('select', {class:'tile-focus-pos'}, []);
    const focusMain = makeEl('select', {class:'tile-focus-main'}, []);
    const fRow1 = makeEl('div', {class:'pip-row'}, [ makeEl('label', {}, ['Focus: ']), focusMode ]);
    const fRow2 = makeEl('div', {class:'pip-row'}, [ makeEl('label', {}, ['Placement: ']), focusPos ]);
    const fRow3 = makeEl('div', {class:'pip-row'}, [ makeEl('label', {}, ['Main: ']), focusMain ]);
    focusWrap.appendChild(fRow1); focusWrap.appendChild(fRow2); focusWrap.appendChild(fRow3);
    layoutRow.appendChild(focusWrap);
    // PIP options
    const pipWrap = makeEl('div', {class:'tile-pip-wrap', style:'display:none;'}, []);
    const pipMain = makeEl('select', {class:'tile-pip-main'}, []);
    const pipPip = makeEl('select', {class:'tile-pip-pip'}, []);
    const pipCorner = makeEl('select', {class:'tile-pip-corner'}, []);
    ['top-left','top-right','bottom-left','bottom-right'].forEach(c => pipCorner.appendChild(makeEl('option', {value:c}, [c])));
    const pipSize = makeEl('input', {type:'number', min:'10', max:'50', class:'tile-pip-size', value: gLayout.pip_size || 25}, []);
    const rowMain = makeEl('div', {class:'pip-row'}, [ makeEl('label', {}, ['Main: ']), pipMain ]);
    const rowPip = makeEl('div', {class:'pip-row'}, [ makeEl('label', {}, ['PIP: ']), pipPip ]);
    const rowCorner = makeEl('div', {class:'pip-row'}, [ makeEl('label', {}, ['Corner: ']), pipCorner ]);
    const rowSize = makeEl('div', {class:'pip-row'}, [ makeEl('label', {}, ['Size: ']), pipSize ]);
    pipWrap.appendChild(rowMain);
    pipWrap.appendChild(rowPip);
    pipWrap.appendChild(rowCorner);
    pipWrap.appendChild(rowSize);
    layoutRow.appendChild(pipWrap);
    body.appendChild(layoutRow);

    // Visual layout gallery
    const gallery = makeEl('div', {class:'layout-gallery'}, []);
    function buildGridPreview(rows, cols) {
      const p = makeEl('div', {class:'layout-preview'}, []);
      p.style.display = 'grid';
      p.style.gridTemplateColumns = `repeat(${Math.max(1, cols)}, 1fr)`;
      p.style.gridTemplateRows = `repeat(${Math.max(1, rows)}, 1fr)`;
      const n = Math.max(rows*cols, 4);
      for (let i=0;i<n;i++) p.appendChild(makeEl('div', {class:'cell'}, []));
      return p;
    }
    function buildFocusPreview(mode, pos) {
      // produce a small grid preview based on mode/pos
      if (mode === '1-2') {
        const p = makeEl('div', {class:'layout-preview'}, []);
        p.style.display = 'grid'; p.style.gridTemplateColumns='repeat(2,1fr)'; p.style.gridTemplateRows='repeat(2,1fr)';
        const order = (pos==='right') ? ['a','main','b','main'] : ['main','a','main','b'];
        order.forEach(()=>p.appendChild(makeEl('div',{class:'cell'},[])));
        // tint main cells
        Array.from(p.children).forEach((c,i)=>{ if (order[i]==='main') c.style.background='#262626'; });
        return p;
      }
      if (mode === '1-3') {
        const p = makeEl('div', {class:'layout-preview'}, []);
        p.style.display='grid'; p.style.gridTemplateColumns='repeat(3,1fr)'; p.style.gridTemplateRows='repeat(2,1fr)';
        const top = (pos==='top');
        const order = top ? ['m','m','m','a','b','c'] : ['a','b','c','m','m','m'];
        order.forEach(()=>p.appendChild(makeEl('div',{class:'cell'},[])));
        Array.from(p.children).forEach((c,i)=>{ if (order[i]==='m') c.style.background='#262626'; });
        return p;
      }
      // 1-5 default
      const p = makeEl('div', {class:'layout-preview'}, []);
      p.style.display='grid'; p.style.gridTemplateColumns='repeat(3,1fr)'; p.style.gridTemplateRows='repeat(3,1fr)';
      let mat;
      switch(pos){
        case 'top-left': mat=[[1,1,0],[1,1,0],[0,0,0]]; break;
        case 'top-right': mat=[[0,1,1],[0,1,1],[0,0,0]]; break;
        case 'bottom-left': mat=[[0,0,0],[1,1,0],[1,1,0]]; break;
        default: mat=[[0,0,0],[0,1,1],[0,1,1]]; // bottom-right
      }
      for (let r=0;r<3;r++) for (let c=0;c<3;c++) {
        const cell = makeEl('div',{class:'cell'},[]);
        if (mat[r][c]===1) cell.style.background='#262626';
        p.appendChild(cell);
      }
      return p;
    }
    function buildPipPreview() {
      const p = makeEl('div', {class:'layout-preview'}, []);
      const main = makeEl('div', {class:'cell'}, []); main.style.height = '100%'; main.style.width='100%';
      main.style.background = '#1d1d1d';
      const pip = makeEl('div', {class:'cell'}, []);
      pip.style.position='absolute'; pip.style.width='35%'; pip.style.height='35%'; pip.style.right='6%'; pip.style.bottom='6%'; pip.style.background='#262626';
      p.appendChild(main); p.appendChild(pip);
      return p;
    }
    function addOption(kind, label, onPick, builder) {
      const opt = makeEl('div', {class:'layout-option', 'data-kind': kind}, []);
      const preview = builder ? builder() : makeEl('div', {class:'layout-preview'}, []);
      opt.appendChild(preview);
      opt.appendChild(makeEl('span', {class:'caption'}, [label]));
      opt.addEventListener('click', () => { onPick(); setActive(kind); });
      gallery.appendChild(opt);
      return opt;
    }
    function setActive(kind) {
      gallery.querySelectorAll('.layout-option').forEach(o => o.classList.toggle('active', o.dataset.kind===kind));
    }
    const gridOpt = addOption('grid', 'Grid', () => { layoutSelect.value='grid'; updateLayoutVisibility(); }, () => buildGridPreview(parseInt(rowsInput.value||'2',10), parseInt(colsInput.value||'2',10)));
    const focusOpt = addOption('focus', 'Focus', () => { layoutSelect.value='focus'; updateLayoutVisibility(); }, () => buildFocusPreview(focusMode.value||'1-5', focusPos.value||'bottom-right'));
    const pipOpt = addOption('pip', 'PiP', () => { layoutSelect.value='pip'; updateLayoutVisibility(); }, () => buildPipPreview());
    body.appendChild(gallery);
    function syncActiveFromInputs() {
      const v = layoutSelect.value;
      if (v==='grid') setActive('grid'); else setActive(v);
      // update grid preview on change
      const oldPrev = gridOpt.querySelector('.layout-preview');
      const newPrev = buildGridPreview(parseInt(rowsInput.value||'2',10), parseInt(colsInput.value||'2',10));
      gridOpt.replaceChild(newPrev, oldPrev);
      // update focus preview on change
      const oldF = focusOpt.querySelector('.layout-preview');
      const newF = buildFocusPreview(focusMode.value||'1-5', focusPos.value||'bottom-right');
      focusOpt.replaceChild(newF, oldF);
      refreshChips();
    }
    layoutSelect.addEventListener('change', syncActiveFromInputs);
    colsInput.addEventListener('change', syncActiveFromInputs);
    rowsInput.addEventListener('change', syncActiveFromInputs);
    focusMode.addEventListener('change', () => { updateLayoutVisibility(); syncActiveFromInputs(); refreshFocusPosOptions(); });
    focusPos.addEventListener('change', syncActiveFromInputs);

    function refreshFocusPosOptions() {
      // update placement options according to mode
      const mode = focusMode.value;
      focusPos.innerHTML = '';
      let opts = [];
      if (mode === '1-2') opts = ['left','right'];
      else if (mode === '1-3') opts = ['top','bottom'];
      else opts = ['top-left','top-right','bottom-left','bottom-right'];
      opts.forEach(v => focusPos.appendChild(makeEl('option',{value:v},[v.replace('-', ' ')])));
      // select from gLayout or defaults
      const desired = gLayout.focus_pos || opts[opts.length-1];
      Array.from(focusPos.options).forEach(o => o.selected = (o.value===desired));
    }
    // Bulk row (Add all / Remove all) above the combo
    const bulkRow = makeEl('div', {class:'tile-row tile-bulk-row'}, []);
    const addAllBtn = makeEl('button', {class:'tile-addall-btn', type:'button'}, ['Add all']);
    const removeAllBtn = makeEl('button', {class:'tile-removeall-btn', type:'button'}, ['Remove all']);
    bulkRow.appendChild(addAllBtn);
    bulkRow.appendChild(removeAllBtn);
    body.appendChild(bulkRow);
    // Add row with combined search/select + Add
    const addRow = makeEl('div', {class:'tile-row tile-add-row'}, []);
    const dlId = `dl-${Math.random().toString(36).slice(2)}`;
    const combo = makeEl('input', {type:'text', class:'tile-combo', placeholder:'Search or select…', list: dlId}, []);
    const datalist = makeEl('datalist', {id: dlId}, []);
    const addBtn = makeEl('button', {class:'tile-add-btn', type:'button'}, ['Add']);
    addRow.appendChild(combo);
    addRow.appendChild(addBtn);
    body.appendChild(addRow);
    body.appendChild(datalist);
    // Chips container for current members
    const chips = makeEl('div', {class:'member-chips'}, []);
    body.appendChild(chips);
    // helpers to refresh UI
    function labelOf(id) { return (streamsMeta[id]?.label || id) + ''; }
    function candidates(query) {
      const q = (query||'').toLowerCase();
      return Object.keys(streamsMeta)
        .filter(id => labelOf(id).toLowerCase().includes(q))
        .sort((a,b) => labelOf(a).localeCompare(labelOf(b)));
    }
    function refreshTitleCount() {
      const count = order.length;
      title.textContent = (tile.querySelector('.tile-name')?.value?.trim() || name || 'New Group') + ` (${count})`;
    }
    function refreshPipSelects() {
      const mem = uniqueMembers();
      let mainSel = gLayout.pip_main && hasMember(gLayout.pip_main) ? gLayout.pip_main : (mem[0] || '');
      let pipSel = gLayout.pip_pip && hasMember(gLayout.pip_pip) ? gLayout.pip_pip : ((mem.length > 1 ? mem[1] : mem[0]) || '');
      // fill main
      pipMain.innerHTML = '';
      mem.forEach(id => {
        const o = makeEl('option', {value:id}, [labelOf(id)]);
        if (id === mainSel) o.selected = true;
        pipMain.appendChild(o);
      });
      // fill pip selection
      pipPip.innerHTML = '';
      mem.forEach(id => {
        const o = makeEl('option', {value:id}, [labelOf(id)]);
        if (id === pipSel) o.selected = true;
        pipPip.appendChild(o);
      });
      pipMain.value = mainSel || '';
      pipPip.value = pipSel || '';
      gLayout.pip_main = pipMain.value || null;
      gLayout.pip_pip = pipPip.value || null;
      // corner and size
      Array.from(pipCorner.options).forEach(opt => opt.selected = (opt.value === (gLayout.pip_corner || 'bottom-right')));
      pipSize.value = gLayout.pip_size || 25;
    }
    function refreshFocusMainOptions() {
      const mem = uniqueMembers();
      focusMain.innerHTML = '';
      mem.forEach(id => {
        const o = makeEl('option', {value:id}, [labelOf(id)]);
        focusMain.appendChild(o);
      });
      const desired = gLayout.focus_main && hasMember(gLayout.focus_main) ? gLayout.focus_main : (mem[0] || '');
      Array.from(focusMain.options).forEach(o => o.selected = (o.value===desired));
      focusMain.value = desired || '';
      gLayout.focus_main = focusMain.value || null;
    }
    function updateLayoutVisibility() {
      const v = layoutSelect.value;
      colsWrap.style.display = (v==='grid') ? '' : 'none';
      rowsWrap.style.display = (v==='grid') ? '' : 'none';
      focusWrap.style.display = (v==='focus') ? '' : 'none';
      pipWrap.style.display = (v==='pip') ? '' : 'none';
      if (v==='pip') refreshPipSelects();
      refreshChips();
    }
    function refreshDatalist() {
      datalist.innerHTML = '';
      candidates(combo.value).forEach(id => {
        datalist.appendChild(makeEl('option', {value: labelOf(id)}, []));
      });
    }
    function capacityForLayout() {
      const v = layoutSelect.value;
      if (v==='grid') {
        const r = parseInt(rowsInput.value||'');
        const c = parseInt(colsInput.value||'');
        if (Number.isFinite(r) && Number.isFinite(c) && r>0 && c>0) return r*c;
        return Infinity;
      }
      if (v==='focus') {
        const m = (focusMode.value||'1-5');
        if (m==='1-2') return 3; if (m==='1-3') return 4; return 6;
      }
      if (v==='pip') return 2;
      return Infinity;
    }
    let draggingIndex = null;
    function refreshChips() {
      chips.innerHTML = '';
      const list = makeEl('ul', {class:'member-list'}, []);
      const cap = capacityForLayout();
      order.forEach((id, idx) => {
        const li = makeEl('li', {class:'member-item' + (idx >= cap ? ' overflow' : '')}, []);
        li.setAttribute('draggable', 'true');
        const nameSpan = makeEl('span', {class:'member-name'}, [labelOf(id)]);
        const rm = makeEl('button', {class:'member-remove', title:'Remove'}, ['Remove']);
        rm.addEventListener('click', () => {
          order.splice(idx, 1);
          refreshDatalist();
          refreshChips();
          refreshPipSelects();
          refreshFocusMainOptions();
        });
        li.addEventListener('dragstart', (e) => {
          draggingIndex = idx;
          try { e.dataTransfer.setData('text/plain', id); } catch {}
          e.dataTransfer.effectAllowed = 'move';
        });
        li.addEventListener('dragover', (e) => {
          e.preventDefault();
          const rect = li.getBoundingClientRect();
          const before = (e.clientY - rect.top) < rect.height/2;
          li.classList.toggle('drag-over-top', before);
          li.classList.toggle('drag-over-bottom', !before);
        });
        li.addEventListener('dragleave', () => {
          li.classList.remove('drag-over-top','drag-over-bottom');
        });
        li.addEventListener('drop', (e) => {
          e.preventDefault();
          li.classList.remove('drag-over-top','drag-over-bottom');
          if (draggingIndex === null || draggingIndex === idx) {
            draggingIndex = null;
            return;
          }
          const fromIdx = draggingIndex;
          const rect = li.getBoundingClientRect();
          const before = (e.clientY - rect.top) < rect.height/2;
          const item = order.splice(fromIdx, 1)[0];
          let target = before ? idx : idx + 1;
          if (fromIdx < target) target -= 1;
          if (target < 0) target = 0;
          if (target > order.length) target = order.length;
          order.splice(target, 0, item);
          draggingIndex = null;
          refreshChips();
          refreshPipSelects();
          refreshFocusMainOptions();
        });
        li.addEventListener('dragend', () => {
          const els = list.querySelectorAll('.drag-over-top,.drag-over-bottom');
          els.forEach(el => el.classList.remove('drag-over-top','drag-over-bottom'));
          draggingIndex = null;
        });
        li.appendChild(nameSpan);
        li.appendChild(rm);
        list.appendChild(li);
      });
      chips.appendChild(list);
      refreshTitleCount();
    }
    // init layout controls
    layoutSelect.value = gLayout.layout || 'grid';
    // initialize focus controls from gLayout
    Array.from(focusMode.options).forEach(o => o.selected = (o.value === (gLayout.focus_mode || '1-5')));
    refreshFocusPosOptions();
    refreshFocusMainOptions();
    updateLayoutVisibility();
    refreshDatalist();
    refreshChips();
    combo.addEventListener('input', refreshDatalist);
    combo.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); addBtn.click(); } });
    function resolveSingleSelection() {
      const q = combo.value.trim();
      if (!q) return null;
      // try exact label match first
      const byLabel = candidates('').find(id => labelOf(id).toLowerCase() === q.toLowerCase());
      if (byLabel) return byLabel;
      // fallback: if only one candidate matches substring, use it
      const cands = candidates(q);
      if (cands.length === 1) return cands[0];
      // also allow exact id match
      if (streamsMeta[q]) return q;
      return null;
    }
    addBtn.addEventListener('click', () => {
      const id = resolveSingleSelection();
      if (!id) return;
      order.push(id);
      combo.value = '';
      refreshDatalist();
      refreshChips();
      refreshFocusMainOptions();
      updateLayoutVisibility();
    });
    addAllBtn.addEventListener('click', () => {
      const q = combo.value;
      candidates(q).forEach(id => { order.push(id); });
      combo.value = '';
      refreshDatalist();
      refreshChips();
      refreshFocusMainOptions();
      updateLayoutVisibility();
    });
    removeAllBtn.addEventListener('click', () => {
      order.splice(0, order.length);
      combo.value = '';
      refreshDatalist();
      refreshChips();
      refreshFocusMainOptions();
      updateLayoutVisibility();
    });
    layoutSelect.addEventListener('change', updateLayoutVisibility);
    pipMain.addEventListener('change', () => { gLayout.pip_main = pipMain.value || null; refreshPipSelects(); });
    pipPip.addEventListener('change', () => { gLayout.pip_pip = pipPip.value || null; refreshPipSelects(); });
    const btns = makeEl('div', {class:'tile-buttons'}, []);
    const save = makeEl('button', {class:'tile-save'}, ['Save']);
    const cancel = makeEl('button', {class:'tile-cancel'}, ['Cancel']);
    btns.appendChild(save);
    btns.appendChild(cancel);
    body.appendChild(btns);
    tile.appendChild(body);

    // interactions
    edit.addEventListener('click', () => {
      body.hidden = !body.hidden;
      tile.classList.toggle('expanded', !body.hidden);
      markOpen(name, !body.hidden);
    });
    if (del) {
      del.addEventListener('click', async () => {
        if (!confirm(`Delete group ${name}?`)) return;
        await fetch(`/groups/${encodeURIComponent(name)}`, {method:'DELETE'});
        markOpen(name, false);
        tile.classList.remove('expanded');
        loadGroupsUI();
      });
    }
    save.addEventListener('click', async () => {
      const nameVal = (tile.querySelector('.tile-name').value || '').trim();
      if (!nameVal) { alert('Enter a group name'); return; }
      // preemptive duplicate check (case-insensitive), allow same-name when editing
      const wanted = nameVal.toLowerCase();
      if (wanted === 'default') { showNotification("'default' is a reserved group name"); return; }
      const exists = Object.keys(gmCurrentGroups || {}).some(g => g.toLowerCase() === wanted && g !== (name || ''));
      if (exists) { showNotification('A group with this name already exists'); return; }
      const ids = order.slice();
      // Build layout payload
      const layoutVal = layoutSelect.value;
      const payloadLayout = { layout: layoutVal };
      if (layoutVal==='grid') { payloadLayout.cols = parseInt(colsInput.value||'2',10); payloadLayout.rows = parseInt(rowsInput.value||'2',10); }
      if (layoutVal==='focus') { payloadLayout.focus_mode = focusMode.value; payloadLayout.focus_pos = focusPos.value; payloadLayout.focus_main = focusMain.value || null; }
      if (layoutVal==='pip') {
        payloadLayout.pip_main = pipMain.value || null;
        payloadLayout.pip_pip = pipPip.value || null;
        payloadLayout.pip_corner = pipCorner.value || 'bottom-right';
        payloadLayout.pip_size = parseInt(pipSize.value||'25',10);
      }
      const res = await fetch('/groups', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name: nameVal, streams: ids, layout: payloadLayout})});
      if (res.ok) {
        showNotification('Saved group');
        // If renamed, delete the old group name
        if (name && nameVal !== name) {
          try { await fetch(`/groups/${encodeURIComponent(name)}`, {method:'DELETE'}); } catch (e) {}
        }
        // Collapse after save (and handle rename)
        if (name && nameVal !== name) markOpen(name, false);
        markOpen(nameVal, false);
        tile.classList.remove('expanded');
        loadGroupsUI();
      } else {
        let msg = 'Failed to save group';
        try { const j = await res.json(); if (j && j.error) msg = j.error; } catch (e) {}
        alert(msg);
      }
    });
    cancel.addEventListener('click', () => { body.hidden = true; tile.classList.remove('expanded'); });
    // Keep count/title in sync with name edits
    const nameInputEl = nameRow.querySelector('.tile-name');
    if (nameInputEl) nameInputEl.addEventListener('input', () => { refreshTitleCount(); });
    return tile;
  }

  // initial load
  document.addEventListener('DOMContentLoaded', loadGroupsUI);

  

  // Add a new stream by posting to /streams
  addStreamBtn.addEventListener('click', () => {
    fetch('/streams', {method:'POST'})
      .then(res => res.json())
      .then(data => {
        if (data.stream_id) {
          location.reload();
        }
      });
  });

  // Card menu interactions: close on outside click and handle remove
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.card-menu')) {
      document.querySelectorAll('.card-menu .menu-dropdown').forEach(dd => dd.hidden = true);
    }
    if (e.target.classList.contains('menu-remove')) {
      const id = e.target.dataset.stream;
      if (confirm('Delete ' + id + '?')) {
        fetch('/streams/' + encodeURIComponent(id), {method:'DELETE'})
          .then(res => res.json())
          .then(data => {
            if (data.status === 'deleted') {
              location.reload();
            }
          });
      }
    }
  });

  // Socket.IO listener for layout updates if needed in future
  socket.on('streams_changed', (data) => {
    // This could be used to update the UI without reload.
    console.log('Streams changed:', data);
  });

  socket.on('ai_job_update', (data) => {
    const streamId = data && data.stream_id;
    if (!streamId) return;
    const card = document.querySelector(`.stream-card[data-stream="${streamId}"]`);
    if (!card) return;
    if (data.job) {
      aiActiveJobs.set(streamId, data.job);
      const status = (data.job.status || '').toLowerCase();
      if (['completed', 'error', 'timeout', 'cancelled'].includes(status)) {
        aiActiveJobs.delete(streamId);
      }
    }
    if (data.state) {
      if (Array.isArray(data.state.images)) {
        renderAiResults(card, data.state.images);
      }
      renderAiStatus(card, data.state, data.job || aiActiveJobs.get(streamId) || null);
    } else if (data.job) {
      renderAiStatus(card, {}, data.job);
    }
  });

  socket.on('refresh', (data) => {
    const streamId = data && data.stream_id;
    const conf = data && data.config;
    if (!streamId || !conf) return;
    const card = document.querySelector(`.stream-card[data-stream="${streamId}"]`);
    if (!card) return;
    if (Array.isArray(conf.tags)) {
      setCardTags(card, conf.tags);
      applyFiltersAndSorting();
      renderTagManager();
    }
    if (Array.isArray(data.tags)) {
      syncGlobalTags(data.tags);
    }
    if (conf.selected_image !== undefined) {
      const display = card.querySelector('.selected-image-display');
      if (display) display.textContent = conf.selected_image || 'None';
      const resultsEl = card.querySelector('.ai-results');
      if (resultsEl) {
        highlightAiSelection(resultsEl, conf.selected_image || '');
      }
    }
    if (conf.ai_state) {
      if (Array.isArray(conf.ai_state.images)) {
        renderAiResults(card, conf.ai_state.images);
      }
      renderAiStatus(card, conf.ai_state, aiActiveJobs.get(streamId) || null);
    }
  });

  // Notepad behaviour
  const notepad = document.getElementById('notepad');
  const toggleNotepadBtn = document.getElementById('toggle-notepad');
  const notepadText = document.getElementById('notepad-text');
  const saveNotesBtn = document.getElementById('save-notes');
  // Load notes from server
  fetch('/notes')
    .then(r => r.json())
    .then(({text}) => { notepadText.value = text || ''; })
    .catch(() => {});
  toggleNotepadBtn.addEventListener('click', () => {
    notepad.classList.toggle('collapsed');
  });
  saveNotesBtn.addEventListener('click', () => {
    fetch('/notes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: notepadText.value })
    })
    .then(r => r.json())
    .then(() => showNotification('Notes saved on server'))
    .catch(() => showNotification('Failed to save notes'));
  });

  // The following section reuses much of the original dashboard logic to
  // handle folder/mode selection, duration changes, image selection and
  // reload.  It has been lightly adapted to support dynamically added
  // streams.  The logic is encapsulated in functions that operate on
  // elements with a ``data-stream`` attribute.

  const aiModelCache = { models: null, promise: null };
  const aiPresetCache = { items: null, promise: null };
  const aiPresetSelects = new Set();
  let presetManagerModal = null;
  let presetManagerBackdrop = null;
  let presetManagerList = null;
  let presetManagerEmpty = null;
  const aiActiveJobs = new Map();

  function setupPresetManagerElements() {
    if (presetManagerModal) return;
    presetManagerModal = document.getElementById('ai-preset-manager');
    presetManagerBackdrop = document.getElementById('ai-preset-manager-backdrop');
    if (presetManagerModal) {
      presetManagerList = presetManagerModal.querySelector('.ai-preset-list');
      presetManagerEmpty = presetManagerModal.querySelector('.ai-preset-empty');
      const closeBtn = presetManagerModal.querySelector('.ai-preset-manager-close');
      if (closeBtn) closeBtn.addEventListener('click', closePresetManager);
    }
    if (presetManagerBackdrop) {
      presetManagerBackdrop.addEventListener('click', closePresetManager);
    }
  }

  function closePresetManager() {
    if (presetManagerModal) {
      presetManagerModal.hidden = true;
      presetManagerModal.setAttribute('aria-hidden', 'true');
    }
    if (presetManagerBackdrop) presetManagerBackdrop.hidden = true;
    document.body.classList.remove('ai-preset-manager-open');
  }

  function renderPresetManager(presets) {
    setupPresetManagerElements();
    if (!presetManagerModal || !presetManagerList) return;
    presetManagerList.innerHTML = '';
    const items = Array.isArray(presets) ? presets : [];
    if (!items.length) {
      if (presetManagerEmpty) presetManagerEmpty.hidden = false;
      return;
    }
    if (presetManagerEmpty) presetManagerEmpty.hidden = true;
    items.forEach(preset => {
      const li = document.createElement('li');
      li.className = 'ai-preset-item';
      const header = document.createElement('div');
      header.className = 'ai-preset-item-row';
      const nameEl = document.createElement('span');
      nameEl.className = 'ai-preset-item-name';
      nameEl.textContent = preset.name;
      header.appendChild(nameEl);
      const actions = document.createElement('div');
      actions.className = 'ai-preset-item-actions';
      const renameBtn = document.createElement('button');
      renameBtn.type = 'button';
      renameBtn.className = 'ai-preset-rename';
      renameBtn.textContent = 'Rename';
      renameBtn.addEventListener('click', async () => {
        const next = prompt('Rename preset', preset.name);
        if (next === null) return;
        const trimmed = (next || '').trim();
        if (!trimmed || trimmed === preset.name) {
          if (!trimmed) showNotification('Preset name cannot be empty');
          return;
        }
        try {
          await renamePresetRequest(preset.name, trimmed);
          const updated = await fetchAiPresets(true);
          refreshAllPresetSelects(updated);
          renderPresetManager(updated);
          showNotification(\`Preset "${trimmed}" saved\`);
        } catch (err) {
          showNotification(err && err.message ? err.message : 'Failed to rename preset');
        }
      });
      actions.appendChild(renameBtn);
      const deleteBtn = document.createElement('button');
      deleteBtn.type = 'button';
      deleteBtn.className = 'ai-preset-delete';
      deleteBtn.textContent = 'Delete';
      deleteBtn.addEventListener('click', async () => {
        if (!confirm(\`Delete preset "${preset.name}"?\`)) return;
        try {
          await deletePresetRequest(preset.name);
          const updated = await fetchAiPresets(true);
          refreshAllPresetSelects(updated);
          renderPresetManager(updated);
          showNotification(\`Preset "${preset.name}" deleted\`);
        } catch (err) {
          showNotification(err && err.message ? err.message : 'Failed to delete preset');
        }
      });
      actions.appendChild(deleteBtn);
      const viewDetails = document.createElement('details');
      viewDetails.className = 'ai-preset-item-details';
      const summary = document.createElement('summary');
      summary.textContent = 'View settings';
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify(preset.settings || {}, null, 2);
      viewDetails.appendChild(summary);
      viewDetails.appendChild(pre);
      header.appendChild(actions);
      li.appendChild(header);
      li.appendChild(viewDetails);
      presetManagerList.appendChild(li);
    });
  }

  async function openPresetManager() {
    setupPresetManagerElements();
    if (!presetManagerModal) return;
    presetManagerModal.hidden = false;
    presetManagerModal.setAttribute('aria-hidden', 'false');
    if (presetManagerBackdrop) presetManagerBackdrop.hidden = false;
    document.body.classList.add('ai-preset-manager-open');
    try {
      const presets = await fetchAiPresets();
      renderPresetManager(presets);
    } catch (err) {
      renderPresetManager([]);
      showNotification(err && err.message ? err.message : 'Failed to load presets');
    }
  }

  function populatePresetSelect(select, presets) {
    if (!select) return;
    const items = Array.isArray(presets) ? presets : [];
    const previous = select.value;
    select.innerHTML = '';
    if (!items.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = 'No presets saved';
      option.disabled = true;
      option.selected = true;
      select.appendChild(option);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.disabled = true;
    placeholder.textContent = 'Select a preset...';
    select.appendChild(placeholder);
    items.forEach(preset => {
      const option = document.createElement('option');
      option.value = preset.name;
      option.textContent = preset.name;
      select.appendChild(option);
    });
    if (items.some(p => p.name === previous)) {
      select.value = previous;
    } else {
      placeholder.selected = true;
    }
  }

  function refreshAllPresetSelects(presets) {
    Array.from(aiPresetSelects).forEach(select => {
      if (!select || !select.isConnected) {
        aiPresetSelects.delete(select);
        return;
      }
      populatePresetSelect(select, presets);
    });
  }

  async function fetchAiPresets(force = false) {
    if (force) {
      aiPresetCache.items = null;
    } else if (aiPresetCache.items) {
      return aiPresetCache.items;
    }
    if (!force && aiPresetCache.promise) {
      return aiPresetCache.promise;
    }
    aiPresetCache.promise = fetch('/ai/presets')
      .then(async res => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data.error) {
          throw new Error(data.error || Request failed ());
        }
        const presets = Array.isArray(data.presets) ? data.presets.slice() : [];
        presets.sort((a, b) => a.name.localeCompare(b.name));
        aiPresetCache.items = presets;
        refreshAllPresetSelects(presets);
        return presets;
      })
      .finally(() => {
        aiPresetCache.promise = null;
      });
    return aiPresetCache.promise;
  }

  async function savePresetRequest(name, settings, overwrite = false) {
    const res = await fetch('/ai/presets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, settings, overwrite })
    });
    const data = await res.json().catch(() => ({}));
    if (res.status === 409 && data.status === 'exists' && !overwrite) {
      const err = new Error(data.error || 'Preset already exists');
      err.code = 'exists';
      throw err;
    }
    if (!res.ok || data.error) {
      throw new Error(data.error || Request failed ());
    }
    aiPresetCache.items = null;
    return data.preset;
  }

  async function renamePresetRequest(oldName, newName) {
    const res = await fetch(/ai/presets/, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) {
      throw new Error(data.error || Request failed ());
    }
    aiPresetCache.items = null;
    return data.preset;
  }

  async function deletePresetRequest(name) {
    const res = await fetch(/ai/presets/, {
      method: 'DELETE'
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) {
      throw new Error(data.error || Request failed ());
    }
    aiPresetCache.items = null;
    return data;
  }


  async function fetchAiModels() {
    if (aiModelCache.models) {
      return aiModelCache.models;
    }
    if (!aiModelCache.promise) {
      aiModelCache.promise = fetch('/ai/models')
        .then(res => {
          if (!res.ok) {
            throw new Error(HTTP );
          }
          return res.json();
        })
        .then(data => Array.isArray(data.models) ? data.models : [])
        .catch(err => {
          console.error('Failed to fetch Stable Horde models', err);
          return [];
        })
        .finally(() => {
          aiModelCache.promise = null;
        });
      aiModelCache.promise.then(models => {
        aiModelCache.models = models;
      });
    }
    return aiModelCache.promise;
  }

  function highlightAiSelection(resultsEl, selectedPath) {
    if (!resultsEl) return;
    const target = selectedPath || '';
    resultsEl.dataset.selected = target;
    resultsEl.querySelectorAll('.ai-generated-item').forEach(item => {
      if (!item.dataset.path) return;
      if (item.dataset.path === target) {
        item.classList.add('is-selected');
      } else {
        item.classList.remove('is-selected');
      }
    });
  }

  function renderAiResults(card, images) {
    const resultsEl = card.querySelector('.ai-results');
    const grid = card.querySelector('.ai-generated-grid');
    const empty = card.querySelector('.ai-generated-empty');
    if (!resultsEl || !grid || !empty) return;
    grid.innerHTML = '';
    const list = Array.isArray(images) ? images : [];
    if (!list.length) {
      empty.style.display = '';
      highlightAiSelection(resultsEl, resultsEl.dataset.selected || '');
      return;
    }
    empty.style.display = 'none';
    list.forEach(img => {
      if (!img || !img.path) return;
      const item = document.createElement('div');
      item.className = 'ai-generated-item';
      item.dataset.path = img.path;
      item.dataset.persisted = img.persisted ? 'true' : 'false';
      const imageEl = document.createElement('img');
      imageEl.src = `/stream/image/${img.path}`;
      imageEl.alt = (`Generated image ${img.seed || ''}`).trim();
      item.appendChild(imageEl);
      const meta = document.createElement('div');
      meta.className = 'ai-generated-meta';
      if (img.model) {
        const modelTag = document.createElement('span');
        modelTag.className = 'ai-model-tag';
        modelTag.textContent = img.model;
        meta.appendChild(modelTag);
      }
      if (img.seed) {
        const seedTag = document.createElement('span');
        seedTag.className = 'ai-seed-tag';
        seedTag.textContent = `#${img.seed}`;
        meta.appendChild(seedTag);
      }
      item.appendChild(meta);
      grid.appendChild(item);
    });
    highlightAiSelection(resultsEl, resultsEl.dataset.selected || '');
  }

  const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  function formatUnifiedTimestamp(value, emptyPlaceholder = '\u2014') {
    if (value === null || value === undefined) return emptyPlaceholder;
    const text = String(value).trim();
    if (!text) return emptyPlaceholder;
    if (/^\d{2} [A-Za-z]{3} \d{2}:\d{2}$/.test(text)) {
      return text;
    }
    const parsed = new Date(text);
    if (!Number.isNaN(parsed.getTime())) {
      const day = String(parsed.getDate()).padStart(2, '0');
      const month = MONTH_LABELS[parsed.getMonth()] || MONTH_LABELS[0];
      const hours = String(parsed.getHours()).padStart(2, '0');
      const minutes = String(parsed.getMinutes()).padStart(2, '0');
      return `${day} ${month} ${hours}:${minutes}`;
    }
    return text;
  }

  function formatAutoTime(value) {
    return formatUnifiedTimestamp(value, '\u2014');
  }

  function updateAutoIndicators(card, state) {
    if (!card) return;
    const info = state || {};
    if (Object.prototype.hasOwnProperty.call(info, 'next_auto_trigger')) {
      const summaryNext = card.querySelector('.ai-summary-next');
      const modalNext = card.querySelector('.ai-auto-next');
      const formatted = formatAutoTime(info.next_auto_trigger);
      if (summaryNext) summaryNext.textContent = formatted;
      if (modalNext) modalNext.textContent = formatted;
    }
    if (Object.prototype.hasOwnProperty.call(info, 'last_auto_error')) {
      const errorEl = card.querySelector('.ai-auto-error');
      if (errorEl) {
        const message = info.last_auto_error;
        if (message) {
          errorEl.textContent = message;
          errorEl.hidden = false;
        } else {
          errorEl.textContent = '';
          errorEl.hidden = true;
        }
      }
    }
  }

  function renderAiStatus(card, state, job) {
    const statusEl = card.querySelector('.ai-status');
    if (!statusEl) return;
    const info = state || {};
    const jobInfo = job || {};
    const status = (jobInfo.status || info.status || 'idle').toLowerCase();
    let label = status.charAt(0).toUpperCase() + status.slice(1);
    const queuePos = jobInfo.queue_position ?? info.queue_position;
    if (queuePos !== undefined && queuePos !== null && queuePos !== '') {
      label += ` (queue ${queuePos})`;
    }
    const note = jobInfo.message || info.message;
    if (note && ['error', 'timeout', 'cancelled'].includes(status)) {
      label += ` - ${note}`;
    } else if (status === 'cancelling') {
      label += note ? ` - ${note}` : '...';
    }
    statusEl.dataset.status = status;
    statusEl.textContent = label;
    const activeStatuses = ['queued', 'accepted', 'running', 'cancelling'];
    card.querySelectorAll('.ai-generate-btn').forEach(btn => {
      btn.disabled = activeStatuses.includes(status);
    });
    const cancelButtons = card.querySelectorAll('.ai-cancel-btn');
    const showCancel = activeStatuses.includes(status);
    cancelButtons.forEach(btn => {
      btn.hidden = !showCancel;
      if (showCancel) {
        btn.disabled = status === 'cancelling';
      } else {
        btn.disabled = false;
      }
    });
    updateAutoIndicators(card, info);
  }

  function gatherPostProcessing(card) {
    const selected = [];
    card.querySelectorAll('.ai-post-proc').forEach(chk => {
      if (chk.checked && chk.value) {
        selected.push(chk.value);
      }
    });
    return selected;
  }

  function gatherLoras(card) {
    const rows = card.querySelectorAll('.ai-lora-row');
    const result = [];
    rows.forEach(row => {
      const nameInput = row.querySelector('.ai-lora-name');
      const name = nameInput ? nameInput.value.trim() : '';
      if (!name) return;
      const entry = { name };
      const modelInput = row.querySelector('.ai-lora-model');
      if (modelInput && modelInput.value.trim() !== '') {
        const parsed = parseFloat(modelInput.value);
        if (!Number.isNaN(parsed)) entry.model = parsed;
      }
      const clipInput = row.querySelector('.ai-lora-clip');
      if (clipInput && clipInput.value.trim() !== '') {
        const parsed = parseFloat(clipInput.value);
        if (!Number.isNaN(parsed)) entry.clip = parsed;
      }
      const triggerInput = row.querySelector('.ai-lora-trigger');
      if (triggerInput) {
        const trig = triggerInput.value.trim();
        if (trig) entry.inject_trigger = trig;
      }
      const versionChk = row.querySelector('.ai-lora-is-version');
      if (versionChk && versionChk.checked) {
        entry.is_version = true;
      }
      result.push(entry);
    });
    return result;
  }

  function collectAiPayload(card) {
    const payload = {};
    const promptInput = card.querySelector('.ai-prompt-input');
    const negativeInput = card.querySelector('.ai-negative-input');
    const modelSelect = card.querySelector('.ai-model-select');
    const samplerSelect = card.querySelector('.ai-sampler-select');
    const widthInput = card.querySelector('.ai-width-input');
    const heightInput = card.querySelector('.ai-height-input');
    const stepsInput = card.querySelector('.ai-steps-input');
    const cfgInput = card.querySelector('.ai-cfg-input');
    const samplesInput = card.querySelector('.ai-samples-input');
    const seedInput = card.querySelector('.ai-seed-input');
    const saveOutput = card.querySelector('.ai-save-output');
    const nsfw = card.querySelector('.ai-nsfw');
    const censor = card.querySelector('.ai-censor');
    payload.prompt = promptInput ? promptInput.value : '';
    payload.negative_prompt = negativeInput ? negativeInput.value : '';
    payload.model = modelSelect ? modelSelect.value : '';
    payload.sampler = samplerSelect ? samplerSelect.value : '';
    payload.width = widthInput ? parseInt(widthInput.value, 10) : undefined;
    payload.height = heightInput ? parseInt(heightInput.value, 10) : undefined;
    payload.steps = stepsInput ? parseInt(stepsInput.value, 10) : undefined;
    payload.cfg_scale = cfgInput ? parseFloat(cfgInput.value) : undefined;
    payload.samples = samplesInput ? parseInt(samplesInput.value, 10) : undefined;
    payload.seed = seedInput ? seedInput.value : '';
    payload.save_output = saveOutput ? saveOutput.checked : undefined;
    payload.nsfw = nsfw ? nsfw.checked : undefined;
    payload.censor_nsfw = censor ? censor.checked : undefined;
    payload.post_processing = gatherPostProcessing(card);
    payload.loras = gatherLoras(card);
    const boolPairs = [
      ['.ai-hires-fix', 'hires_fix'],
      ['.ai-karras', 'karras'],
      ['.ai-tiling', 'tiling'],
      ['.ai-transparent', 'transparent'],
      ['.ai-trusted-workers', 'trusted_workers'],
      ['.ai-validated-backends', 'validated_backends'],
      ['.ai-slow-workers', 'slow_workers'],
      ['.ai-extra-slow-workers', 'extra_slow_workers'],
      ['.ai-disable-batching', 'disable_batching'],
      ['.ai-allow-downgrade', 'allow_downgrade'],
    ];
    boolPairs.forEach(([selector, key]) => {
      const el = card.querySelector(selector);
      if (el) payload[key] = el.checked;
    });
    const styleInput = card.querySelector('.ai-style-input');
    payload.style = styleInput ? styleInput.value.trim() : '';
    const clipSkipInput = card.querySelector('.ai-clip-skip');
    if (clipSkipInput && clipSkipInput.value.trim() !== '') {
      const parsedClip = parseInt(clipSkipInput.value, 10);
      payload.clip_skip = Number.isNaN(parsedClip) ? null : parsedClip;
    } else {
      payload.clip_skip = null;
    }
    const facefixerInput = card.querySelector('.ai-facefixer');
    if (facefixerInput && facefixerInput.value.trim() !== '') {
      const parsed = parseFloat(facefixerInput.value);
      payload.facefixer_strength = Number.isNaN(parsed) ? null : parsed;
    } else {
      payload.facefixer_strength = null;
    }
    const denoiseInput = card.querySelector('.ai-denoise');
    if (denoiseInput && denoiseInput.value.trim() !== '') {
      const parsed = parseFloat(denoiseInput.value);
      payload.denoising_strength = Number.isNaN(parsed) ? null : parsed;
    } else {
      payload.denoising_strength = null;
    }
    const hiresDenoiseInput = card.querySelector('.ai-hires-denoise');
    if (hiresDenoiseInput && hiresDenoiseInput.value.trim() !== '') {
      const parsed = parseFloat(hiresDenoiseInput.value);
      payload.hires_fix_denoising_strength = Number.isNaN(parsed) ? null : parsed;
    } else {
      payload.hires_fix_denoising_strength = null;
    }
    const aiSection = card.querySelector('.ai-generator');
    if (aiSection && aiSection.dataset && 'timeout' in aiSection.dataset) {
      const stored = aiSection.dataset.timeout;
      if (stored) {
        const parsedTimeout = parseFloat(stored);
        payload.timeout = Number.isFinite(parsedTimeout) ? parsedTimeout : null;
      } else {
        payload.timeout = null;
      }
    }
    const autoModeSelect = card.querySelector('.ai-auto-mode');
    if (autoModeSelect) {
      payload.auto_generate_mode = autoModeSelect.value;
    }
    const autoIntervalInput = card.querySelector('.ai-auto-interval');
    if (autoIntervalInput) {
      const parsedInterval = parseFloat(autoIntervalInput.value);
      if (!Number.isNaN(parsedInterval)) {
        payload.auto_generate_interval_value = parsedInterval;
      }
    }
    const autoUnitSelect = card.querySelector('.ai-auto-interval-unit');
    if (autoUnitSelect) {
      payload.auto_generate_interval_unit = autoUnitSelect.value;
    }
    const autoClockInput = card.querySelector('.ai-auto-clock');
    if (autoClockInput) {
      payload.auto_generate_clock_time = autoClockInput.value;
    }
    return payload;
  }

  function setupAiControls(card, streamId) {
    const aiSection = card.querySelector('.ai-generator');
    if (!aiSection || aiSection.dataset.ready === 'true') {
      return;
    }
    aiSection.dataset.ready = 'true';
    aiSection.setAttribute('aria-hidden', 'true');
    const summaryRoot = card.querySelector('.ai-summary');
    const modalBackdrop = card.querySelector('.ai-modal-backdrop');
    const presetSelect = card.querySelector('.ai-preset-select');
    const savePresetBtn = card.querySelector('.ai-preset-save');
    const managePresetBtn = card.querySelector('.ai-preset-manage');
    const openSettingsBtn = card.querySelector('.ai-open-settings');
    const closeSettingsButtons = aiSection.querySelectorAll('.ai-close-settings, .ai-close-settings-secondary');
    const autoModeSelect = card.querySelector('.ai-auto-mode');
    const autoTimerRow = card.querySelector('.ai-auto-timer-row');
    const autoClockRow = card.querySelector('.ai-auto-clock-row');
    const autoIntervalInput = card.querySelector('.ai-auto-interval');
    const autoUnitSelect = card.querySelector('.ai-auto-interval-unit');
    const autoClockInput = card.querySelector('.ai-auto-clock');
    let applyingPreset = false;
    const updateAutoVisibility = () => {
      const modeValue = autoModeSelect ? autoModeSelect.value : 'off';
      if (autoTimerRow) autoTimerRow.hidden = modeValue !== 'timer';
      if (autoClockRow) autoClockRow.hidden = modeValue !== 'clock';
      if (autoIntervalInput) autoIntervalInput.disabled = modeValue !== 'timer';
      if (autoUnitSelect) autoUnitSelect.disabled = modeValue !== 'timer';
      if (autoClockInput) autoClockInput.disabled = modeValue !== 'clock';
    };
    updateAutoVisibility();
    if (modalBackdrop) modalBackdrop.hidden = true;

    function updateSummary() {
      if (!summaryRoot) return;
      const data = collectAiPayload(card);
      const aiGen = card.querySelector('.ai-generator');
      const width = Number(data.width);
      const height = Number(data.height);
      const stepsVal = Number(data.steps);
      const cfgVal = Number(data.cfg_scale);
      const samplesVal = Number(data.samples);
      const summaryValues = {
        model: data.model && data.model.trim() ? data.model : 'Auto',
        sampler: data.sampler ? data.sampler : 'k_euler',
        size: `${Number.isFinite(width) ? width : 512}x${Number.isFinite(height) ? height : 512}`,
        steps: Number.isFinite(stepsVal) ? stepsVal : 30,
        cfg: Number.isFinite(cfgVal) ? cfgVal : 7.5,
        samples: Number.isFinite(samplesVal) ? samplesVal : 1,
        loras: (data.loras || []).length,
        post: (data.post_processing || []).length,
        save: data.save_output ? 'Yes' : 'Temp',
      };
      let timeoutValue = null;
      if (data.timeout !== undefined && data.timeout !== null && data.timeout !== '') {
        const parsed = parseFloat(data.timeout);
        if (Number.isFinite(parsed)) timeoutValue = parsed;
      } else if (aiGen && aiGen.dataset.timeout !== undefined) {
        const stored = aiGen.dataset.timeout;
        if (stored) {
          const parsed = parseFloat(stored);
          if (Number.isFinite(parsed)) timeoutValue = parsed;
        }
      }
      if (aiGen) {
        if (Number.isFinite(timeoutValue)) {
          aiGen.dataset.timeout = String(timeoutValue);
        } else {
          aiGen.dataset.timeout = '';
        }
      }
      const timeoutLabel = Number.isFinite(timeoutValue) && timeoutValue > 0 ? timeoutValue : 'No limit';
      const mapping = {
        '.ai-summary-model': summaryValues.model,
        '.ai-summary-sampler': summaryValues.sampler,
        '.ai-summary-size': summaryValues.size,
        '.ai-summary-steps': summaryValues.steps,
        '.ai-summary-cfg': summaryValues.cfg,
        '.ai-summary-samples': summaryValues.samples,
        '.ai-summary-loras': summaryValues.loras,
        '.ai-summary-post': summaryValues.post,
        '.ai-summary-save': summaryValues.save,
        '.ai-summary-timeout': timeoutLabel,
      };
      Object.entries(mapping).forEach(([selector, value]) => {
        const el = summaryRoot.querySelector(selector);
        if (el) el.textContent = value;
      });
      const autoModeSelect = card.querySelector('.ai-auto-mode');
      let autoLabel = 'Off';
      if (autoModeSelect) {
        const modeValue = autoModeSelect.value;
        if (modeValue === 'timer') {
          const intervalInput = card.querySelector('.ai-auto-interval');
          const unitSelect = card.querySelector('.ai-auto-interval-unit');
          const unitLabel = unitSelect && unitSelect.value === 'hours' ? 'hr' : 'min';
          const intervalValue = intervalInput ? parseFloat(intervalInput.value) : NaN;
          if (!Number.isNaN(intervalValue) && intervalValue > 0) {
            const rounded = Number(intervalValue.toFixed(2));
            const display = Number.isInteger(rounded) ? rounded.toString() : rounded.toString();
            autoLabel = `Timer (${display} ${unitLabel})`;
          } else {
            autoLabel = 'Timer';
          }
        } else if (modeValue === 'clock') {
          const clockInput = card.querySelector('.ai-auto-clock');
          const timeValue = clockInput && clockInput.value ? clockInput.value : '--:--';
          autoLabel = `Clock (${timeValue})`;
        }
      }
      const autoSummary = summaryRoot.querySelector('.ai-summary-auto');
      if (autoSummary) autoSummary.textContent = autoLabel;
    }

    if (autoModeSelect) {
      autoModeSelect.addEventListener('change', e => {
        updateAutoVisibility();
        saveSettings(streamId, { ai_settings: { auto_generate_mode: e.target.value } });
        updateSummary();
      });
    }
    if (autoIntervalInput) {
      const handleIntervalChange = () => {
        const value = parseFloat(autoIntervalInput.value);
        if (Number.isNaN(value) || value <= 0) return;
        saveSettings(streamId, { ai_settings: { auto_generate_interval_value: value } });
        updateSummary();
      };
      autoIntervalInput.addEventListener('change', handleIntervalChange);
      autoIntervalInput.addEventListener('blur', handleIntervalChange);
    }
    if (autoUnitSelect) {
      autoUnitSelect.addEventListener('change', e => {
        saveSettings(streamId, { ai_settings: { auto_generate_interval_unit: e.target.value } });
        updateSummary();
      });
    }
    if (autoClockInput) {
      const handleClockChange = () => {
        saveSettings(streamId, { ai_settings: { auto_generate_clock_time: autoClockInput.value } });
        updateSummary();
      };
      autoClockInput.addEventListener('change', handleClockChange);
      autoClockInput.addEventListener('blur', handleClockChange);
    }
    aiSummaryUpdaters.set(card, updateSummary);
    updateSummary();

    if (presetSelect) {
      aiPresetSelects.add(presetSelect);
      if (aiPresetCache.items) {
        populatePresetSelect(presetSelect, aiPresetCache.items);
      } else {
        populatePresetSelect(presetSelect, []);
        fetchAiPresets()
          .then(presets => populatePresetSelect(presetSelect, presets))
          .catch(() => populatePresetSelect(presetSelect, []));
      }
      presetSelect.addEventListener('change', e => {
        const selectedName = e.target.value;
        if (!selectedName) return;
        applyPresetByName(selectedName);
      });
    } else {
      fetchAiPresets().catch(() => {});
    }

    if (savePresetBtn) {
      savePresetBtn.addEventListener('click', async () => {
        const payload = collectAiPayload(card);
        const suggested = presetSelect && presetSelect.value ? presetSelect.value : '';
        const input = prompt('Save preset as', suggested);
        if (input === null) return;
        const trimmed = (input || '').trim();
        if (!trimmed) {
          showNotification('Preset name cannot be empty');
          return;
        }
        try {
          await savePresetRequest(trimmed, payload);
        } catch (err) {
          if (err && err.code === 'exists') {
            const overwrite = confirm(\`Preset "${trimmed}" already exists. Overwrite?\`);
            if (!overwrite) return;
            try {
              await savePresetRequest(trimmed, payload, true);
            } catch (overwriteErr) {
              showNotification(overwriteErr && overwriteErr.message ? overwriteErr.message : 'Failed to save preset');
              return;
            }
          } else {
            showNotification(err && err.message ? err.message : 'Failed to save preset');
            return;
          }
        }
        try {
          const presets = await fetchAiPresets(true);
          if (presetManagerModal && !presetManagerModal.hidden) {
            renderPresetManager(presets);
          }
          if (presetSelect) {
            populatePresetSelect(presetSelect, presets);
            presetSelect.value = trimmed;
          }
          showNotification(\`Preset "${trimmed}" saved\`);
        } catch (refreshErr) {
          showNotification(refreshErr && refreshErr.message ? refreshErr.message : 'Preset saved, but list failed to refresh');
        }
      });
    } else {
      fetchAiPresets().catch(() => {});
    }

    if (managePresetBtn) {
      managePresetBtn.addEventListener('click', () => {
        openPresetManager();
      });
    }

    if (openSettingsBtn) {
      openSettingsBtn.addEventListener('click', () => {
        updateSummary();
        openAiSettings(card);
      });
    }
    if (closeSettingsButtons.length) {
      closeSettingsButtons.forEach(btn => btn.addEventListener('click', () => closeAiSettings(card)));
    }
    if (modalBackdrop) {
      modalBackdrop.addEventListener('click', () => closeAiSettings(card));
    }

    const modeSelect = card.querySelector('.mode-select');
    const promptInput = card.querySelector('.ai-prompt-input');
    if (promptInput) {
      promptInput.addEventListener('blur', e => {
        saveSettings(streamId, { ai_settings: { prompt: e.target.value } });
      });
    }
    const negativeInput = card.querySelector('.ai-negative-input');
    if (negativeInput) {
      negativeInput.addEventListener('blur', e => {
        saveSettings(streamId, { ai_settings: { negative_prompt: e.target.value } });
      });
    }
    const samplerSelect = card.querySelector('.ai-sampler-select');
    if (samplerSelect) {
      samplerSelect.addEventListener('change', e => {
        saveSettings(streamId, { ai_settings: { sampler: e.target.value } });
        updateSummary();
      });
    }
    const seedInput = card.querySelector('.ai-seed-input');
    if (seedInput) {
      seedInput.addEventListener('blur', e => {
        saveSettings(streamId, { ai_settings: { seed: e.target.value } });
      });
    }
    const modelSelect = card.querySelector('.ai-model-select');
    if (modelSelect) {
      modelSelect.addEventListener('focus', () => {
        fetchAiModels().then(models => {
          if (!models || !modelSelect) return;
          const values = new Set(Array.from(modelSelect.options).map(opt => opt.value));
          models.forEach(model => {
            const name = model && model.name ? model.name : '';
            if (!name || values.has(name)) return;
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            modelSelect.appendChild(opt);
            values.add(name);
          });
        });
      });
      modelSelect.addEventListener('change', e => {
        saveSettings(streamId, { ai_settings: { model: e.target.value } });
        updateSummary();
      });
    }
    const loraSection = card.querySelector('.ai-lora-section');
    const loraList = loraSection ? loraSection.querySelector('.ai-lora-list') : null;
    const loraSearchInput = loraSection ? loraSection.querySelector('.ai-lora-search-input') : null;
    const loraSearchBtn = loraSection ? loraSection.querySelector('.ai-lora-search-btn') : null;
    const loraResultsWrap = loraSection ? loraSection.querySelector('.ai-lora-results') : null;
    const loraResultsHeader = loraResultsWrap ? loraResultsWrap.querySelector('.ai-lora-results-header') : null;
    const loraResultsList = loraResultsWrap ? loraResultsWrap.querySelector('.ai-lora-results-list') : null;


    function bindNumeric(selector, key) {
      const input = card.querySelector(selector);
      if (!input) return;
      input.addEventListener('change', e => {
        const val = parseFloat(e.target.value);
        if (!Number.isFinite(val)) return;
        saveSettings(streamId, { ai_settings: { [key]: val } });
        updateSummary();
      });
    }

    function bindCheckboxSetting(selector, key) {
      const input = card.querySelector(selector);
      if (!input) return;
      input.addEventListener('change', e => {
        saveSettings(streamId, { ai_settings: { [key]: e.target.checked } });
        updateSummary();
      });
    }

    function bindOptionalNumber(selector, key, parser) {
      const input = card.querySelector(selector);
      if (!input) return;
      input.addEventListener('change', e => {
        const raw = (e.target.value || '').trim();
        if (!raw) {
          saveSettings(streamId, { ai_settings: { [key]: null } });
          updateSummary();
          return;
        }
        const parsed = parser(raw);
        if (Number.isNaN(parsed)) return;
        saveSettings(streamId, { ai_settings: { [key]: parsed } });
        updateSummary();
      });
    }

    function updateLoraAddState() {
      if (!loraSection) return;
      const addBtn = loraSection.querySelector('.ai-add-lora');
      if (!addBtn) return;
      const max = parseInt(loraSection.dataset.max || '0', 10) || 0;
      const current = loraSection.querySelectorAll('.ai-lora-row').length;
      addBtn.disabled = max > 0 && current >= max;
    }

    function appendEmptyLoraRow() {
      if (!loraSection || !loraList) return null;
      const max = parseInt(loraSection.dataset.max || '0', 10) || 0;
      const current = loraList.querySelectorAll('.ai-lora-row').length;
      if (max > 0 && current >= max) {
        showNotification(`Maximum of ${max} LoRAs reached`);
        return null;
      }
      const row = document.createElement('div');
      row.className = 'ai-lora-row';
      row.innerHTML = `
        <input type="text" class="ai-lora-name" placeholder="Name or CivitAI ID">
        <input type="number" class="ai-lora-model" placeholder="Model" step="0.05" min="-5" max="5">
        <input type="number" class="ai-lora-clip" placeholder="Clip" step="0.05" min="-5" max="5">
        <input type="text" class="ai-lora-trigger" placeholder="Trigger (optional)">
        <label class="toggle compact ai-lora-flag"><input type="checkbox" class="ai-lora-is-version"><span class="toggle-switch"></span><span class="toggle-label">Version ID</span></label>
        <button type="button" class="ai-lora-remove">Remove</button>
      `;
      loraList.appendChild(row);
      bindLoraRow(row);
      updateLoraAddState();
      return row;
    }

    function renderLoraResults(results, query) {
      if (!loraResultsWrap || !loraResultsList) return;
      loraResultsWrap.hidden = false;
      loraResultsList.innerHTML = '';
      if (loraResultsHeader) {
        if (results.length) {
          loraResultsHeader.textContent = query ? `Results for "${query}"` : 'LoRA results';
        } else {
          loraResultsHeader.textContent = query ? `No LoRAs found for "${query}"` : 'No LoRAs found';
        }
      }
      if (!results.length) {
        const empty = document.createElement('div');
        empty.className = 'ai-lora-result-empty';
        empty.textContent = 'Try a different search term.';
        loraResultsList.appendChild(empty);
        return;
      }
      results.forEach(result => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'ai-lora-result';
        const title = document.createElement('div');
        title.className = 'ai-lora-result-title';
        const modelName = result.modelName || 'Unnamed LoRA';
        const versionName = result.versionName || `Version ${result.versionId}`;
        title.textContent = `${modelName} - ${versionName}`;
        button.appendChild(title);
        if (Array.isArray(result.triggerWords) && result.triggerWords.length) {
          const triggers = document.createElement('div');
          triggers.className = 'ai-lora-result-triggers';
          triggers.textContent = `Triggers: ${result.triggerWords.slice(0, 3).join(', ')}`;
          button.appendChild(triggers);
        }
        button.addEventListener('click', () => {
          const row = appendEmptyLoraRow();
          if (!row) return;
          const nameInput = row.querySelector('.ai-lora-name');
          if (nameInput) {
            nameInput.value = (result.versionId || '').toString() || modelName;
          }
          const triggerInput = row.querySelector('.ai-lora-trigger');
          if (triggerInput && Array.isArray(result.triggerWords) && result.triggerWords.length) {
            triggerInput.value = result.triggerWords.slice(0, 2).join(', ');
          }
          const isVersionChk = row.querySelector('.ai-lora-is-version');
          if (isVersionChk && result.versionId) {
            isVersionChk.checked = true;
          }
          syncLoras();
          showNotification('LoRA added from search');
        });
        loraResultsList.appendChild(button);
      });
    }

    async function performLoraSearch(query) {
      if (!loraResultsWrap) return;
      const term = (query || '').trim();
      if (!term) {
        showNotification('Enter a search term for LoRAs');
        return;
      }
      loraResultsWrap.hidden = false;
      if (loraResultsHeader) loraResultsHeader.textContent = 'Searching...';
      if (loraResultsList) loraResultsList.innerHTML = '';
      try {
        const res = await fetch(`/ai/loras?q=${encodeURIComponent(term)}`);
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data.error) {
          throw new Error(data.error || `Request failed (${res.status})`);
        }
        renderLoraResults(data.results || [], term);
      } catch (err) {
        if (loraResultsHeader) loraResultsHeader.textContent = 'Search failed';
        if (loraResultsList) {
          const error = document.createElement('div');
          error.className = 'ai-lora-result-error';
          error.textContent = err && err.message ? err.message : 'Unable to load LoRAs';
          loraResultsList.appendChild(error);
        }
      }
    }


    function applyPresetSettings(preset) {
      if (!preset || typeof preset !== 'object') return;
      const aiSection = card.querySelector('.ai-generator');
      const promptInput = card.querySelector('.ai-prompt-input');
      if (promptInput) promptInput.value = preset.prompt || '';
      const negativeInput = card.querySelector('.ai-negative-input');
      if (negativeInput) negativeInput.value = preset.negative_prompt || '';
      const modelSelect = card.querySelector('.ai-model-select');
      if (modelSelect) {
        const value = preset.model || '';
        if (value && !Array.from(modelSelect.options).some(opt => opt.value === value)) {
          const opt = document.createElement('option');
          opt.value = value;
          opt.textContent = value;
          modelSelect.appendChild(opt);
        }
        modelSelect.value = value;
      }
      const samplerSelect = card.querySelector('.ai-sampler-select');
      if (samplerSelect) {
        const samplerValue = preset.sampler || 'k_euler';
        if (samplerValue && !Array.from(samplerSelect.options).some(opt => opt.value === samplerValue)) {
          const opt = document.createElement('option');
          opt.value = samplerValue;
          opt.textContent = samplerValue;
          samplerSelect.appendChild(opt);
        }
        samplerSelect.value = samplerValue;
      }
      const widthInput = card.querySelector('.ai-width-input');
      if (widthInput) {
        const width = Number(preset.width);
        widthInput.value = Number.isFinite(width) ? String(width) : '';
      }
      const heightInput = card.querySelector('.ai-height-input');
      if (heightInput) {
        const height = Number(preset.height);
        heightInput.value = Number.isFinite(height) ? String(height) : '';
      }
      const stepsInput = card.querySelector('.ai-steps-input');
      if (stepsInput) {
        const steps = Number(preset.steps);
        stepsInput.value = Number.isFinite(steps) ? String(steps) : '';
      }
      const cfgInput = card.querySelector('.ai-cfg-input');
      if (cfgInput) {
        const cfg = Number(preset.cfg_scale);
        cfgInput.value = Number.isFinite(cfg) ? String(cfg) : '';
      }
      const samplesInput = card.querySelector('.ai-samples-input');
      if (samplesInput) {
        const samples = Number(preset.samples);
        samplesInput.value = Number.isFinite(samples) ? String(samples) : '';
      }
      const seedInput = card.querySelector('.ai-seed-input');
      if (seedInput) seedInput.value = preset.seed && preset.seed !== 'random' ? preset.seed : '';
      const styleInput = card.querySelector('.ai-style-input');
      if (styleInput) styleInput.value = preset.style || '';
      const clipSkipInput = card.querySelector('.ai-clip-skip');
      if (clipSkipInput) {
        const clip = Number(preset.clip_skip);
        clipSkipInput.value = Number.isFinite(clip) ? String(clip) : '';
      }
      const facefixerInput = card.querySelector('.ai-facefixer');
      if (facefixerInput) {
        const val = Number(preset.facefixer_strength);
        facefixerInput.value = Number.isFinite(val) ? String(val) : '';
      }
      const denoiseInput = card.querySelector('.ai-denoise');
      if (denoiseInput) {
        const val = Number(preset.denoising_strength);
        denoiseInput.value = Number.isFinite(val) ? String(val) : '';
      }
      const hiresDenoiseInput = card.querySelector('.ai-hires-denoise');
      if (hiresDenoiseInput) {
        const val = Number(preset.hires_fix_denoising_strength);
        hiresDenoiseInput.value = Number.isFinite(val) ? String(val) : '';
      }
      const boolPairs = [
        ['.ai-hires-fix', 'hires_fix'],
        ['.ai-karras', 'karras'],
        ['.ai-tiling', 'tiling'],
        ['.ai-transparent', 'transparent'],
        ['.ai-trusted-workers', 'trusted_workers'],
        ['.ai-validated-backends', 'validated_backends'],
        ['.ai-slow-workers', 'slow_workers'],
        ['.ai-extra-slow-workers', 'extra_slow_workers'],
        ['.ai-disable-batching', 'disable_batching'],
        ['.ai-allow-downgrade', 'allow_downgrade']
      ];
      boolPairs.forEach(([selector, key]) => {
        const input = card.querySelector(selector);
        if (input) input.checked = !!preset[key];
      });
      const saveOutput = card.querySelector('.ai-save-output');
      if (saveOutput) saveOutput.checked = !!preset.save_output;
      const nsfw = card.querySelector('.ai-nsfw');
      if (nsfw) nsfw.checked = !!preset.nsfw;
      const censor = card.querySelector('.ai-censor');
      if (censor) censor.checked = !!preset.censor_nsfw;
      const postChecks = card.querySelectorAll('.ai-post-proc');
      if (postChecks.length) {
        const selected = new Set(Array.isArray(preset.post_processing) ? preset.post_processing : []);
        postChecks.forEach(chk => {
          chk.checked = selected.has(chk.value);
        });
      }
      if (loraList) {
        loraList.innerHTML = '';
        const loras = Array.isArray(preset.loras) ? preset.loras : [];
        loras.forEach(lora => {
          const row = appendEmptyLoraRow();
          if (!row) return;
          const nameInput = row.querySelector('.ai-lora-name');
          if (nameInput) nameInput.value = lora && lora.name ? lora.name : '';
          const modelInput = row.querySelector('.ai-lora-model');
          if (modelInput) modelInput.value = lora && lora.model !== undefined && lora.model !== null ? String(lora.model) : '';
          const clipInput = row.querySelector('.ai-lora-clip');
          if (clipInput) clipInput.value = lora && lora.clip !== undefined && lora.clip !== null ? String(lora.clip) : '';
          const triggerInput = row.querySelector('.ai-lora-trigger');
          if (triggerInput) triggerInput.value = lora && lora.inject_trigger ? lora.inject_trigger : '';
          const flagInput = row.querySelector('.ai-lora-is-version');
          if (flagInput) flagInput.checked = !!(lora && lora.is_version);
        });
        updateLoraAddState();
      }
      const autoModeSelect = card.querySelector('.ai-auto-mode');
      if (autoModeSelect) {
        const modeValue = typeof preset.auto_generate_mode === 'string' ? preset.auto_generate_mode.toLowerCase() : 'off';
        autoModeSelect.value = modeValue;
        updateAutoVisibility();
      }
      const autoIntervalInput = card.querySelector('.ai-auto-interval');
      if (autoIntervalInput) {
        const interval = Number(preset.auto_generate_interval_value);
        autoIntervalInput.value = Number.isFinite(interval) ? String(interval) : '';
      }
      const autoUnitSelect = card.querySelector('.ai-auto-interval-unit');
      if (autoUnitSelect) {
        const unit = typeof preset.auto_generate_interval_unit === 'string' ? preset.auto_generate_interval_unit.toLowerCase() : 'minutes';
        autoUnitSelect.value = unit === 'hours' ? 'hours' : 'minutes';
      }
      const autoClockInputEl = card.querySelector('.ai-auto-clock');
      if (autoClockInputEl) autoClockInputEl.value = preset.auto_generate_clock_time || '';
      if (aiSection) {
        const timeoutValue = Number(preset.timeout);
        if (Number.isFinite(timeoutValue) && timeoutValue > 0) {
          aiSection.dataset.timeout = String(timeoutValue);
        } else {
          aiSection.dataset.timeout = '';
        }
      }
      updateSummary();
    }

    async function applyPresetByName(presetName) {
      if (!presetName || applyingPreset) return;
      applyingPreset = true;
      try {
        const presets = await fetchAiPresets();
        const preset = presets.find(item => item.name === presetName);
        if (!preset) {
          showNotification(\`Preset "${presetName}" not found\`);
          populatePresetSelect(presetSelect, presets);
          return;
        }
        applyPresetSettings(preset.settings || {});
        const payload = collectAiPayload(card);
        const result = await saveSettings(streamId, { ai_settings: payload });
        if (!result || result.error) return;
        showNotification(\`Preset "${presetName}" applied\`);
      } catch (err) {
        showNotification(err && err.message ? err.message : 'Failed to apply preset');
      } finally {
        applyingPreset = false;
      }
    }

    function syncLoras() {
      if (!loraSection) return;
      const loras = gatherLoras(card);
      saveSettings(streamId, { ai_settings: { loras } });
      updateLoraAddState();
      updateSummary();
    }

    function bindLoraRow(row) {
      if (!row) return;
      const inputs = row.querySelectorAll('input');
      inputs.forEach(input => {
        const handler = () => syncLoras();
        input.addEventListener('change', handler);
        if (input.type === 'text') {
          input.addEventListener('blur', handler);
        }
      });
      const removeBtn = row.querySelector('.ai-lora-remove');
      if (removeBtn) {
        removeBtn.addEventListener('click', () => {
          row.remove();
          syncLoras();
        });
      }
    }
    bindNumeric('.ai-width-input', 'width');
    bindNumeric('.ai-height-input', 'height');
    bindNumeric('.ai-steps-input', 'steps');
    bindNumeric('.ai-cfg-input', 'cfg_scale');
    bindNumeric('.ai-samples-input', 'samples');
    [
      ['.ai-hires-fix', 'hires_fix'],
      ['.ai-karras', 'karras'],
      ['.ai-tiling', 'tiling'],
      ['.ai-transparent', 'transparent'],
      ['.ai-trusted-workers', 'trusted_workers'],
      ['.ai-validated-backends', 'validated_backends'],
      ['.ai-slow-workers', 'slow_workers'],
      ['.ai-extra-slow-workers', 'extra_slow_workers'],
      ['.ai-disable-batching', 'disable_batching'],
      ['.ai-allow-downgrade', 'allow_downgrade'],
    ].forEach(([selector, key]) => bindCheckboxSetting(selector, key));
    bindOptionalNumber('.ai-clip-skip', 'clip_skip', raw => parseInt(raw, 10));
    bindOptionalNumber('.ai-facefixer', 'facefixer_strength', raw => parseFloat(raw));
    bindOptionalNumber('.ai-denoise', 'denoising_strength', raw => parseFloat(raw));
    bindOptionalNumber('.ai-hires-denoise', 'hires_fix_denoising_strength', raw => parseFloat(raw));
    const saveOutput = card.querySelector('.ai-save-output');
    if (saveOutput) {
      saveOutput.addEventListener('change', e => {
        saveSettings(streamId, { ai_settings: { save_output: e.target.checked } });
        updateSummary();
      });
    }
    const nsfw = card.querySelector('.ai-nsfw');
    if (nsfw) {
      nsfw.addEventListener('change', e => {
        saveSettings(streamId, { ai_settings: { nsfw: e.target.checked } });
      });
    }
    const censor = card.querySelector('.ai-censor');
    if (censor) {
      censor.addEventListener('change', e => {
        saveSettings(streamId, { ai_settings: { censor_nsfw: e.target.checked } });
      });
    }
    const styleInput = card.querySelector('.ai-style-input');
    if (styleInput) {
      styleInput.addEventListener('blur', e => {
        saveSettings(streamId, { ai_settings: { style: e.target.value.trim() } });
      });
    }
    const postProcChecks = card.querySelectorAll('.ai-post-proc');
    if (postProcChecks.length) {
      postProcChecks.forEach(chk => {
        chk.addEventListener('change', () => {
          saveSettings(streamId, { ai_settings: { post_processing: gatherPostProcessing(card) } });
          updateSummary();
        });
      });
    }
    if (loraSection) {
      const addLoraBtn = loraSection.querySelector('.ai-add-lora');
      if (addLoraBtn) {
        addLoraBtn.addEventListener('click', () => {
          appendEmptyLoraRow();
          updateSummary();
        });
      }
      if (loraSearchBtn) {
        loraSearchBtn.addEventListener('click', () => {
          performLoraSearch(loraSearchInput ? loraSearchInput.value : '');
        });
      }
      if (loraSearchInput) {
        loraSearchInput.addEventListener('keydown', e => {
          if (e.key === 'Enter') {
            e.preventDefault();
            performLoraSearch(loraSearchInput.value);
          }
        });
      }
      if (loraList) {
        loraList.querySelectorAll('.ai-lora-row').forEach(row => bindLoraRow(row));
      }
      updateLoraAddState();
    }
    const generateButtons = card.querySelectorAll('.ai-generate-btn');
    const setGenerateDisabled = disabled => {
      generateButtons.forEach(btn => { btn.disabled = !!disabled; });
    };
    if (generateButtons.length) {
      const handleGenerate = async () => {
        const payload = collectAiPayload(card);
        if (!payload.prompt || !payload.prompt.trim()) {
          showNotification('Prompt is required for AI generation');
          if (promptInput) promptInput.focus();
          return;
        }
        updateSummary();
        setGenerateDisabled(true);
        renderAiStatus(card, { status: 'queued' }, { status: 'queued' });
        try {
          const res = await fetch(`/ai/generate/${encodeURIComponent(streamId)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await res.json().catch(() => ({}));
          if (!res.ok || data.error) {
            const note = data.error || `Request failed (${res.status})`;
            showNotification(note);
            renderAiStatus(card, { status: 'error', message: note }, { status: 'error', message: note });
            setGenerateDisabled(false);
            return;
          }
          if (data.job) {
            aiActiveJobs.set(streamId, data.job);
          }
          if (data.state) {
            renderAiStatus(card, data.state, data.job || null);
          }
          showNotification('AI generation queued');
        } catch (err) {
          console.error('AI generate failed', err);
          showNotification('AI generation failed to start');
          renderAiStatus(card, { status: 'error', message: 'Request failed' }, { status: 'error', message: 'Request failed' });
          setGenerateDisabled(false);
        }
      };
      generateButtons.forEach(btn => btn.addEventListener('click', handleGenerate));
    }

    const cancelButtons = card.querySelectorAll('.ai-cancel-btn');
    const setCancelDisabled = disabled => {
      cancelButtons.forEach(btn => { btn.disabled = !!disabled; });
    };
    if (cancelButtons.length) {
      const handleCancel = async event => {
        event.preventDefault();
        const currentJob = aiActiveJobs.get(streamId) || null;
        const currentStatus = (currentJob && currentJob.status) || (card.querySelector('.ai-status')?.dataset.status || 'idle');
        setCancelDisabled(true);
        try {
          const res = await fetch(`/ai/cancel/${encodeURIComponent(streamId)}`, { method: 'POST' });
          let data = null;
          try {
            data = await res.json();
          } catch (parseErr) {
            data = null;
          }
          if (!res.ok || (data && data.error)) {
            const note = data && (data.error || data.detail) || `Request failed (${res.status})`;
            showNotification(note);
            throw new Error(note);
          }
          const updatedJob = Object.assign({}, currentJob || {}, { status: 'cancelling', message: 'Cancellation requested', cancel_requested: true });
          aiActiveJobs.set(streamId, updatedJob);
          renderAiStatus(card, { status: 'cancelling', message: 'Cancellation requested' }, updatedJob);
          const label = data && data.status ? `AI ${data.status}` : 'Cancellation requested';
          showNotification(label);
          if (data && data.warning) {
            showNotification(data.warning);
          }
        } catch (err) {
          console.error('AI cancel failed', err);
          const message = err && err.message ? err.message : 'Failed to cancel job';
          showNotification(message);
          setCancelDisabled(false);
          const revertJob = currentJob ? Object.assign({}, currentJob) : null;
          renderAiStatus(card, { status: currentStatus, message: revertJob && revertJob.message ? revertJob.message : undefined }, revertJob);
        }
      };
      cancelButtons.forEach(btn => btn.addEventListener('click', handleCancel));
    }

    const resultsEl = card.querySelector('.ai-results');
    if (resultsEl) {
      resultsEl.addEventListener('click', e => {
        const target = e.target.closest('.ai-generated-item');
        if (!target || !target.dataset.path) return;
        const path = target.dataset.path;
        highlightAiSelection(resultsEl, path);
        saveSettings(streamId, { selected_image: path });
        const display = card.querySelector('.selected-image-display');
        if (display) display.textContent = path || 'None';
      });
    }
    if (modeSelect) {
      if (modeSelect.value === 'ai' && modelSelect) {
        modelSelect.dispatchEvent(new Event('focus'));
      }
      modeSelect.addEventListener('change', e => {
        if (e.target.value === 'ai' && modelSelect) {
          modelSelect.dispatchEvent(new Event('focus'));
        }
      });
    }
    renderAiStatus(card, { status: card.querySelector('.ai-status')?.dataset.status || 'idle' }, aiActiveJobs.get(streamId) || null);
    highlightAiSelection(resultsEl, resultsEl ? resultsEl.dataset.selected || '' : '');
  }

  function saveSettings(streamId, payload, opts = {}) {
    return fetch(`/settings/${encodeURIComponent(streamId)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
      if (data.status === 'success') {
        showNotification(`Updated settings for ${streamId}`);
        const card = document.querySelector(`.stream-card[data-stream="${streamId}"]`);
        if (card) {
          if (data.new_config && data.new_config.selected_image !== undefined) {
            const selectedPath = data.new_config.selected_image || "";
            const disp = card.querySelector('.selected-image-display');
            if (disp) disp.textContent = selectedPath || "None";
            const aiResults = card.querySelector('.ai-results');
            if (aiResults) {
              highlightAiSelection(aiResults, selectedPath);
            }
          }
          if (data.new_config && data.new_config.image_quality !== undefined) {
            const qualitySelect = card.querySelector('.image-quality-select');
            if (qualitySelect) {
              qualitySelect.value = (data.new_config.image_quality || 'auto');
            }
          }
          if (data.new_config && data.new_config.hide_nsfw !== undefined) {
            card.dataset.hideNsfw = data.new_config.hide_nsfw ? 'true' : 'false';
            const hideToggle = card.querySelector('.hide-nsfw-toggle');
            if (hideToggle) {
              hideToggle.checked = !!data.new_config.hide_nsfw;
            }
          }
          if (data.new_config && data.new_config.background_blur_enabled !== undefined) {
            const enabled = !!data.new_config.background_blur_enabled;
            const bgToggle = card.querySelector('.background-toggle');
            const slider = card.querySelector('.background-blur-slider');
            const wrap = card.querySelector('.background-blur-slider-wrap');
            if (bgToggle) bgToggle.checked = enabled;
            if (slider) slider.disabled = !enabled;
            if (wrap) wrap.classList.toggle('is-disabled', !enabled);
          }
          if (data.new_config && data.new_config.background_blur_amount !== undefined) {
            const amount = Number(data.new_config.background_blur_amount) || 0;
            const slider = card.querySelector('.background-blur-slider');
            const valEl = card.querySelector('.background-blur-value');
            if (slider) slider.value = String(amount);
            if (valEl) valEl.textContent = `${amount}%`;
          }
          if (data.new_config && data.new_config.tags !== undefined) {
            const tagList = Array.isArray(data.new_config.tags) ? data.new_config.tags : [];
            setCardTags(card, tagList);
            applyFiltersAndSorting();
            renderTagManager();
          }
          if (Array.isArray(data.tags)) {
            syncGlobalTags(data.tags);
          }
          const aiSettings = data.new_config && data.new_config.ai_settings;
          if (aiSettings) {
            const aiSection = card.querySelector('.ai-generator');
            if (aiSection) {
              const timeoutValue = aiSettings.timeout;
              if (timeoutValue !== undefined && timeoutValue !== null) {
                const numericTimeout = Number(timeoutValue);
                if (Number.isFinite(numericTimeout) && numericTimeout > 0) {
                  aiSection.dataset.timeout = String(numericTimeout);
                } else {
                  aiSection.dataset.timeout = '';
                }
              }
            }
            const autoWrapper = card.querySelector('.ai-auto-settings');
            if (autoWrapper) {
              const modeSelect = autoWrapper.querySelector('.ai-auto-mode');
              const intervalInput = autoWrapper.querySelector('.ai-auto-interval');
              const unitSelect = autoWrapper.querySelector('.ai-auto-interval-unit');
              const clockInput = autoWrapper.querySelector('.ai-auto-clock');
              const timerRow = autoWrapper.querySelector('.ai-auto-timer-row');
              const clockRow = autoWrapper.querySelector('.ai-auto-clock-row');
              const normalizedMode = typeof aiSettings.auto_generate_mode === 'string' ? aiSettings.auto_generate_mode.toLowerCase() : 'off';
              if (modeSelect) modeSelect.value = normalizedMode;
              if (intervalInput && aiSettings.auto_generate_interval_value !== undefined) {
                const intervalVal = Number(aiSettings.auto_generate_interval_value);
                if (Number.isFinite(intervalVal)) {
                  intervalInput.value = Number(intervalVal.toFixed(2)).toString();
                }
              }
              if (unitSelect && typeof aiSettings.auto_generate_interval_unit === 'string') {
                unitSelect.value = aiSettings.auto_generate_interval_unit.toLowerCase();
              }
              if (clockInput && aiSettings.auto_generate_clock_time !== undefined) {
                clockInput.value = aiSettings.auto_generate_clock_time || '';
              }
              const effectiveMode = normalizedMode;
              if (timerRow) timerRow.hidden = effectiveMode !== 'timer';
              if (clockRow) clockRow.hidden = effectiveMode !== 'clock';
              if (intervalInput) intervalInput.disabled = effectiveMode !== 'timer';
              if (unitSelect) unitSelect.disabled = effectiveMode !== 'timer';
              if (clockInput) clockInput.disabled = effectiveMode !== 'clock';
            }
          }
          if (data.new_config && data.new_config.ai_state) {
            updateAutoIndicators(card, data.new_config.ai_state);
          }
          const updater = aiSummaryUpdaters.get(card);
          if (updater) updater();
        }
        if (opts.onSuccess) opts.onSuccess(data, card);
      } else {
        console.error(data.error || 'Unknown error updating settings.');
        showNotification(data.error || 'Error updating settings');
        if (opts.onError) opts.onError(data);
      }
      return data;
    })
    .catch(err => {
      console.error('Error saving settings:', err);
      showNotification('Error updating settings');
      if (opts.onError) opts.onError(err);
      return null;
    })
    .finally(() => {
      if (opts.onFinally) opts.onFinally();
    });
  }

  function toggleVisibilityForMode(card, mode, streamUrl) {
    const durationDiv = card.querySelector('.duration-container');
    const shuffleChk = card.querySelector('.shuffle-chk');
    const urlDiv = card.querySelector('.stream-url-container');
    const imagePicker = card.querySelector('.image-picker');
    const ytSettingsDiv = card.querySelector('.yt-settings');
    const selectedRow = card.querySelector('.selected-image-row');
    const folderRow = card.querySelector('.folder-row');
    const qualityRow = card.querySelector('.image-quality-row');
    const nsfwRow = card.querySelector('.nsfw-row');
    const backgroundRow = card.querySelector('.background-row');
    if (durationDiv) durationDiv.style.display = (mode === 'random') ? '' : 'none';
    if (shuffleChk) shuffleChk.closest('label').style.display = (mode === 'random') ? '' : 'none';
    if (urlDiv) urlDiv.style.display = (mode === 'livestream') ? '' : 'none';
    if (imagePicker) imagePicker.style.display = (mode === 'specific') ? '' : 'none';
    const aiSection = card.querySelector('.ai-generator');
    const aiSummary = card.querySelector('.ai-summary');
    if (mode === 'ai') {
      if (aiSummary) aiSummary.style.display = '';
      if (aiSection && !card.classList.contains('ai-settings-open')) aiSection.setAttribute('aria-hidden', 'true');
    } else {
      if (aiSummary) aiSummary.style.display = 'none';
      if (aiSection) aiSection.setAttribute('aria-hidden', 'true');
      closeAiSettings(card);
    }
    const summaryUpdater = aiSummaryUpdaters.get(card);
    if (summaryUpdater) summaryUpdater();
    if (selectedRow) selectedRow.style.display = (mode === 'specific' || mode === 'ai') ? '' : 'none';
    if (folderRow) folderRow.style.display = (mode === 'livestream' || mode === 'ai') ? 'none' : '';
    if (nsfwRow) nsfwRow.style.display = (mode === 'livestream') ? 'none' : '';
    if (qualityRow) qualityRow.style.display = (mode === 'livestream') ? 'none' : '';
    if (backgroundRow) backgroundRow.style.display = (mode === 'livestream') ? 'none' : '';
    // Only show YT options if URL looks like YouTube
    const type = detectUrlType(streamUrl);
    if (ytSettingsDiv) ytSettingsDiv.style.display = (mode === 'livestream' && type === 'YouTube') ? '' : 'none';
  }

  function loadImagesFor(streamId, folder) {
    const card = document.querySelector(`.stream-card[data-stream="${streamId}"]`);
    if (!card) return;
    const imageGrid = card.querySelector('.image-picker .image-grid');
    if (!imageGrid) return;
    const selectedDisplay = card.querySelector('.selected-image-display');
    const currentSelected = selectedDisplay ? selectedDisplay.textContent.trim() : null;
    const hideNsfw = card.dataset.hideNsfw === 'true';
    const params = new URLSearchParams({ folder });
    if (hideNsfw) {
      params.set('hide_nsfw', '1');
    }
    imageGrid.innerHTML = 'Loading...';
    fetch(`/images?${params.toString()}`)
      .then(res => res.json())
      .then(imgs => {
        imageGrid.innerHTML = '';
        imgs.forEach(path => {
          const imgEl = document.createElement('img');
          imgEl.src = `/stream/image/${path}?size=thumb`;
          imgEl.title = path;
          imgEl.classList.add('picker-thumbnail');
          if (currentSelected === path) {
            imgEl.classList.add('selected-thumb');
          }
          imgEl.addEventListener('click', () => {
            saveSettings(streamId, { selected_image: path });
          });
          imageGrid.appendChild(imgEl);
        });
        if (!imgs.length) {
          imageGrid.innerHTML = '(No images found in this folder)';
        }
      })
      .catch(err => {
        console.error('Error fetching images for folder:', err);
        imageGrid.innerHTML = 'Error loading images.';
      });
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.stream-card').forEach(card => {
      const streamId = card.dataset.stream;
      if (streamId) {
        attachTagEditor(card);
      }
      setupAiControls(card, streamId);
      // Card menu toggle per card
      const menuBtn = card.querySelector('.menu-button');
      const menuDd = card.querySelector('.card-menu .menu-dropdown');
      if (menuBtn && menuDd) {
        menuBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          const isOpen = !menuDd.hidden;
          document.querySelectorAll('.card-menu .menu-dropdown').forEach(dd => dd.hidden = true);
          menuDd.hidden = isOpen;
          menuBtn.setAttribute('aria-expanded', String(!isOpen));
        });
      }
      // Folder change
      const folderSelect = card.querySelector('.folder-select');
      if (folderSelect) {
        const initialOption = folderSelect.options[folderSelect.selectedIndex];
        const initialAssigned = folderSelect.dataset.currentFolder || (initialOption ? (initialOption.dataset.filtered === 'true' ? (initialOption.title || 'all') : (initialOption.value || 'all')) : 'all');
        folderSelect.dataset.currentFolder = initialAssigned || 'all';
        if (initialOption) {
          folderSelect.title = initialOption.title || initialOption.textContent || folderSelect.value;
        } else {
          folderSelect.title = folderSelect.dataset.currentFolder || 'all';
        }
        folderSelect.addEventListener('change', e => {
          const selectEl = e.target;
          if (selectEl.value === '__filtered__') {
            const filteredOption = selectEl.options[selectEl.selectedIndex];
            selectEl.title = filteredOption ? (filteredOption.title || filteredOption.textContent || selectEl.title) : selectEl.title;
            return;
          }
          const folder = selectEl.value || 'all';
          selectEl.dataset.currentFolder = folder;
          const filteredOption = selectEl.querySelector('option[data-filtered="true"]');
          if (filteredOption) {
            filteredOption.remove();
          }
          const selectedOption = selectEl.options[selectEl.selectedIndex];
          if (selectedOption) {
            selectEl.title = selectedOption.title || selectedOption.textContent || folder;
          } else {
            selectEl.title = folder;
          }
          saveSettings(streamId, { folder, selected_image: null });
          loadImagesFor(streamId, folder);
        });
      }
      const hideNsfwToggle = card.querySelector('.hide-nsfw-toggle');
      if (hideNsfwToggle) {
        hideNsfwToggle.addEventListener('change', e => {
          const enabled = e.target.checked;
          hideNsfwToggle.disabled = true;
          saveSettings(streamId, { hide_nsfw: enabled }, {
            onSuccess: () => {
              const refreshResult = refreshFoldersForCard(card, enabled);
              if (refreshResult && typeof refreshResult.then === 'function') {
                refreshResult.finally(() => {
                  hideNsfwToggle.disabled = false;
                });
              } else {
                hideNsfwToggle.disabled = false;
              }
            },
            onError: () => {
              hideNsfwToggle.checked = !enabled;
              hideNsfwToggle.disabled = false;
            }
          });
        });
      }
      const imageQualitySelect = card.querySelector('.image-quality-select');
      if (imageQualitySelect) {
        imageQualitySelect.addEventListener('change', e => {
          const nextValue = e.target.value || 'auto';
          saveSettings(streamId, { image_quality: nextValue });
        });
      }
      const backgroundToggle = card.querySelector('.background-toggle');
      const backgroundSlider = card.querySelector('.background-blur-slider');
      const backgroundValue = card.querySelector('.background-blur-value');
      const backgroundWrap = card.querySelector('.background-blur-slider-wrap');
      if (backgroundToggle) {
        backgroundToggle.addEventListener('change', e => {
          const enabled = e.target.checked;
          if (backgroundSlider) backgroundSlider.disabled = !enabled;
          if (backgroundWrap) backgroundWrap.classList.toggle('is-disabled', !enabled);
          saveSettings(streamId, { background_blur_enabled: enabled });
        });
      }
      if (backgroundSlider) {
        backgroundSlider.addEventListener('input', e => {
          if (backgroundValue) backgroundValue.textContent = `${e.target.value}%`;
        });
        backgroundSlider.addEventListener('change', e => {
          const amount = Number(e.target.value);
          saveSettings(streamId, { background_blur_amount: amount });
        });
      }
      // Mode change
      const modeSelect = card.querySelector('.mode-select');
      const urlInput = card.querySelector('.stream-url-input');
      if (modeSelect) {
        modeSelect.addEventListener('change', e => {
          const mode = e.target.value;
          const payload = { mode };
          if (mode === 'random') {
            const durInput = card.querySelector('.duration-input');
            if (durInput) payload.duration = durInput.value;
          }
          if (mode === 'livestream') {
            if (urlInput) payload.stream_url = urlInput.value;
            const ccChk = card.querySelector('.yt-cc-chk');
            const muteChk = card.querySelector('.yt-mute-chk');
            const qualitySelect = card.querySelector('.yt-quality-select');
            if (ccChk) payload.yt_cc = ccChk.checked;
            if (muteChk) payload.yt_mute = muteChk.checked;
            if (qualitySelect) payload.yt_quality = qualitySelect.value;
            // If URL isn't YouTube, force AUTO for compatibility
            const t = detectUrlType(urlInput ? urlInput.value : '');
            if (t !== 'YouTube' && qualitySelect) {
              qualitySelect.value = 'auto';
              payload.yt_quality = 'auto';
            }
          }
          saveSettings(streamId, payload);
          toggleVisibilityForMode(card, mode, urlInput ? urlInput.value : '');
        });
      }
      // Duration change
      const durInput = card.querySelector('.duration-input');
      if (durInput) {
        durInput.addEventListener('change', e => {
          saveSettings(streamId, { duration: e.target.value });
        });
      }
      // Shuffle toggle
      const shuffleChk = card.querySelector('.shuffle-chk');
      if (shuffleChk) {
        shuffleChk.addEventListener('change', e => {
          saveSettings(streamId, { shuffle: e.target.checked });
        });
      }
      // URL helpers: type badge + background embed test + save
      function setUrlBadge(url) {
        const badge = card.querySelector('.url-type-badge');
        if (!badge) return;
        const t = detectUrlType(url);
        badge.textContent = t || '';
        badge.dataset.type = (t || '').toLowerCase();
        badge.style.visibility = t ? 'visible' : 'hidden';
      }
      let testAbort = null;
      let testTimer = null;
      async function runEmbedTest(url) {
        const statusEl = card.querySelector('.embed-status-badge');
        if (!statusEl || !url) { if (statusEl) statusEl.textContent=''; return; }
        // Debounce rapid typing
        if (testTimer) clearTimeout(testTimer);
        statusEl.textContent = 'Testing…';
        statusEl.dataset.state = 'testing';
        testTimer = setTimeout(async () => {
          if (testAbort) { try { testAbort.abort(); } catch {} }
          const ctrl = new AbortController();
          testAbort = ctrl;
          try {
            const res = await fetch('/test_embed', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ url }),
              signal: ctrl.signal
            });
            const data = await res.json();
            const state = (data.status || 'error').toLowerCase();
            statusEl.dataset.state = state;
            statusEl.textContent = (data.note || data.status || '').toUpperCase();
          } catch (e) {
            statusEl.dataset.state = 'error';
            statusEl.textContent = 'ERROR';
          }
        }, 700);
      }
      if (urlInput) {
        setUrlBadge(urlInput.value || '');
        runEmbedTest(urlInput.value || '');
        urlInput.addEventListener('input', e => {
          setUrlBadge(e.target.value);
          toggleVisibilityForMode(card, modeSelect.value, e.target.value);
          runEmbedTest(e.target.value);
        });
        urlInput.addEventListener('change', e => {
          saveSettings(streamId, { stream_url: e.target.value });
          toggleVisibilityForMode(card, modeSelect.value, e.target.value);
          // If not YouTube, force quality AUTO
          const qualitySelect = card.querySelector('.yt-quality-select');
          const t = detectUrlType(e.target.value);
          if (t !== 'YouTube' && qualitySelect) {
            qualitySelect.value = 'auto';
            saveSettings(streamId, { yt_quality: 'auto' });
          }
        });
      }
      // YT settings changes
      const ccChk = card.querySelector('.yt-cc-chk');
      const muteChk = card.querySelector('.yt-mute-chk');
      const qualitySelect = card.querySelector('.yt-quality-select');
      const renameBtn = card.querySelector('.menu-rename');
      if (ccChk) {
        ccChk.addEventListener('change', e => {
          saveSettings(streamId, { yt_cc: e.target.checked });
        });
      }
      if (muteChk) {
        muteChk.addEventListener('change', e => {
          saveSettings(streamId, { yt_mute: e.target.checked });
        });
      }
      if (qualitySelect) {
        qualitySelect.addEventListener('change', e => {
          saveSettings(streamId, { yt_quality: e.target.value });
        });
      }
      
      function slugify(s){
        return (s||'').toLowerCase().trim().replace(/[^a-z0-9]+/g,'-').replace(/-+/g,'-').replace(/^-|-$/g,'');
      }
      if (renameBtn) {
        renameBtn.addEventListener('click', () => {
          const a = card.querySelector('.card-header h2 a.stream-link');
          const current = a ? a.textContent.trim() : streamId;
          const next = prompt('Rename stream', current);
          if (next === null) return; // cancelled
          // preemptive duplicate check
          const currentSlug = slugify(current || streamId);
          const nextSlug = slugify(next || '');
          if (!nextSlug) { showNotification('Name cannot be empty'); return; }
          if (takenSlugs[nextSlug] && takenSlugs[nextSlug] !== streamId) {
            showNotification('Another stream already uses this name');
            return;
          }
          // Update map to new slug
          if (takenSlugs[currentSlug] === streamId) { delete takenSlugs[currentSlug]; }
          takenSlugs[nextSlug] = streamId;
          saveSettings(streamId, { label: next });
          if (a) {
            a.textContent = next || streamId;
            const slug = slugify(next || streamId);
            a.href = '/stream/' + encodeURIComponent(slug);
          }
          const dd = card.querySelector('.card-menu .menu-dropdown');
          if (dd) dd.hidden = true;
        });
      }
      // Reload images button
      const reloadBtn = card.querySelector('.reload-images-btn');
      if (reloadBtn) {
        reloadBtn.addEventListener('click', () => {
          const folder = folderSelect ? (folderSelect.value === '__filtered__' ? (folderSelect.dataset.currentFolder || 'all') : folderSelect.value) : 'all';
          loadImagesFor(streamId, folder);
        });
      }

      toggleVisibilityForMode(card, modeSelect ? modeSelect.value : '', urlInput ? urlInput.value : '');
    });
    originalCardOrder = Array.from(cardTagsMap.keys());
    renderTagManager();
    renderFilterChips();
    currentSortMode = (sortSelect && sortSelect.value === 'group') ? 'group' : 'default';
    applyFiltersAndSorting();
    if (tagFilterInput) {
      tagFilterInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ',') {
          e.preventDefault();
          commitFilterInput();
        }
      });
      tagFilterInput.addEventListener('blur', () => {
        commitFilterInput();
      });
    }
    if (sortSelect) {
      sortSelect.addEventListener('change', e => {
        currentSortMode = e.target.value === 'group' ? 'group' : 'default';
        applyFiltersAndSorting();
      });
    }
    if (createTagBtn) {
      createTagBtn.addEventListener('click', createGlobalTag);
    }
    if (newTagInput) {
      newTagInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
          e.preventDefault();
          createGlobalTag();
        }
      });
    }
  });
