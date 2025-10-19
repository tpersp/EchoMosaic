(() => {
  const bootstrap = window.MEDIA_MANAGER_BOOTSTRAP || {};
  const allowEdit = Boolean(bootstrap.allowEdit);
  const roots = Array.isArray(bootstrap.roots) ? bootstrap.roots : [];
  const allowedExts = Array.isArray(bootstrap.allowedExts) ? bootstrap.allowedExts : [];
  const uploadMaxMB = Number(bootstrap.uploadMaxMB || 0);
  const previewEnabled = bootstrap.previewEnabled !== false;
  const PREVIEW_INTERVAL = (() => {
    const value = Number(bootstrap.previewIntervalMs || 150);
    if (!Number.isFinite(value) || value <= 0) return 150;
    return Math.min(Math.max(value, 60), 1000);
  })();
  const previewStates = new WeakMap();

  const state = {
    currentPath: "",
    page: 1,
    pageSize: 100,
    sort: "name",
    order: "asc",
    hideNsfw: true,
  };

  const treeContainer = document.getElementById("media-tree-content");
  const breadcrumbEl = document.getElementById("media-breadcrumb");
  const subfolderContainer = document.getElementById("subfolder-list");
  const grid = document.getElementById("media-grid");
  const pagination = document.getElementById("media-pagination");
  const uploadQueue = document.getElementById("upload-queue");
  const hideNsfwToggle = document.getElementById("media-hide-nsfw");
  const sortSelect = document.getElementById("media-sort");
  const uploadInput = document.getElementById("upload-input");
  const toastContainer = document.getElementById("toast-container");
  const panel = document.querySelector(".media-panel");

  const actionCreateFolder = document.getElementById("action-create-folder");
  const actionRenameFolder = document.getElementById("action-rename-folder");
  const actionDeleteFolder = document.getElementById("action-delete-folder");
  const actionUpload = document.getElementById("action-upload");
  const refreshButton = document.getElementById("tree-refresh");

  const thumbObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const img = entry.target;
      const src = img.dataset.src;
      if (!src) return;
      img.onload = () => {
        img.dataset.loaded = "true";
      };
      img.onerror = () => {
        img.removeAttribute("data-src");
        img.dataset.loaded = "true";
      };
      img.src = src;
      thumbObserver.unobserve(img);
    });
  }, {
    rootMargin: "120px",
    threshold: 0.01,
  });

  function normalizePreviewUrls(frames) {
    if (!Array.isArray(frames)) return [];
    const urls = [];
    frames.forEach((item) => {
      const text = typeof item === "string" ? item.trim() : "";
      if (!text) return;
      if (!urls.includes(text)) {
        urls.push(text);
      }
    });
    return urls;
  }

  function attachPreview(card, img, frames) {
    if (!previewEnabled) return;
    const urls = normalizePreviewUrls(frames);
    if (urls.length <= 1) return;
    const existing = previewStates.get(card);
    if (existing) {
      existing.urls = urls;
      existing.index = 0;
      return;
    }
    const state = {
      urls,
      timer: null,
      index: 0,
      original: null,
    };
    previewStates.set(card, state);
    const start = () => startPreview(card, img);
    const stop = () => stopPreview(card, img);
    card.addEventListener("mouseenter", start);
    card.addEventListener("mouseleave", stop);
    card.addEventListener("focus", start);
    card.addEventListener("blur", stop);
    card.dataset.preview = "ready";
  }

  function startPreview(card, img) {
    const state = previewStates.get(card);
    if (!state || state.timer) return;
    if (!state.urls || state.urls.length <= 1) return;
    if (img.dataset && img.dataset.src && !img.getAttribute("src")) {
      img.src = img.dataset.src;
    }
    if (!state.original) {
      state.original = img.currentSrc || img.getAttribute("src") || img.dataset.src || "";
    }
    state.index = 0;
    const advance = () => {
      if (!state.urls.length) {
        stopPreview(card, img);
        return;
      }
      const url = state.urls[state.index % state.urls.length];
      if (url) {
        img.src = url;
      }
      state.index = (state.index + 1) % state.urls.length;
    };
    advance();
    state.timer = window.setInterval(advance, PREVIEW_INTERVAL);
    card.dataset.preview = "playing";
  }

  function stopPreview(card, img) {
    const state = previewStates.get(card);
    if (!state) return;
    if (state.timer) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
    state.index = 0;
    const fallback = state.original || (img.dataset ? img.dataset.src : "") || "";
    if (fallback) {
      img.src = fallback;
    }
    card.dataset.preview = "ready";
  }

  function stopAllPreviews() {
    document.querySelectorAll(".media-card[data-preview=\"playing\"]").forEach((cardEl) => {
      const imgEl = cardEl.querySelector(".media-thumb img");
      if (imgEl) {
        stopPreview(cardEl, imgEl);
      }
    });
  }

  function initThemeToggle() {
    const root = document.documentElement;
    const btn = document.getElementById("theme-toggle");
    function apply(theme) {
      const target = theme === "light" ? "light" : "dark";
      root.setAttribute("data-theme", target);
      if (btn) btn.textContent = target === "light" ? "\u2600" : "\u263D";
    }
    const saved = localStorage.getItem("theme") || "dark";
    apply(saved);
    if (btn) {
      btn.addEventListener("click", () => {
        const current = root.getAttribute("data-theme") === "light" ? "light" : "dark";
        const next = current === "light" ? "dark" : "light";
        apply(next);
        try {
          localStorage.setItem("theme", next);
        } catch (err) {
          console.debug("Theme toggle storage failed", err);
        }
      });
    }
  }

  function parseVirtual(path) {
    if (!path) return { alias: null, segments: [] };
    const idx = path.indexOf(":/");
    if (idx === -1) {
      const parts = path.split("/").filter(Boolean);
      return { alias: parts.shift() || null, segments: parts };
    }
    const alias = path.slice(0, idx);
    const remainder = path.slice(idx + 2).replace(/^\/+/, "");
    const segments = remainder ? remainder.split("/").filter(Boolean) : [];
    return { alias, segments };
  }

  function buildVirtual(alias, segments) {
    if (!alias) return "";
    const rest = segments && segments.length ? `/${segments.join("/")}` : "";
    return `${alias}:/${rest}`.replace(/\/{2,}/g, "/");
  }

  function parentPath(path) {
    const info = parseVirtual(path);
    if (!info.alias || info.segments.length === 0) return "";
    const segments = info.segments.slice(0, -1);
    return buildVirtual(info.alias, segments);
  }

  function leafName(path) {
    const info = parseVirtual(path);
    if (info.segments.length === 0) return info.alias || "";
    return info.segments[info.segments.length - 1];
  }

  function aliasLabel(alias) {
    const root = roots.find((r) => r.alias === alias);
    return root ? root.display_name : alias;
  }

  function prettyPath(path) {
    const info = parseVirtual(path);
    if (!info.alias) return [];
    const crumbs = [{ label: aliasLabel(info.alias), path: buildVirtual(info.alias, []) }];
    let accum = [];
    info.segments.forEach((segment) => {
      accum.push(segment);
      crumbs.push({ label: segment, path: buildVirtual(info.alias, accum.slice()) });
    });
    return crumbs;
  }

  function showToast(message, type = "info") {
    if (!toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    const text = document.createElement("span");
    text.textContent = message;
    const dismiss = document.createElement("button");
    dismiss.type = "button";
    dismiss.setAttribute("aria-label", "Dismiss");
    dismiss.textContent = "\u2715";
    dismiss.addEventListener("click", () => {
      toast.remove();
    });
    toast.append(text, dismiss);
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.remove();
    }, 4000);
  }

  async function fetchDirectory(path, opts = {}) {
    const params = new URLSearchParams();
    if (path) params.set("path", path);
    params.set("page", opts.page || state.page || 1);
    params.set("page_size", opts.pageSize || state.pageSize || 100);
    params.set("sort", state.sort);
    params.set("order", state.order);
    params.set("hide_nsfw", state.hideNsfw ? "1" : "0");
    try {
      const res = await fetch(`/api/media/list?${params.toString()}`);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const error = data && data.error ? data.error : `Request failed (${res.status})`;
        throw new Error(error);
      }
      return res.json();
    } catch (err) {
      throw err;
    }
  }

  function clearElement(el) {
    if (!el) return;
    while (el.firstChild) {
      el.removeChild(el.firstChild);
    }
  }

  async function ensureTreeChildren(detailsEl, path) {
    if (!detailsEl || !path) return;
    const container = detailsEl.querySelector(".tree-children");
    if (!container) return;
    detailsEl.dataset.path = path;
    clearElement(container);
    try {
      const data = await fetchDirectory(path, { pageSize: 1 });
      (data.folders || []).forEach((folder) => {
        const node = createTreeNode(folder.path, folder.name);
        container.appendChild(node);
      });
      detailsEl.dataset.loaded = "true";
      highlightTreeSelection();
    } catch (err) {
      container.textContent = "Unable to load folders";
      console.error(err);
    }
  }

  function createTreeNode(path, label) {
    const info = parseVirtual(path);
    const isRoot = info.segments.length === 0;
    const details = document.createElement("details");
    details.className = "tree-node";
    details.dataset.path = path;
    const summary = document.createElement("summary");
    summary.textContent = isRoot ? aliasLabel(info.alias) : label;
    summary.addEventListener("click", (event) => {
      event.stopPropagation();
      if (state.currentPath !== path) {
        selectFolder(path);
      }
    });
    details.addEventListener("toggle", () => {
      if (details.open) {
        ensureTreeChildren(details, path);
      }
    });
    const children = document.createElement("div");
    children.className = "tree-children";
    details.append(summary, children);
    return details;
  }

  function buildTree() {
    if (!treeContainer) return;
    clearElement(treeContainer);
    roots.forEach((root) => {
      const node = createTreeNode(root.path, root.display_name || root.alias);
      treeContainer.appendChild(node);
    });
  }

  function highlightTreeSelection() {
    const nodes = treeContainer ? treeContainer.querySelectorAll(".tree-node") : [];
    nodes.forEach((node) => {
      if (node.dataset.path === state.currentPath) {
        node.classList.add("active");
        let parent = node.parentElement;
        while (parent && parent.classList && parent.classList.contains("tree-node")) {
          parent.classList.add("active");
          parent.open = true;
          parent = parent.parentElement;
        }
      } else {
        node.classList.remove("active");
      }
    });
  }

  function renderBreadcrumb(path) {
    if (!breadcrumbEl) return;
    clearElement(breadcrumbEl);
    const crumbs = prettyPath(path);
    if (!crumbs.length) {
      const span = document.createElement("span");
      span.textContent = "All roots";
      breadcrumbEl.appendChild(span);
      return;
    }
    crumbs.forEach((crumb, idx) => {
      if (idx > 0) {
        const sep = document.createElement("span");
        sep.textContent = "/";
        breadcrumbEl.appendChild(sep);
      }
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = crumb.label;
      button.addEventListener("click", () => {
        selectFolder(crumb.path);
      });
      breadcrumbEl.appendChild(button);
    });
  }

  function renderSubfolders(folders) {
    if (!subfolderContainer) return;
    clearElement(subfolderContainer);
    if (!folders || !folders.length) {
      const hint = document.createElement("div");
      hint.className = "subfolder-meta";
      hint.textContent = "No subfolders in this directory.";
      subfolderContainer.appendChild(hint);
      return;
    }
    folders.forEach((folder) => {
      const card = document.createElement("div");
      card.className = "subfolder-card";
      const header = document.createElement("div");
      header.className = "subfolder-header";
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = folder.name;
      button.addEventListener("click", () => selectFolder(folder.path));
      button.title = folder.path;
      header.appendChild(button);
      if (allowEdit) {
        const actions = document.createElement("div");
        actions.className = "media-card-actions";
        const rename = document.createElement("button");
        rename.type = "button";
        rename.textContent = "Rename";
        rename.addEventListener("click", (event) => {
          event.stopPropagation();
          renameEntry(folder.path);
        });
        const del = document.createElement("button");
        del.type = "button";
        del.textContent = "Delete";
        del.addEventListener("click", (event) => {
          event.stopPropagation();
          deleteEntry(folder.path);
        });
        actions.append(rename, del);
        header.appendChild(actions);
      }
      const meta = document.createElement("div");
      meta.className = "subfolder-meta";
      const count = folder.count || 0;
      meta.textContent = `${count} item${count === 1 ? "" : "s"}`;
      card.append(header, meta);
      subfolderContainer.appendChild(card);
    });
  }

  function createFileCard(file) {
    const card = document.createElement("div");
    card.className = "media-card";
    card.dataset.path = file.path;

    const thumb = document.createElement("div");
    thumb.className = "media-thumb";
    const img = document.createElement("img");
    if (file.thumbUrl) {
      img.dataset.src = file.thumbUrl;
      img.dataset.thumb = file.thumbUrl;
      thumbObserver.observe(img);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "placeholder";
      placeholder.textContent = file.ext ? file.ext.toUpperCase() : "FILE";
      thumb.appendChild(placeholder);
    }
    img.alt = file.name || "";
    thumb.appendChild(img);
    card.appendChild(thumb);

    if (allowEdit) {
      const menuTrigger = document.createElement("button");
      menuTrigger.type = "button";
      menuTrigger.className = "menu-trigger";
      menuTrigger.textContent = "\u22EE";
      const menu = document.createElement("div");
      menu.className = "card-menu";
      const rename = document.createElement("button");
      rename.type = "button";
      rename.textContent = "Rename";
      rename.addEventListener("click", (event) => {
        event.stopPropagation();
        menu.classList.remove("open");
        renameEntry(file.path);
      });
      const del = document.createElement("button");
      del.type = "button";
      del.textContent = "Delete";
      del.addEventListener("click", (event) => {
        event.stopPropagation();
        menu.classList.remove("open");
        deleteEntry(file.path);
      });
      menu.append(rename, del);
      menuTrigger.addEventListener("click", (event) => {
        event.stopPropagation();
        closeAllMenus();
        menu.classList.toggle("open");
      });
      thumb.append(menuTrigger, menu);
    }

    const body = document.createElement("div");
    body.className = "media-card-body";
    const name = document.createElement("div");
    name.className = "media-card-name";
    name.textContent = file.name;
    body.appendChild(name);

    const meta = document.createElement("div");
    meta.className = "media-card-meta";
    const size = document.createElement("span");
    size.textContent = file.size_text || "";
    const mtime = document.createElement("span");
    if (file.mtime) {
      const date = new Date(file.mtime * 1000);
      mtime.textContent = date.toLocaleString();
    }
    meta.append(size, mtime);
    body.appendChild(meta);
    card.appendChild(body);

    if (Array.isArray(file.frames) && file.frames.length > 1) {
      attachPreview(card, img, file.frames);
    }

    return card;
  }

  function closeAllMenus() {
    document.querySelectorAll(".card-menu.open").forEach((menu) => {
      menu.classList.remove("open");
    });
  }

  function renderFiles(files) {
    if (!grid) return;
    stopAllPreviews();
    clearElement(grid);
    if (!files || !files.length) {
      const message = document.createElement("div");
      message.className = "subfolder-meta";
      message.textContent = "No files found in this folder.";
      grid.appendChild(message);
      return;
    }
    files.forEach((file) => {
      grid.appendChild(createFileCard(file));
    });
  }

  function renderPagination(page, totalPages) {
    if (!pagination) return;
    clearElement(pagination);
    if (totalPages <= 1) return;
    const info = document.createElement("span");
    info.textContent = `Page ${page} of ${totalPages}`;
    const prev = document.createElement("button");
    prev.type = "button";
    prev.textContent = "Prev";
    prev.disabled = page <= 1;
    prev.addEventListener("click", () => {
      if (page > 1) {
        state.page = page - 1;
        loadContent();
      }
    });
    const next = document.createElement("button");
    next.type = "button";
    next.textContent = "Next";
    next.disabled = page >= totalPages;
    next.addEventListener("click", () => {
      if (page < totalPages) {
        state.page = page + 1;
        loadContent();
      }
    });
    pagination.append(prev, info, next);
  }

  async function loadContent() {
    try {
      const data = await fetchDirectory(state.currentPath, { page: state.page, pageSize: state.pageSize });
      state.page = data.page || 1;
      renderBreadcrumb(data.path || state.currentPath);
      renderSubfolders(data.folders || []);
      renderFiles(data.files || []);
      renderPagination(data.page || 1, data.total_pages || 1);
      highlightTreeSelection();
    } catch (err) {
      showToast(err.message || "Failed to load folder", "error");
    }
  }

  async function selectFolder(path) {
    state.currentPath = path || "";
    state.page = 1;
    await loadContent();
    await openTreeForPath(state.currentPath);
    updateToolbarState();
  }

  async function openTreeForPath(path) {
    if (!treeContainer || !path) return;
    const info = parseVirtual(path);
    if (!info.alias) return;
    const rootPath = buildVirtual(info.alias, []);
    const rootNode = treeContainer.querySelector(`.tree-node[data-path="${rootPath}"]`);
    if (rootNode) {
      rootNode.open = true;
      await ensureTreeChildren(rootNode, rootPath);
    }
    const segments = [];
    for (const segment of info.segments) {
      segments.push(segment);
      const segmentPath = buildVirtual(info.alias, segments);
      const details = treeContainer.querySelector(`.tree-node[data-path="${segmentPath}"]`);
      if (details) {
        details.open = true;
        await ensureTreeChildren(details, segmentPath);
      }
    }
    highlightTreeSelection();
  }

  async function renameEntry(path) {
    const current = leafName(path);
    const newName = prompt("Rename to:", current);
    if (!newName || newName === current) return;
    try {
      const res = await fetch("/api/media/rename", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, new_name: newName }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || "Rename failed");
      }
      showToast("Rename successful", "success");
      if (state.currentPath === path) {
        state.currentPath = data.path || path;
      }
      refreshTree(parentPath(path));
      loadContent();
    } catch (err) {
      showToast(err.message || "Unable to rename", "error");
    }
  }

  async function deleteEntry(path) {
    const targetLeaf = leafName(path);
    const confirmDelete = confirm(`Delete "${targetLeaf}"? This cannot be undone.`);
    if (!confirmDelete) return;
    try {
      const res = await fetch("/api/media/delete", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || "Delete failed");
      }
      showToast("Deleted successfully", "success");
      const parent = parentPath(path);
      if (state.currentPath === path) {
        state.currentPath = parent;
      }
      refreshTree(parent);
      loadContent();
    } catch (err) {
      showToast(err.message || "Unable to delete", "error");
    }
  }

  function refreshTree(path) {
    if (!treeContainer) return;
    if (!path) {
      buildTree();
      openTreeForPath(state.currentPath);
      return;
    }
    const details = treeContainer.querySelector(`.tree-node[data-path="${path}"]`);
    if (details) {
      details.dataset.loaded = "false";
      const children = details.querySelector(".tree-children");
      if (children) clearElement(children);
      if (details.open) ensureTreeChildren(details, path);
    } else {
      buildTree();
      openTreeForPath(state.currentPath);
    }
  }

  function validateCanModify() {
    if (!allowEdit) {
      showToast("Editing is disabled in configuration.", "error");
      return false;
    }
    return true;
  }

  async function createFolder() {
    if (!validateCanModify()) return;
    const base = state.currentPath;
    const info = parseVirtual(base);
    if (!info.alias) {
      showToast("Select a media root before creating a folder.", "error");
      return;
    }
    const name = prompt("Folder name:");
    if (!name) return;
    try {
      const res = await fetch("/api/media/create_folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: base || `${info.alias}:/`, name }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || "Create folder failed");
      }
      showToast("Folder created", "success");
      refreshTree(base || buildVirtual(info.alias, []));
      loadContent();
    } catch (err) {
      showToast(err.message || "Unable to create folder", "error");
    }
  }

  async function renameCurrentFolder() {
    if (!validateCanModify()) return;
    const current = state.currentPath;
    const info = parseVirtual(current);
    if (!info.alias || info.segments.length === 0) {
      showToast("Cannot rename root folders.", "error");
      return;
    }
    renameEntry(current);
  }

  async function deleteCurrentFolder() {
    if (!validateCanModify()) return;
    const current = state.currentPath;
    const info = parseVirtual(current);
    if (!info.alias || info.segments.length === 0) {
      showToast("Cannot delete root folders.", "error");
      return;
    }
    deleteEntry(current);
  }

  function ensureUploadTarget() {
    const info = parseVirtual(state.currentPath);
    if (!info.alias) {
      showToast("Select a folder before uploading.", "error");
      return false;
    }
    return true;
  }

  function prepareUploadRow(file) {
    if (!uploadQueue) return null;
    uploadQueue.hidden = false;
    const row = document.createElement("div");
    row.className = "upload-row";
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = file.name;
    const progress = document.createElement("div");
    progress.className = "progress-bar";
    const span = document.createElement("span");
    progress.appendChild(span);
    row.append(title, progress);
    uploadQueue.appendChild(row);
    return { row, bar: span };
  }

  function finalizeUploadRow(rowInfo, success) {
    if (!rowInfo) return;
    rowInfo.row.style.opacity = success ? "0.75" : "1";
    if (!success) {
      rowInfo.row.style.border = "1px solid #f44336";
    }
    setTimeout(() => {
      rowInfo.row.remove();
      if (uploadQueue && uploadQueue.children.length === 0) {
        uploadQueue.hidden = true;
      }
    }, 1500);
  }

  function uploadFiles(files) {
    if (!validateCanModify()) return;
    if (!ensureUploadTarget()) return;
    const items = Array.from(files || []);
    if (!items.length) return;
    (async () => {
      for (const file of items) {
        const row = prepareUploadRow(file);
        try {
          await uploadSingle(file, row);
          finalizeUploadRow(row, true);
          showToast(`Uploaded ${file.name}`, "success");
        } catch (err) {
          finalizeUploadRow(row, false);
          showToast(err.message || `Failed to upload ${file.name}`, "error");
          break;
        }
      }
      refreshTree(state.currentPath);
      loadContent();
    })();
  }

  function uploadSingle(file, rowInfo) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/media/upload");
      xhr.responseType = "json";
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(xhr.response);
        } else {
          const msg = xhr.response && xhr.response.error ? xhr.response.error : `Upload failed (${xhr.status})`;
          reject(new Error(msg));
        }
      };
      xhr.onerror = () => reject(new Error("Upload failed"));
      if (rowInfo && rowInfo.bar) {
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const pct = Math.round((event.loaded / event.total) * 100);
            rowInfo.bar.style.width = `${pct}%`;
          }
        };
      }
      const form = new FormData();
      form.append("path", state.currentPath);
      form.append("files", file);
      xhr.send(form);
    });
  }

  function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    panel.classList.remove("drag-over");
    if (event.dataTransfer && event.dataTransfer.files) {
      uploadFiles(event.dataTransfer.files);
    }
  }

  function bindEvents() {
    document.addEventListener("click", closeAllMenus);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeAllMenus();
      if (event.key === "Escape") stopAllPreviews();
    });
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState !== "visible") {
        stopAllPreviews();
      }
    });
    window.addEventListener("blur", stopAllPreviews);
    if (hideNsfwToggle) {
      hideNsfwToggle.addEventListener("change", () => {
        state.hideNsfw = hideNsfwToggle.checked;
        state.page = 1;
        loadContent();
      });
    }
    if (sortSelect) {
      sortSelect.addEventListener("change", () => {
        const [sort, order] = sortSelect.value.split("|");
        state.sort = sort;
        state.order = order;
        state.page = 1;
        loadContent();
      });
    }
    if (allowEdit && uploadInput) {
      uploadInput.addEventListener("change", () => {
        uploadFiles(uploadInput.files);
        uploadInput.value = "";
      });
    }
    if (allowEdit && actionCreateFolder) {
      actionCreateFolder.addEventListener("click", createFolder);
    }
    if (allowEdit && actionRenameFolder) {
      actionRenameFolder.addEventListener("click", renameCurrentFolder);
    }
    if (allowEdit && actionDeleteFolder) {
      actionDeleteFolder.addEventListener("click", deleteCurrentFolder);
    }
    if (allowEdit && actionUpload) {
      actionUpload.addEventListener("click", () => {
        if (!ensureUploadTarget()) return;
        if (!uploadInput) return;
        if (allowedExts.length) {
          uploadInput.setAttribute("accept", allowedExts.join(","));
        }
        uploadInput.click();
      });
    }
    if (refreshButton) {
      refreshButton.addEventListener("click", () => {
        refreshTree("");
        loadContent();
      });
    }
    if (panel) {
      panel.addEventListener("dragenter", (event) => {
        event.preventDefault();
        event.stopPropagation();
        panel.classList.add("drag-over");
      });
      panel.addEventListener("dragover", (event) => {
        event.preventDefault();
        panel.classList.add("drag-over");
      });
      panel.addEventListener("dragleave", (event) => {
        event.preventDefault();
        if (event.target === panel) {
          panel.classList.remove("drag-over");
        }
      });
      panel.addEventListener("drop", handleDrop);
    }
    document.addEventListener("dragover", (event) => event.preventDefault());
    document.addEventListener("drop", (event) => {
      if (event.target === document) {
        event.preventDefault();
      }
    });
  }

  function updateToolbarState() {
    if (!allowEdit) return;
    const info = parseVirtual(state.currentPath);
    const disableRoot = !info.alias || info.segments.length === 0;
    if (actionRenameFolder) actionRenameFolder.disabled = disableRoot;
    if (actionDeleteFolder) actionDeleteFolder.disabled = disableRoot;
  }

  async function bootstrapSelectInitial() {
    if (roots.length) {
      await selectFolder(roots[0].path);
    } else {
      await loadContent();
    }
    updateToolbarState();
  }

  function displayUploadLimit() {
    if (!uploadMaxMB || !toastContainer) return;
    const notice = document.createElement("div");
    notice.className = "subfolder-meta";
    notice.textContent = `Max upload size: ${uploadMaxMB} MB.`;
    if (uploadQueue && uploadQueue.parentElement) {
      uploadQueue.parentElement.insertBefore(notice, uploadQueue);
    }
  }

  function init() {
    initThemeToggle();
    buildTree();
    bindEvents();
    displayUploadLimit();
    bootstrapSelectInitial();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
