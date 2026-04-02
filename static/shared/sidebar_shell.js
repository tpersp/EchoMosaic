(() => {
  const STORAGE_KEY = "sidebar-collapsed";
  const root = document.documentElement;

  function applyRootState(collapsed) {
    if (!root) {
      return;
    }
    root.classList.toggle("sidebar-collapsed", collapsed);
  }

  function applySidebarState(button, collapsed) {
    applyRootState(collapsed);
    if (!document.body) {
      return;
    }
    document.body.classList.toggle("sidebar-collapsed", collapsed);
    if (!button) {
      return;
    }
    const label = collapsed ? "Expand sidebar" : "Collapse sidebar";
    button.setAttribute("aria-label", label);
    button.setAttribute("title", label);
    button.setAttribute("aria-pressed", collapsed ? "true" : "false");
    const icon = button.querySelector(".sidebar-shell-toggle-icon");
    if (icon) {
      icon.textContent = collapsed ? "\u00BB" : "\u00AB";
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("sidebar-toggle");
    if (!button) {
      return;
    }
    let collapsed = false;
    try {
      collapsed = window.localStorage.getItem(STORAGE_KEY) === "1";
    } catch (err) {
      collapsed = false;
    }
    applySidebarState(button, collapsed);
    button.addEventListener("click", () => {
      const next = !document.body.classList.contains("sidebar-collapsed");
      applySidebarState(button, next);
      try {
        window.localStorage.setItem(STORAGE_KEY, next ? "1" : "0");
      } catch (err) {
        /* ignore localStorage failures */
      }
    });
  });
})();
