/**
 * MSE Dashboard — shared shell
 * Handles scanline animation, sidebar state, real-time polling
 */

(function () {
    'use strict';

    // ── Scanline drift ──────────────────────────────────────────────────────
    const scanline = document.querySelector('.scanline');
    if (scanline) {
        let offset = 0;
        function drift() {
            offset = (offset + 0.15) % 100;
            scanline.style.backgroundPosition = `0 ${offset}px`;
            requestAnimationFrame(() => setTimeout(drift, 40));
        }
        drift();
    }

    // ── Sidebar active link ─────────────────────────────────────────────────
    const currentPath = window.location.pathname;
    document.querySelectorAll('.sidebar-link').forEach(function (link) {
        const href = link.getAttribute('href');
        if (href === currentPath || (currentPath.startsWith('/signal/') && href === '/')) {
            link.classList.add('active');
        } else if (href !== '/' && currentPath.startsWith(href)) {
            link.classList.add('active');
        }
    });

    // ── Mobile sidebar toggle ──────────────────────────────────────────────
    var hamburger = document.getElementById('hamburger');
    var sidebar = document.getElementById('sidebar');
    var overlay = document.getElementById('sidebar-overlay');

    function openSidebar() {
        sidebar.classList.add('open');
        hamburger.classList.add('open');
        overlay.classList.add('visible');
        hamburger.setAttribute('aria-expanded', 'true');
        document.body.style.overflow = 'hidden';
    }

    function closeSidebar() {
        sidebar.classList.remove('open');
        hamburger.classList.remove('open');
        overlay.classList.remove('visible');
        hamburger.setAttribute('aria-expanded', 'false');
        document.body.style.overflow = '';
    }

    if (hamburger) {
        hamburger.addEventListener('click', function () {
            if (sidebar.classList.contains('open')) {
                closeSidebar();
            } else {
                openSidebar();
            }
        });
    }

    if (overlay) {
        overlay.addEventListener('click', closeSidebar);
    }

    // Close sidebar on Escape key
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && sidebar && sidebar.classList.contains('open')) {
            closeSidebar();
        }
    });

    // Close sidebar when a nav link is clicked (mobile)
    document.querySelectorAll('.sidebar-link').forEach(function (link) {
        link.addEventListener('click', function () {
            if (sidebar && sidebar.classList.contains('open')) {
                closeSidebar();
            }
        });
    });

    // ── Health poll ─────────────────────────────────────────────────────────
    function pollHealth() {
        fetch('/health')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                const dots = document.querySelectorAll('.status-dot');
                dots.forEach(function (dot) {
                    if (data.ok) {
                        dot.classList.add('active');
                    } else {
                        dot.classList.remove('active');
                    }
                });
            })
            .catch(function () { /* silent */ });
    }

    setInterval(pollHealth, 30000);

    // ── Agent card click routing ────────────────────────────────────────────
    document.querySelectorAll('[data-agent-id]').forEach(function (card) {
        card.addEventListener('click', function () {
            const id = card.getAttribute('data-agent-id');
            if (id) {
                window.location.href = '/agents#' + id;
            }
        });
        card.style.cursor = 'pointer';
    });

})();
