/**
 * MSE Agents Page
 * Handles search, filter, sort, and agent card interactions
 */

(function () {
    'use strict';

    // ── Hash routing — scroll to agent card if #id in URL ──────────────────
    function handleHash() {
        var hash = window.location.hash;
        if (hash) {
            var card = document.querySelector('[data-agent-id="' + hash.slice(1) + '"]');
            if (card) {
                card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                card.style.boxShadow = '0 0 0 2px var(--plasma)';
                card.style.borderColor = 'var(--plasma)';
                setTimeout(function () {
                    card.style.boxShadow = '';
                    card.style.borderColor = '';
                }, 2000);
            }
        }
    }

    handleHash();
    window.addEventListener('hashchange', handleHash);

    // ── Agent card click — copy agent ID to hash ────────────────────────────
    document.querySelectorAll('.agent-card').forEach(function (card) {
        card.addEventListener('click', function () {
            var id = card.getAttribute('data-agent-id');
            if (id) {
                window.location.hash = id;
            }
        });
    });

    // ── Poll agent states every 30s ─────────────────────────────────────────
    function refreshAgentStates() {
        fetch('/api/agents/summary')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                data.agents.forEach(function (agent) {
                    var card = document.querySelector('[data-agent-id="' + agent.id + '"]');
                    if (!card) return;

                    // Update status dot
                    var dot = card.querySelector('.agent-status');
                    if (dot) {
                        dot.className = 'agent-status ' + agent.status;
                    }

                    // Update accuracy
                    var accEl = card.querySelector('.agent-metric .m-value');
                    if (accEl) {
                        accEl.textContent = (agent.accuracy * 100).toFixed(1) + '%';
                        accEl.className = 'm-value ' +
                            (agent.accuracy >= 0.7 ? 'high' : agent.accuracy >= 0.6 ? 'mid' : 'low');
                    }

                    // Update weight bar
                    var bar = card.querySelector('.weight-bar-fill');
                    if (bar) {
                        bar.style.width = (agent.weight * 1000).toFixed(0) + '%';
                    }
                    var barVal = card.querySelector('.weight-bar-value');
                    if (barVal) {
                        barVal.textContent = agent.weight.toFixed(4);
                    }

                    // Update signal pill
                    var pill = card.querySelector('.signal-pill');
                    if (pill) {
                        pill.textContent = agent.current_signal;
                        pill.className = 'signal-pill ' + agent.current_signal;
                    }
                });
            })
            .catch(function () { /* silent */ });
    }

    setInterval(refreshAgentStates, 30000);

})();
