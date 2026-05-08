/**
 * MSE Signal Detail Page
 * Vote table interactions, row highlighting, confidence sparkline
 */

(function () {
    'use strict';

    // ── Row highlight on hover ──────────────────────────────────────────────
    var rows = document.querySelectorAll('.votes-table tbody tr');
    rows.forEach(function (row) {
        row.addEventListener('mouseenter', function () {
            row.style.background = 'var(--bg-highlight)';
        });
        row.addEventListener('mouseleave', function () {
            row.style.background = '';
        });
    });

    // ── Vote filter ─────────────────────────────────────────────────────────
    var filterActive = null;

    function filterVotes(direction) {
        if (filterActive === direction) {
            filterActive = null;
        } else {
            filterActive = direction;
        }

        rows.forEach(function (row) {
            var pill = row.querySelector('.signal-pill');
            if (!pill) return;
            if (!filterActive || pill.textContent.trim() === filterActive) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }

    // Attach filter to consensus legend items
    var legendItems = document.querySelectorAll('.consensus-legend-item');
    legendItems.forEach(function (item) {
        item.style.cursor = 'pointer';
        item.addEventListener('click', function () {
            var text = item.textContent.trim().toLowerCase();
            if (text.indexOf('bullish') === 0) filterVotes('bullish');
            else if (text.indexOf('bearish') === 0) filterVotes('bearish');
            else if (text.indexOf('neutral') === 0) filterVotes('neutral');

            // Visual toggle
            legendItems.forEach(function (li) { li.style.opacity = '0.6'; });
            if (filterActive) {
                item.style.opacity = '1';
            }
        });
    });

    // Click on consensus segments also filters
    var segments = document.querySelectorAll('.consensus-segment');
    segments.forEach(function (seg) {
        seg.style.cursor = 'pointer';
        seg.addEventListener('click', function () {
            if (seg.classList.contains('bullish')) filterVotes('bullish');
            else if (seg.classList.contains('bearish')) filterVotes('bearish');
            else if (seg.classList.contains('neutral')) filterVotes('neutral');
        });
    });

    // ── Signal detail poll (refresh agent votes every 60s) ─────────────────
    var signalId = window.location.pathname.split('/').pop();
    if (signalId && !isNaN(signalId)) {
        setInterval(function () {
            fetch('/api/signals/' + signalId + '/agents')
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (!data.votes) return;
                    // Update tally counts if displayed
                    var tallyInfo = document.querySelector('.consensus-legend');
                    if (tallyInfo && data.tally) {
                        var counts = tallyInfo.querySelectorAll('.legend-count');
                        if (counts.length >= 3) {
                            counts[0].textContent = data.tally.bullish;
                            counts[1].textContent = data.tally.neutral;
                            counts[2].textContent = data.tally.bearish;
                        }
                    }
                })
                .catch(function () { /* silent */ });
        }, 60000);
    }

})();
