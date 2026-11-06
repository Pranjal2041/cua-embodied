/* Computer-Use Agents Are Embodied Generalists project page */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

/* What We Think? Moving question chips. Clicking one opens its answer card below. */
(function questions() {
  const rows = [$('#marquee-track'), $('#marquee-track-2')];
  const chips = $$('.chip', rows[0]);
  // With many questions, alternate them between two rows that scroll in opposite directions.
  if (chips.length > 8) {
    chips.forEach((c, i) => { if (i % 2) rows[1].appendChild(c); });
    $('#marquee-2').hidden = false;
  }
  // A second copy of the chips makes each loop seamless.
  rows.forEach(track => {
    $$('.chip', track).forEach(c => {
      const copy = c.cloneNode(true);
      copy.setAttribute('aria-hidden', 'true');
      copy.tabIndex = -1;
      track.appendChild(copy);
    });
  });
  // Scroll at a fixed speed in pixels per second, so more questions do not mean faster chips.
  const speed = () => (window.innerWidth <= 640 ? 32 : 40);
  const pace = () => rows.forEach(t => { t.style.animationDuration = (t.scrollWidth / 2 / speed()) + 's'; });
  pace();
  window.addEventListener('resize', pace);
  document.fonts?.ready.then(pace);

  const cards = $$('.qa');
  let open = null;
  const show = id => {
    open = id;
    cards.forEach(c => { c.hidden = c.id !== id; });
    $$('.chip').forEach(c => c.classList.toggle('active', c.dataset.q === id));
    if (id) {
      const card = $('#' + id);
      const r = card.getBoundingClientRect();
      if (r.bottom > window.innerHeight) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };
  rows.forEach(track => track.addEventListener('click', e => {
    const chip = e.target.closest('.chip');
    if (chip) show(open === chip.dataset.q ? null : chip.dataset.q);
  }));
  $$('.qa-close').forEach(b => b.addEventListener('click', () => show(null)));
})();

/* Competitions dropdown */
(function competitions() {
  const btn = $('#comp-toggle'), grid = $('#comp-grid');
  btn.addEventListener('click', () => {
    const expand = btn.getAttribute('aria-expanded') !== 'true';
    btn.setAttribute('aria-expanded', String(expand));
    grid.hidden = !expand;
  });
})();

/* BibTeX copy */
$('#copy-bib').onclick = () => {
  navigator.clipboard.writeText($('#bibtex').textContent);
  $('#copy-bib').textContent = 'Copied';
  setTimeout(() => ($('#copy-bib').textContent = 'Copy'), 1500);
};
